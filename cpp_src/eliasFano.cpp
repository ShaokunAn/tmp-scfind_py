#include <cmath>
#include <iterator>
#include <pybind11/stl.h>
#include <numeric>
#include <functional>
#include <exception>
#include <stdexcept>
//#include <chrono>

#include "EliasFano.h"
#include "Serialization.h"
#include "QueryScore.h"
#include "typedef.h"
#include "utils.h"

CellMeta::CellMeta() : reads(0), features(0)
{
}

GeneMeta::GeneMeta() : total_reads(0)
{
}

void GeneMeta::merge(const GeneMeta &other)
{
  this->total_reads += other.total_reads;
}

CellID::CellID(CellTypeID ct, int cid) : cell_type(ct), cell_id(cid)
{
}

void EliasFanoDB::clearDB()
{
  // Clear the database
  index.clear();
  cell_types.clear();
  inverse_cell_type.clear();
}

int EliasFanoDB::setQuantizationBits(unsigned int qbvalue)
{

  if (ef_data.empty() and qbvalue < 32)
  {
    quantization_bits = qbvalue;
  }
  else
  {
    std::cerr << "Quantized bits not set, DB not empty or qbvalue to high!" << std::endl;
    return 1;
  }

  if (qbvalue > 10)
  {
    std::cerr << "Setting to high value may be a performance hog in retrieving cell expression" << std::endl;
  }
  return 0;
}

unsigned int EliasFanoDB::getQuantizationBits() const
{
  return this->quantization_bits;
}

int EliasFanoDB::loadByteStream(const py::bytes &stream)
{
  clearDB();
  SerializationDB ser;

  ser.loadByteStream(stream);
  ser.deserializeDB(*this);
  return 0;
}

py::bytes EliasFanoDB::getByteStream() const
{
  SerializationDB ser;
  py::bytes byte_stream = ser.getByteStream(*this);

  // Convert std::vector<unsigned char> to py::bytes
  return byte_stream;
}
long EliasFanoDB::eliasFanoCodingNoExpr(const std::vector<int> &ids, int total_cells)
{
  if (ids.empty())
  {
    return -1;
  }

  EliasFano ef;
  ef.l = int(log2(total_cells / static_cast<float>(ids.size())) + 0.5) + 1;
  ef.idf = log2(total_cells / static_cast<float>(ids.size()));
  int l = ef.l;

  int prev_indexH = 0;
  ef.L.resize(l * ids.size(), false);

  BoolVec::iterator l_iter = ef.L.begin();
//  Quantile lognormalcdf(const std::vector<int>& ids, const py::array_t<double>& v, unsigned int bits, bool raw_counts = true);
  ef.expr = Quantile();

  for (auto expr = ids.begin(); expr != ids.end(); ++expr)
  {
    BitSet32 c = int2bin_bounded(*expr, l);

    for (int i = 0; i < l; i++, ++l_iter)
    {
      *l_iter = c.second[i];
    }
    unsigned int upper_bits = (*expr >> l);
    unsigned int m = ef.H.size() + upper_bits - prev_indexH + 1;
    prev_indexH = upper_bits;
    ef.H.resize(m, false);
    ef.H[m - 1] = true;
  }

  // std::cout<<"before push back: ef_data.size() = " << ef_data.size() << std::endl;
  ef_data.push_back(ef);
  // std::cout<<"after push back: ef_data.size() = " << ef_data.size() << std::endl;

  return ef_data.size() - 1;
}

long EliasFanoDB::eliasFanoCoding(const std::vector<int> &ids, const std::vector<double> &values)
{
  if (ids.empty())
  {
    return -1;
  }
  int items = values.size();

  EliasFano ef;
  ef.l = int(log2(items / static_cast<float>(ids.size())) + 0.5) + 1;
  ef.idf = log2(items / static_cast<float>(ids.size()));
  int l = ef.l;

  int prev_indexH = 0;
  ef.L.resize(l * ids.size(), false);

  BoolVec::iterator l_iter = ef.L.begin();
//  Quantile lognormalcdf(const std::vector<int>& ids, const py::array_t<double>& v, unsigned int bits, bool raw_counts = true);
  ef.expr = lognormalcdf(ids, values, this->quantization_bits);

  for (auto expr = ids.begin(); expr != ids.end(); ++expr)
  {
    BitSet32 c = int2bin_bounded(*expr, l);

    for (int i = 0; i < l; i++, ++l_iter)
    {
      *l_iter = c.second[i];
    }
    unsigned int upper_bits = (*expr >> l);
    unsigned int m = ef.H.size() + upper_bits - prev_indexH + 1;
    prev_indexH = upper_bits;
    ef.H.resize(m, false);
    ef.H[m - 1] = true;
  }

  std::cout<<"before push back: ef_data.size() = " << ef_data.size() << std::endl;
  ef_data.push_back(ef);
  std::cout<<"after push back: ef_data.size() = " << ef_data.size() << std::endl;

  return ef_data.size() - 1;
}

std::vector<int> EliasFanoDB::eliasFanoDecoding(const EliasFano &ef) const
{
  // This step inflates the vector by a factor of 8
  std::vector<char> H;
  std::vector<int> ids(ef.L.size() / ef.l);
  H.reserve(ef.H.size());
  H.insert(H.end(), ef.H.begin(), ef.H.end());

  unsigned int H_i = 0;
  // Warning: Very very dodgy I might want to replace this with a check in the loop
  auto prev_it = H.begin() - 1;
  size_t i = 0;
  for (auto true_it = std::find(H.begin(), H.end(), true);
       true_it != H.end() && i < ids.size();
       true_it = std::find(true_it + 1, H.end(), true), ++i)
  {
    size_t offset = std::distance(prev_it, true_it);
    prev_it = true_it;
    H_i += offset - 1;
    int id = H_i << ef.l;
    for (unsigned short k = 0; k < ef.l; ++k)
    {
      id |= (ef.L[(i * ef.l) + k] << k);
    }
    ids[i] = id;
  }
  return ids;
}

const CellType &EliasFanoDB::getCellType(const CellTypeName &name) const
{

  auto id = this->cell_types.at(name);
  return this->inverse_cell_type.at(id);
}


const py::tuple EliasFanoDB::getCellTypeMatrix(const CellTypeName &cell_type) const
{
  const CellType ct = getCellType(cell_type);
  const CellTypeID ct_id = this->cell_types.at(cell_type);
  std::vector<GeneName> feature_names;

  // Feature number will be the feature names size
  for (auto const &record : index)
  {
    auto rec_it = record.second.find(ct_id);
    if (rec_it != record.second.end())
    {
      feature_names.push_back(record.first);
    }
  }

  int qb = quantization_bits;
  size_t feature_name_size = feature_names.size();

  // Determine return sparse matrix or dense matrix
  bool issparse = this->issparse;
  if (issparse)
  {
    std::vector<double> values; // non-zero values
    std::vector<ssize_t> row_indices; // row index
    std::vector<ssize_t> col_indices; // column index

    for (size_t row = 0; row < feature_name_size; ++row) {
      // for each gene in the database extract the values
      const auto &rec = getEntry(feature_names[row], cell_type);
      const auto indices_val = eliasFanoDecoding(rec);
      const auto exp_val = decompressValues(rec.expr, qb);

      // check if indices_val and exp_val have the same amount of elements
      if (indices_val.size() != exp_val.size()) {
          std::cerr << "not equal number of genes" << std::endl;
          std::cerr << feature_names[row] << std::endl;
          continue;
      }

      // store non-zero values and indices
      for (size_t i = 0; i < indices_val.size(); ++i) {
          values.push_back(exp_val[i]);
          row_indices.push_back(indices_val[i]-1);  // store the cell index of non-zero values
          col_indices.push_back(row);  // store the gene id
      }
    }

    int n_cells = ct.total_cells;

    // return a python tuple including sparse matrix information and gene names
    return py::make_tuple(values, row_indices, col_indices, n_cells, feature_names);
  }
  else
  {
    // Initialize matrix
    py::array_t<double> mat(feature_name_size * ct.total_cells);
    std::vector<ssize_t> shape = { static_cast<ssize_t>(ct.total_cells), static_cast<ssize_t>(feature_name_size) };
    mat.resize(shape);

    // for the sparse expression  vector matrix get the indices and deconvolute the quantized values
    for (size_t col = 0; col < feature_name_size; ++col){
      // for each gene in the database extract the values
      const auto &rec = getEntry(feature_names[col], cell_type);
      const auto indices_val = eliasFanoDecoding(rec);
      const auto exp_val = decompressValues(rec.expr, qb);

      if (indices_val.size() != exp_val.size()) {
        std::cerr << "not equal number of genes" << std::endl;
        continue;
      }

      std::vector<double> na_vec(ct.total_cells);
      auto exp_it = exp_val.begin();

      if (exp_val.size() != indices_val.size()) {
        std::cerr << "Sparse vector representation mismatch" << std::endl;
        std::cerr << feature_names[col] << std::endl;
        continue;
      }

      for (auto const &index : indices_val)
      {
        na_vec[index - 1] = (*exp_it);
        ++exp_it;
      }

      for (int row = 0; row < mat.shape(0); ++row) {
          *mat.mutable_data(row, col) = na_vec[row];
      }
    }
    return py::make_tuple(mat, feature_names);
  }

}

const EliasFano &EliasFanoDB::getEntry(const GeneName &gene_name, const CellTypeName &cell_type) const
{
  try
  {
    return this->ef_data.at(this->index.at(gene_name).at(this->cell_types.at(cell_type)));
  }
  catch (const std::out_of_range &e)
  {
    std::cerr << e.what() << std::endl;
    auto g_it = index.find(gene_name);
    if (g_it == index.end())
    {
      std::cerr << gene_name << "Gene not found" << std::endl;
    }
    auto ct_it = this->cell_types.find(cell_type);

    if (ct_it == this->cell_types.end())
    {
      std::cerr << "Cell type" << cell_type << " not found in the database" << std::endl;
    }
    else
    {
      auto ef_it = g_it->second.find(ct_it->second);

      if (ef_it == g_it->second.end())
      {
        std::cerr << "Cell type " << cell_type << " not found for gene " << gene_name << std::endl;
      }
    }
    throw std::invalid_argument("Unable to retrieve entry from database");
  }
}

// constructor
EliasFanoDB::EliasFanoDB() : warnings(0),
                             total_cells(0),
                             quantization_bits(2)
{
}

//EliasFanoDB::EliasFanoDB(SEXPREC *&obj)
//{
//}

int EliasFanoDB::queryZeroGeneSupport(const py::list &datasets) const
{
  int zs = 0;
  for (auto const &g : this->index)
  {
    py::list single_string_list;
    single_string_list.append(g.first);
    auto cell_support = this->totalCells(single_string_list, datasets);
    // std::vector<int> cell_support =  [].cast<std::vector<int>>();
    if (cell_support[0].cast<int>() == 0)
    {
      zs++;
      std::cerr << "Gene " << g.first << " found no support with " << g.second.size() << " cell types" << std::endl;
    }
  }
  return zs;
}

// This is invoked on slices of the expression matrix of the dataset
long EliasFanoDB::encodeMatrix(const std::string &cell_type_name, const py::object &csr_mat, const py::list &cell_type_genes)
{
  // Change python sparse matrix to arma::sp_mat
  const arma::sp_mat gene_matrix = csr_to_sp_mat(csr_mat);

  CellType cell_type;
  cell_type.name = cell_type_name;
  cell_type.total_cells = gene_matrix.n_rows;

  int cell_type_id = insertNewCellType(cell_type);

  // Increase the cell number present in the index
  this->total_cells += gene_matrix.n_rows;
  this->issparse = true;

  // Store the metadata for the cell
  std::vector<CellMeta> current_cells(gene_matrix.n_rows);

  for (unsigned int gene_col = 0; gene_col < gene_matrix.n_cols; ++gene_col)
  {
    const arma::sp_colvec& expression_vector = gene_matrix.col(gene_col);

    std::vector<double> denseVector(gene_matrix.n_rows);
    for (auto it = expression_vector.begin(); it != expression_vector.end(); ++it)
    {
        denseVector[it.row()] = it.value();
    }

    std::deque<int> sparse_index;

    for (size_t cell_idx =0; cell_idx < gene_matrix.n_rows; ++cell_idx)
    {
      double value = denseVector[cell_idx];

      if (value > 0)
      {
        current_cells[cell_idx].reads += value;
        current_cells[cell_idx].features++;
        sparse_index.push_back(cell_idx + 1); // 1 based indexing
      }
    }

    if (sparse_index.empty())
    {
      continue;
    }

    GeneName geneNameKey = cell_type_genes[gene_col].cast<std::string>();
    auto gene_it = this->genes.insert({geneNameKey, GeneMeta()}).first;

//    auto gene_it = this->genes.insertert(std::make_pair(cell_type_genes[gene_row], GeneMeta())).first;
    auto db_entry = this->index.insert({geneNameKey, GeneContainer()}).first;
//    auto db_entry = this->index.insert(std::make_pair(cell_type_genes[gene_row], GeneContainer())).first;

    std::vector<int> ids(sparse_index.begin(), sparse_index.end());

    gene_it->second.total_reads += ids.size();

    // auto ef_index = eliasFanoCoding(ids, denseVector);
    auto ef_index = eliasFanoCodingNoExpr(ids, cell_type.total_cells);
    if (ef_index != -1)
    {
      db_entry->second.insert(std::make_pair(cell_type_id, ef_index));
    }
  }

  int i = 0; // 1 based indexing
  for (auto const &cell : current_cells)
  {
    if (cell.reads == 0)
    {
      std::cerr << "Vector of zeros detected for cell " << cell_type_name << " " << i << std::endl;
    }
    this->cells.insert({CellID(cell_type_id, ++i), cell});
  }

  return 0;
}


// Encode dense matrix
long EliasFanoDB::encodeMatrix_dense(const std::string &cell_type_name, const py::array_t<double> &dense_mat, const py::list &cell_type_genes)
{
  CellType cell_type;
  cell_type.name = cell_type_name;
  int total_cells = static_cast<int>(dense_mat.shape(0));
  int total_genes = static_cast<int>(dense_mat.shape(1));
  cell_type.total_cells = total_cells;

  int cell_type_id = insertNewCellType(cell_type);

  // Increase the cell number present in the index
  this->total_cells += total_cells;
  this->issparse = false;

  // Store the metadata for the cell
  std::vector<CellMeta> current_cells(total_cells);

  for (int gene_col = 0; gene_col < total_genes; ++gene_col)
  {
    // auto expression_vector = dense_mat.unchecked<2>()(py::slice(0, total_cells, 1), gene_col);
    std::vector<double> denseVector(total_cells, 0.0);

    auto dense_mat_proxy = dense_mat.unchecked<2>();
    for (int i = 0; i < total_cells; ++i) {
        denseVector[i] = dense_mat_proxy(i, gene_col);
    }

    std::deque<int> sparse_index;

    for (int cell_idx = 0; cell_idx < total_cells; ++cell_idx)
    {
      double value = denseVector[cell_idx];

      if (value > 0)
      {
        current_cells[cell_idx].reads += value;
        current_cells[cell_idx].features++;
        sparse_index.push_back(cell_idx + 1); // 1 based indexing
      }
    }

    if (sparse_index.empty())
    {
      continue;
    }

    GeneName geneNameKey = cell_type_genes[gene_col].cast<std::string>();
    auto gene_it = this->genes.insert({geneNameKey, GeneMeta()}).first;

//    auto gene_it = this->genes.insertert(std::make_pair(cell_type_genes[gene_row], GeneMeta())).first;
    auto db_entry = this->index.insert({geneNameKey, GeneContainer()}).first;
//    auto db_entry = this->index.insert(std::make_pair(cell_type_genes[gene_row], GeneContainer())).first;

    std::vector<int> ids(sparse_index.begin(), sparse_index.end());

    gene_it->second.total_reads += ids.size();

    // auto ef_index = eliasFanoCoding(ids, denseVector);
    auto ef_index = eliasFanoCodingNoExpr(ids, cell_type.total_cells);
    if (ef_index != -1)
    {
      db_entry->second.insert(std::make_pair(cell_type_id, ef_index));
    }
  }

  int i = 0; // 1 based indexing
  for (auto const &cell : current_cells)
  {
    if (cell.reads == 0)
    {
      std::cerr << "Vector of zeros detected for cell " << cell_type_name << " " << i << std::endl;
    }
    this->cells.insert({CellID(cell_type_id, ++i), cell});
  }

  return 0;
}

const std::vector<EliasFanoDB::CellTypeName> EliasFanoDB::getCellTypes() const
{
  return this->_getCellTypes();
}

const std::vector<EliasFanoDB::CellTypeName> EliasFanoDB::_getCellTypes(const std::vector<std::string> &datasets) const
{
  auto cts = this->_getCellTypes();
  std::vector<EliasFanoDB::CellTypeName> results;
  results.reserve(cts.size());
  std::copy_if(cts.begin(),
               cts.end(),
               std::back_inserter(results),
               [&datasets](const CellTypeName &ct)
               {
                 auto it = ct.find_first_of(".");
                 if (it != std::string::npos)
                 {
                   return std::find(datasets.begin(), datasets.end(), ct.substr(0, it)) != datasets.end();
                 }
                 return false;
               });

  return results;
}

py::dict EliasFanoDB::geneSupportInCellTypes(const py::list &gene_names, const py::list &datasets_active) const
{
  std::vector<std::string> datasetsVector = py::cast<std::vector<std::string>>(datasets_active);

  auto cell_types = this->_getCellTypes(datasetsVector);
  auto genes = gene_names.cast<std::vector<EliasFanoDB::GeneName>>();
  py::dict results;

  for (const auto &g : genes)
  {
    std::map<std::string, int> gene_results;
    for (auto const &ct : cell_types)
    {
      // Querying cell types
      int size = 0;
      try
      {
        const auto r = this->ef_data.at(this->index.at(g).at(this->cell_types.at(ct)));
        size = r.getSize();
      }
      catch (const std::out_of_range &e)
      {
        continue;
      }

      gene_results[ct] = size;
    }
    results[py::str(g)] = gene_results;

  }
  return results;
}

py::list EliasFanoDB::total_genes() const
{
  py::list t;
  for (auto &d : index)
  {
    t.append(d.first);
  }
  return t;
}

py::dict EliasFanoDB::totalCells(const py::list &genes,
                                            const py::list &datasets_active) const
{
  py::dict t;
  std::vector<std::string> datasets = datasets_active.cast<std::vector<std::string>>();
  std::vector<std::string> str = genes.cast<std::vector<std::string>>();

  // Building the inverse index for index cell_types
  std::unordered_map<CellTypeID, CellTypeName> inv_ct;
  for (auto const &ct : this->cell_types)
  {
    inv_ct[ct.second] = ct.first;
  }

  int count = 0;
  for (auto const &g : str)
  {
    auto git = this->index.find(g);
    if (git != this->index.end())
    {
      for (auto const &ct : git->second)
      {
        const std::string &ct_name = inv_ct[ct.first];
        std::string ct_dataset = ct_name.substr(0, ct_name.find("."));
        auto find_dataset = std::find(datasets.begin(), datasets.end(), ct_dataset);
        // check if the cells are in active datasets
        if (find_dataset == datasets.end())
        {
          continue;
        }
        count +=this->ef_data[ct.second].getSize();
      }
    }
    t.attr("setdefault")(g, count);
  }
  return t;
}

EliasFanoDB::CellTypeIndex EliasFanoDB::getCellTypeIDs(const std::set<std::string> &datasets) const
{
  CellTypeIndex cts;
  for (auto const &ct : this->inverse_cell_type)
  {
    auto index = ct.name.find_last_of(".");
    if (index != std::string::npos)
    {
      auto dataset = ct.name.substr(0, index);
      const auto it = datasets.find(dataset);
      if (it != datasets.end())
      {
        cts[ct.name] = this->cell_types.find(ct.name)->second;
      }
    }
  }
  return cts;
}

int EliasFanoDB::cellsInDB() const
{
  return this->total_cells;
}

int EliasFanoDB::getTotalCells(const py::list &datasets) const
{
  std::vector<std::string> act = datasets.cast<std::vector<std::string>>();
  std::set<std::string> act_set(act.begin(), act.end());
  CellTypeIndex active_cell_types = getCellTypeIDs(act_set);
  int total_number_of_cells = 0;
  for (auto const &ct : active_cell_types)
  {
    total_number_of_cells += this->inverse_cell_type[ct.second].total_cells;
  }
  return total_number_of_cells;
}

int EliasFanoDB::numberOfCellTypes(const py::list &datasets) const
{
  std::vector<std::string> act = datasets.cast<std::vector<std::string>>();
  std::set<std::string> act_set(act.begin(), act.end());
  CellTypeIndex active_cell_types = getCellTypeIDs(act_set);

  return active_cell_types.size();
}

py::list EliasFanoDB::getCellTypeSupport(py::list &cell_types)
{
  std::vector<std::string> cts = cell_types.cast<std::vector<std::string>>();
    std::vector<int> ct_support;
    ct_support.reserve(cts.size());
    for (const std::string &ct : cts)
    {
      auto cit = this->cell_types.find(ct);
      int num;
      if (cit != this->cell_types.end()){
        num = this->inverse_cell_type[cit->second].total_cells;
        ct_support.push_back(num);
      }else{
        num = 0;
        ct_support.push_back(num);
      }
    }
  return py::cast(ct_support);
  
}

py::dict EliasFanoDB::queryGenes(const py::list &gene_names, const py::list &datasets_active) const
{
  py::dict t;
  for (const auto &gene_name_obj : gene_names)
  {

    std::string gene_name = gene_name_obj.cast<std::string>();
    py::dict cell_types;

    if (index.find(gene_name) == index.end())
    {

      std::cout << "Gene " << gene_name << " not found in the index " << std::endl;
      continue;
    }
    std::vector<std::string> datasets = datasets_active.cast<std::vector<std::string>>();
    const auto &gene_meta = index.at(gene_name);
    for (auto const &dat : gene_meta)
    {
      CellType current_cell_type = this->inverse_cell_type[dat.first];

      std::string dataset = current_cell_type.name.substr(0, current_cell_type.name.find("."));
      auto ct_find = std::find(datasets.begin(), datasets.end(), dataset);

      if (ct_find == datasets.end())
      {
        continue;
      }
      std::vector<int> ids = eliasFanoDecoding(ef_data[dat.second]);
//      cell_types[current_cell_type.name] = ids;
      cell_types[py::str(current_cell_type.name)] = ids;
    }
    t[py::str(gene_name)] = cell_types;
//    t[gene_name] = cell_types;
  }

  return t;
}

size_t EliasFanoDB::dataMemoryFootprint() const
{
  size_t bytes = 0;
  for (auto &d : ef_data)
  {
    bytes += int((d.H.size() / 8) + 1);
    bytes += int((d.L.size() / 8) + 1);
    bytes += int((d.expr.quantile.size() / 8) + 12);
  }
  bytes += ef_data.size() * 32; // overhead of l idf and deque struct
  return bytes;
}

size_t EliasFanoDB::quantizationMemoryFootprint() const
{
  size_t bytes = 0;
  for (auto &d : ef_data)
  {
    bytes += int((d.expr.quantile.size() / 8) + 12);
  }
  bytes += ef_data.size() * 32; // overhead of l idf and deque struct
  return bytes;
}

size_t EliasFanoDB::dbMemoryFootprint() const
{
  size_t bytes = dataMemoryFootprint();
  std::cout << "bytes="<<bytes<<std::endl;
  std::cout << "Raw elias Fano Index size " << bytes / (1024 * 1024) << "MB" << std::endl;

  // GeneIndex genes GeneExpressionDB
  for (auto const &d : index)
  {
    // One for each
    bytes += d.first.size() * 2;
    bytes += d.second.size() * 4;
  }

  bytes += index.size() * (sizeof(GeneMeta) + 8);

  // CellIndex cells
  bytes += cells.size() * (sizeof(CellID) + sizeof(CellMeta) + 4);

  // CellTypeIndex cell_types std::deque<cellType> inverse_cell_type
  for (auto const &c : cell_types)
  {
    bytes += (c.first.size() * 2) + 4 + sizeof(CellType);
  }

  bytes += 16;

  return bytes;
}

py::list EliasFanoDB::getGenesInDB() const
{
  std::vector<std::string> gene_names;
  gene_names.reserve(this->genes.size());
  for (auto const &g : this->genes)
  {
    gene_names.push_back(g.first);
  }
  return py::cast(gene_names);
}

py::dict EliasFanoDB::findCellTypes(const py::list &gene_names, const py::list &datasets_active) const
{
  std::vector<std::string> datasets = datasets_active.cast<std::vector<std::string>>();
  std::vector<CellTypeName> cell_types_bg;
  for (auto const &ct : this->cell_types)
  {
    cell_types_bg.push_back(ct.first);
  }
  cell_types_bg.erase(std::remove_if(
                          cell_types_bg.begin(),
                          cell_types_bg.end(),
                          [&datasets](const CellTypeName &ct_name)
                          {
                            std::string ct_dataset = ct_name.substr(0, ct_name.find("."));
                            return std::find(datasets.begin(), datasets.end(), ct_dataset) == datasets.end();
                          }),
                      cell_types_bg.end());

  return _findCellTypes(gene_names.cast<std::vector<std::string>>(), cell_types_bg);
}

// TODO(Nikos) REFACTOR
// And query
py::dict EliasFanoDB::_findCellTypes(const std::vector<GeneName> &gene_names, const std::vector<EliasFanoDB::CellTypeName> &cell_types_bg) const
{

  // Store the results here
  py::dict t;
  std::vector<GeneName> genes(gene_names);

  // Remove genes not found in index
  genes.erase(std::remove_if(genes.begin(), genes.end(), [&](const GeneName &g)
                             {
                                                          auto is_missing = (index.find(g) == index.end());
                                                          if (is_missing)
                                                            std::cerr << g << " is ignored, not found in the index"<< std::endl;
                                                          return is_missing; }),
              genes.end());

  // Get Cell types that have all the genes present
  std::vector<CellTypeName> cts = cell_types_bg;
  std::vector<const GeneContainer *> gene_set;
  for (auto const &g : genes)
  {
    gene_set.push_back(&(this->index.at(g)));
  }
  cts.erase(std::remove_if(cts.begin(), cts.end(), [&](const CellTypeName &ct)
                           {
                                                    CellTypeID cid = this->cell_types.at(ct);
                                                    for( auto const& g : gene_set)
                                                      if (g->find(cid) == g->end())
                                                        return true;
                                                    return false; }),
            cts.end());

  for (auto const &ct : cts)
  {
    auto last_intersection = eliasFanoDecoding(getEntry(*(genes.begin()), ct));
    std::vector<int> curr_intersection;
    curr_intersection.reserve(last_intersection.size());
    for (std::size_t i = 1; i < genes.size(); ++i)
    {
      std::vector<int> cells = eliasFanoDecoding(getEntry(genes.at(i), ct));
      std::set_intersection(
          cells.begin(),
          cells.end(),
          last_intersection.begin(),
          last_intersection.end(),
          std::back_inserter(curr_intersection));
      std::swap(last_intersection, curr_intersection);
      curr_intersection.clear();
      curr_intersection.reserve(last_intersection.size());
      if (last_intersection.empty())
      {
        break;
      }
    }
    if (not last_intersection.empty())
      t[py::str(ct)] = py::cast(last_intersection);
  }
  return t;
}

// that casts the results into native python data structures
py::dict EliasFanoDB::findMarkerGenes(const py::list &gene_list, py::list datasets_active, bool exhaustive, int user_cutoff) const
{
  std::vector<std::string> query;
  std::vector<double> query_scores;
  std::vector<double> query_tfidf;
  std::vector<int> query_cell_type_cardinality;
  std::vector<int> query_cell_cardinality;
  std::vector<int> query_gene_cardinality;

  // Perform an OR query on the database as a first step
  const py::dict genes_results = queryGenes(gene_list, datasets_active);

  QueryScore qs;
  qs.estimateExpression(genes_results, *this, datasets_active);

  unsigned int min_support_cutoff = 0;
  if (user_cutoff < 0)
  {
    min_support_cutoff = qs.geneSetCutoffHeuristic();
  }
  else
  {
    min_support_cutoff = user_cutoff;
  }
  std::set<Pattern> patterns = exhaustive ? exhaustiveFrequentItemsetMining(genes_results, min_support_cutoff) : FPGrowthFrequentItemsetMining(genes_results, min_support_cutoff);

  // Iterate through the calculated frequent patterns
  for (auto const &item : patterns)
  {
    const auto &gene_set = item.first;

    // We do not care for queries with cardinality less than 2
    if (gene_set.size() < 2)
    {
      continue;
    }
    std::string view_string = str_join(std::vector<Item>(gene_set.begin(), gene_set.end()), ",");
    // cell_type_relevance
    query_cell_cardinality.push_back(item.second);
    query_tfidf.push_back(qs.cell_tfidf(*this, gene_set));

    // other fields
    query_gene_cardinality.push_back(gene_set.size());
    query.push_back(view_string);

  }

  // Dump the list
  py::dict result;
  result["Genes"] = py::cast(query_gene_cardinality);
  result["Query"] = py::cast(query);
  result["TF-IDF"] = py::cast(query_tfidf);
  result["Number of Cells"] = py::cast(query_cell_cardinality);

  return result;
}

const std::set<std::string> EliasFanoDB::_getValidCellTypes(std::vector<std::string> universe) const
{
  std::set<std::string> active_cell_types;
  std::vector<std::string> db_cell_types(this->_getCellTypes());
  std::sort(universe.begin(), universe.end());
  std::sort(db_cell_types.begin(), db_cell_types.end());
  std::set_intersection(
      db_cell_types.begin(),
      db_cell_types.end(),
      universe.begin(),
      universe.end(),
      std::inserter(active_cell_types, active_cell_types.begin()));
  if (universe.size() != active_cell_types.size())
  {
    std::vector<std::string> cts_not_found;
    std::set_difference(
        universe.begin(),
        universe.end(),
        active_cell_types.begin(),
        active_cell_types.end(),
        std::back_inserter(cts_not_found));
    for (auto const &ct : cts_not_found)
    {
      std::cerr << "Ignoring cell type " << ct << " Not found in DB" << std::endl;
    }
  }

  return active_cell_types;
}

py::dict EliasFanoDB::findCellTypeMarkers(const py::list &cell_types, const py::list &background) const
{
  std::vector<GeneName> gene_set;
  gene_set.reserve(this->genes.size());
  for (auto const &g : this->genes)
  {
    gene_set.push_back(g.first);
  }
  return _findCellTypeMarkers(cell_types, background, gene_set);
}

py::dict EliasFanoDB::_findCellTypeMarkers(const py::list &cell_types, const py::list &background, const std::vector<EliasFanoDB::GeneName> &gene_set, int mode) const
{
  std::vector<std::string>
      bk_cts(background.cast<std::vector<std::string>>()),
      cts(cell_types.cast<std::vector<std::string>>()),
      genes, df_cell_type;

  std::vector<int> tp, fp, tn, fn;
  std::vector<float> precision, recall, f1;

  for (const auto &ct : cts)
  {
    auto marker_genes = this->_cellTypeScore(ct, bk_cts, gene_set, mode);
    if (marker_genes.empty())
    {
      std::cerr << "Marker genes could not be found for cell type " << ct << std::endl;
      continue;
    }

    for (const auto& t : marker_genes) 
    {
      const CellTypeMarker &ctm = t.second;
      genes.push_back(t.first);
      df_cell_type.push_back(ct);
      tp.push_back(ctm.tp);
      fp.push_back(ctm.fp);
      tn.push_back(ctm.tn);
      fn.push_back(ctm.fn);
      precision.push_back(ctm.precision());
      recall.push_back(ctm.recall());
      f1.push_back(ctm.f1());
    }
  }

  py::dict result;
  result["cellType"] = py::cast(df_cell_type);
  result["genes"] = py::cast(genes);
  result["tp"] = py::cast(tp);
  result["fp"] = py::cast(fp);
  result["fn"] = py::cast(fn);
  result["precision"] = py::cast(precision);
  result["recall"] = py::cast(recall);
  result["f1"] = py::cast(f1);

  return result;
}

py::dict EliasFanoDB::evaluateCellTypeMarkersAND(const py::list &cell_types,
                                                        const py::list &gene_set,
                                                        const py::list &background)
{
  return _findCellTypeMarkers(cell_types, background, gene_set.cast<std::vector<GeneName>>(), AND);
}

py::dict EliasFanoDB::evaluateCellTypeMarkers(const py::list &cell_types,
                                                     const py::list &gene_set,
                                                     const py::list &background)
{
  return _findCellTypeMarkers(cell_types, background, gene_set.cast<std::vector<GeneName>>(), ALL);
}

std::map<EliasFanoDB::GeneName, CellTypeMarker> EliasFanoDB::_cellTypeScore(const std::string &cell_type, const std::vector<std::string> &universe, const std::vector<EliasFanoDB::GeneName> &gene_names, int mode) const
{
  auto ct_it = this->cell_types.find(cell_type);
    if (ct_it == this->cell_types.end())
  {
    std::cerr << "Cell type " << cell_type << " not found. exiting..." << std::endl;
    return std::map<std::string, CellTypeMarker>();
  }

  const CellTypeID cell_type_id = ct_it->second;
  int total_cells_in_ct = this->inverse_cell_type[cell_type_id].total_cells;
  
  const auto active_cell_types = this->_getValidCellTypes(universe);

  // Calculate background universe total cells
  const std::deque<CellType> &all_cts = this->inverse_cell_type;
  const CellTypeIndex &cts_index = this->cell_types;
  const int act_total_cells = std::accumulate(active_cell_types.begin(),
                                              active_cell_types.end(),
                                              0,
                                              [&all_cts, &cts_index](const int &sum, const CellTypeName &name)
                                              {
                                                const auto ct_id = cts_index.find(name);
                                                return sum + all_cts[ct_id->second].total_cells;
                                              });

  std::map<GeneName, CellTypeMarker> scores;

  if (mode == ALL)
  {
    CellTypeMarker ctm_template = {0, 0, 0, 0};
    for (auto const &gene_name : gene_names)
    { 
      const auto index_it = this->index.find(gene_name);
      if (index_it == this->index.end())
      {
        std::cerr << "Gene " << gene_name << " not found in the database, Ignoring... " << std::endl;
        continue;
      }

      const auto &gene_entry = *index_it;
      // Make sure the cell type is in then batch
      auto ctm = gene_entry.second.find(cell_type_id);
      if (ctm == gene_entry.second.end())
      {
        continue;
      }



      auto dit = scores.insert(std::make_pair(gene_entry.first, ctm_template));
      CellTypeMarker &gene_ctm_score = dit.first->second;
      // this->cellTypeMarkerGeneMetrics(gene_ctm_score);
      const EliasFano &ex_vec = this->ef_data[ctm->second];
      // std::cout << ctm->second << " size: " << this->ef_data[ctm->second]. << std::endl;
      int cells_in_ct = ex_vec.getSize();
      gene_ctm_score.tp = cells_in_ct;
      gene_ctm_score.fn = total_cells_in_ct - gene_ctm_score.tp;
      for (auto const &ct : gene_entry.second)
      {
        auto bct_it = active_cell_types.find(all_cts[ct.first].name);
        // if we are not interested in the cell type continue
        if (bct_it == active_cell_types.end())
        {
          continue;
        }

        int bkg_cell_number = this->ef_data[ct.second].getSize();
        gene_ctm_score.fp += bkg_cell_number;
      }

      // total cells in the universe - total cells expressing gene - total cells not in cell type
      gene_ctm_score.tn = act_total_cells - gene_ctm_score.fp - total_cells_in_ct;

      // subtract the cells in the cell_type of interest
      gene_ctm_score.fp -= cells_in_ct;
    }
  }
  return scores;
}

const std::vector<EliasFanoDB::CellTypeName> EliasFanoDB::_getCellTypes() const
{
  std::vector<CellTypeName> cts;
  cts.reserve(this->cell_types.size());
  for (auto const &ct : this->inverse_cell_type)
  {
    cts.push_back(ct.name);
  }
  return cts;
}

int EliasFanoDB::dbSize() const
{
  std::cout << index.size() << "genes in the DB" << std::endl;
  return ef_data.size();
}

std::vector<int> EliasFanoDB::decode(int index) const
{
  if (index >= dbSize())
  {
    std::cerr << "Invalid index for database with size " << dbSize() << std::endl;
    return std::vector<int>();
  }
  return eliasFanoDecoding(ef_data.at(index));
}

int EliasFanoDB::insertNewCellType(const CellType &cell_type)
{
  auto ct_it = this->cell_types.insert(std::make_pair(cell_type.name, this->cell_types.size()));

  if (not ct_it.second)
  {
    std::cerr << "This should not happen!! Duplicate Cell Type: " << cell_type.name << std::endl;
  }
  else
  {
    this->inverse_cell_type.push_back(cell_type);
  }

  return ct_it.first->second;
}


int EliasFanoDB::mergeDB(const EliasFanoDB &db)
{
  EliasFanoDB extdb(db);
  if (extdb.getQuantizationBits() != this->getQuantizationBits())
  {
    std::cerr << "Can not perform merging.. Quantization bits are not equal in the two databases. Please fix" << std::endl;
    return 1;
  }
  // the DB will grow by this amount of cells
  this->total_cells += extdb.total_cells;

  // Insert new cell types in the database
  for (auto const &ct : extdb.inverse_cell_type)
  {
    insertNewCellType(ct);
  }

  // Iterate through the data model
  for (auto &gene : extdb.index)
  {
    // Update cell counts for the individual gene
    auto gene_it = this->genes.insert({gene.first, extdb.genes[gene.first]});

    if (not gene_it.second)
    {
      gene_it.first->second.merge(extdb.genes[gene.first]);
    }

    for (auto &ct : gene.second)
    {
      int new_id = ef_data.size();
      // Push the new elias fano index in the database
      ef_data.push_back(extdb.ef_data[ct.second]);
      // Update with the new entry
      int cell_type_id = this->cell_types[extdb.inverse_cell_type[ct.first].name];
      index[gene.first][cell_type_id] = new_id;
    }
  }

  for (auto const &cell : extdb.cells)
  {

    CellID clone(cell.first);
    int old_cell_type_id = clone.cell_type;
    // Trace the new entry in the database
    int new_cell_type_id = this->cell_types[extdb.inverse_cell_type[old_cell_type_id].name];
    clone.cell_type = new_cell_type_id;
    this->cells.insert({clone, cell.second});
  }

  return 0;
}


void EliasFanoDB::updateEliasFanoNoExpr(EliasFano& ef, const std::vector<int>& ids, int total_cells) {
  
  // 更新EliasFano编码
  ef.l = int(log2(total_cells / static_cast<float>(ids.size())) + 0.5) + 1;
  ef.idf = log2(total_cells / static_cast<float>(ids.size()));
  int l = ef.l;
  
  // 清除现有的L和H向量
  ef.L.clear();
  ef.H.clear();

  // 重新设置L向量大小
  ef.L.resize(l * ids.size(), false);

  BoolVec::iterator l_iter = ef.L.begin();
  int prev_indexH = 0;
  for (auto expr = ids.begin(); expr != ids.end(); ++expr) {
      BitSet32 c = int2bin_bounded(*expr, l);

      for (int i = 0; i < l; i++, ++l_iter) {
          *l_iter = c.second[i];
      }
      unsigned int upper_bits = (*expr >> l);
      unsigned int m = ef.H.size() + upper_bits - prev_indexH + 1;
      prev_indexH = upper_bits;
      ef.H.resize(m, false);
      ef.H[m - 1] = true;
  } 
  
  // 保持Quantile为空
  ef.expr=Quantile();
  ef.expr.mu=0.0;
  ef.expr.sigma=0.0;
  ef.expr.quantile.clear();

  
}


int EliasFanoDB::updateDB(const EliasFanoDB &db)
{
  EliasFanoDB extdb(db);
  if (extdb.getQuantizationBits() != this->getQuantizationBits())
  {
    std::cerr << "Can not perform merging.. Quantization bits are not equal in the two databases. Please fix" << std::endl;
    return 1;
  }
  
  // the DB will grow by this amount of cells in newly added data
  this->total_cells += extdb.total_cells;
  std::cout<<"total_cells: "<<this->total_cells<<std::endl;
  
  // 记录各个细胞类型的新旧ID映射和细胞ID偏移量
  std::map<CellTypeID, CellTypeID> cell_type_id_map;
  std::map<CellTypeID, int> cell_id_offsets;
  
  // 处理细胞类型 - 标识重复和新的细胞类型
  for (auto const &ct : extdb.inverse_cell_type)
  {
    auto existing_ct = this->cell_types.find(ct.name);
    if (existing_ct != this->cell_types.end())
    {
      // 找到重复的细胞类型
      CellTypeID existing_id = existing_ct->second;
      CellTypeID incoming_id = extdb.cell_types.at(ct.name);
      
      cell_type_id_map[incoming_id] = existing_id;
      
      // 使用total_cells作为max_cell_id，因为它们是等价的
      int max_cell_id = this->inverse_cell_type[existing_id].total_cells;
      cell_id_offsets[incoming_id] = max_cell_id;
      
      // 更新现有细胞类型的总细胞数
      this->inverse_cell_type[existing_id].total_cells += ct.total_cells;
    }
    else
    {
      // 新的细胞类型
      int new_ct_id = insertNewCellType(ct);
      cell_type_id_map[extdb.cell_types.at(ct.name)] = new_ct_id;
      cell_id_offsets[extdb.cell_types.at(ct.name)] = 0; // 新细胞类型不需要偏移
    }
  }
  
  std::cout<<"cell_type_id_map size: "<<cell_type_id_map.size()<<std::endl;

  // 处理cells集合 - 为减少重复遍历，我们先收集所有需要添加的细胞
  for (auto const &cell : extdb.cells)
  {
    CellID old_cell_id = cell.first;
    CellTypeID old_cell_type_id = old_cell_id.cell_type;
    int old_cell_num = old_cell_id.cell_id;
    
    // 获取映射到的新细胞类型ID
    CellTypeID new_cell_type_id = cell_type_id_map[old_cell_type_id];
    
    // 对于重复的细胞类型，调整细胞ID
    int offset = cell_id_offsets[old_cell_type_id];
    int new_cell_num = old_cell_num + offset;
    
    // 创建新的CellID和相应的元数据
    CellID new_cell_id(new_cell_type_id, new_cell_num);
    this->cells.insert({new_cell_id, cell.second});
  }

  std::cout<<"for cell in extdb.cells done!"<<std::endl;
  

  // 创建映射，跟踪每个基因和细胞类型对应的新EliasFano编码
  // 键: <基因名, 细胞类型ID>, 值: <旧IDs, 新IDs>
  std::map<std::pair<GeneName, CellTypeID>, 
           std::pair<std::vector<int>, std::vector<int>>> gene_ct_ids;
  
  // 遍历输入数据库的索引，收集所有需要合并或添加的数据
  // 简化后的代码
  for (auto &gene : extdb.index)
  {
    const std::string &gene_name = gene.first;
    
    for (auto &ct : gene.second)
    {
      CellTypeID old_ct_id = ct.first;
      CellTypeID new_ct_id = cell_type_id_map[old_ct_id];
      int offset = cell_id_offsets[old_ct_id];
      
      // 提取原始IDs和表达值
      const EliasFano &ef = extdb.ef_data[ct.second];
      std::vector<int> old_ids = extdb.eliasFanoDecoding(ef);

      // if (old_ids.size() != expr_values.size()) 
      // {
      //   std::cerr << "Error: Length mismatch between IDs and expression values for gene " 
      //             << gene_name << " in cell type " << extdb.inverse_cell_type[old_ct_id].name 
      //             << ". IDs length: " << old_ids.size() 
      //             << ", Expression values length: " << expr_values.size() << std::endl;
        
      //   // 记录更多详细信息以便调试
      //   std::cerr << "Gene container size: " << gene.second.size() << std::endl;
      //   std::cerr << "Cell type ID: " << old_ct_id << std::endl;
      //   std::cerr << "EF data index: " << ct.second << std::endl;
        
      //   return 100;
        
      // }
      
      // 调整IDs以反映新的偏移量
      std::vector<int> new_ids;
      new_ids.reserve(old_ids.size());
      for (int id : old_ids)
      {
          new_ids.push_back(id + offset);
      }
      
      // 直接添加到映射中，键在新数据中总是唯一的
      auto key = std::make_pair(gene_name, new_ct_id);
      gene_ct_ids[key] = std::make_pair(old_ids, new_ids);
    }
  }

  std::cout<<"process index done!"<<std::endl;


  // 现在处理基因索引和创建新的EliasFano编码
  for (auto &element : gene_ct_ids)
  {
    auto &key = element.first;
    auto &id_pair = element.second;

    const GeneName &gene_name = key.first;
    CellTypeID cell_type_id = key.second;

    auto &old_ids = id_pair.first;
    auto &new_ids = id_pair.second;
    
    // 更新或创建基因元数据
    auto gene_it = this->genes.find(gene_name);
    if (gene_it == this->genes.end())
    {
      // 这是一个新基因
      this->genes[gene_name] = extdb.genes.at(gene_name);
    }
    else
    {
      // 这是一个现有基因，合并元数据
      gene_it->second.merge(extdb.genes.at(gene_name));
    }
    
    // 检查此基因和细胞类型组合是否已存在
    auto index_it = this->index.find(gene_name);
    if (index_it == this->index.end())
    {
      // 这是一个新基因，创建索引项
      this->index[gene_name] = EliasFanoDB::GeneContainer();
      index_it = this->index.find(gene_name);
    }
    
    // 检查此细胞类型是否已经有此基因的EliasFano编码
    auto ef_it = index_it->second.find(cell_type_id);
    if (ef_it != index_it->second.end())
    {
      // std::cout<<"found exising index"<<std::endl;
      // 已经存在编码，需要合并
      EliasFano &existing_ef = this->ef_data[ef_it->second];
      std::vector<int> existing_ids = this->eliasFanoDecoding(existing_ef);
      
      // 合并IDs
      std::vector<int> combined_ids;
      combined_ids.reserve(existing_ids.size() + new_ids.size());
      
      // 首先添加现有的数据
      combined_ids.insert(combined_ids.end(), existing_ids.begin(), existing_ids.end());
      
      // 然后添加新的数据
      combined_ids.insert(combined_ids.end(), new_ids.begin(), new_ids.end());
      
      // 创建新的EliasFano编码
      // 在updateDB函数修改
      
      int cell_type_total_cells = this->inverse_cell_type[cell_type_id].total_cells;
    
      updateEliasFanoNoExpr(existing_ef, combined_ids, cell_type_total_cells);
    }
    else
    {
      // 这个基因在此细胞类型中还没有编码，创建一个新的
      int cell_type_total_cells = this->inverse_cell_type[cell_type_id].total_cells;
      int new_ef_id = eliasFanoCodingNoExpr(new_ids, cell_type_total_cells);

      if (new_ef_id != -1)
      {
        index_it->second[cell_type_id] = new_ef_id;
        // std::cout<<"for new gene "<<gene_name<<" and cell type "<<cell_type_id<<" pair, creating new index success!"<<std::endl;
      }
    }
  }

  std::cout<<"update index done!"<<std::endl;

  return 0;
}

// pybind11 will take care of the wrapping
py::dict EliasFanoDB::getCellMeta(const std::string &ct, const int &num) const
{
  const auto ct_it = this->cell_types.find(ct);
  const CellID cid(ct_it->second, num);
  const auto cmeta_it = this->cells.find(cid);
  const CellMeta &cmeta = cmeta_it->second;

  py::dict result;
  result["total_reads"] = py::cast(cmeta.getReads()); // the total number of counts in the provided cell type of num-th cell 
  result["total_features"] = py::cast(cmeta.getFeatures()); // the number of features with non-zero expression of num-th cell

  return result;
}

py::dict EliasFanoDB::getCellTypeMeta(const std::string &ct_name) const
{
  const auto ct_it = this->cell_types.find(ct_name);
  py::dict result;
  if (ct_it == this->cell_types.end())
  {
    std::cerr << "Cell type " << ct_name << " not found in the database" << std::endl;
    result["total_cells"] = 0;
    return result;
  }else{
    const CellType &ctmeta = this->inverse_cell_type[ct_it->second];

    
    result["total_cells"] = py::cast(ctmeta.getTotalCells());

    return result;
  }
}

std::vector<py::dict> EliasFanoDB::DEGenes(const py::list &ct1, const py::list &ct2, const py::list genes_obj, const double &min_fraction)
{ 
  // use all genes if no gene list provided
  std::vector<std::string> genes;
  bool use_gene_names = !genes_obj.is_none();
  if (use_gene_names) {
      genes = genes_obj.cast<std::vector<std::string>>();
  }else{
    genes.reserve(this->genes.size());
    for (auto const &g : this->genes){
      genes.push_back(g.first);
    }
  }

  int n_genes = genes.size();
  std::vector<py::dict> results;
  results.reserve(n_genes);

  std::vector<std::string> cell_types1 = ct1.cast<std::vector<std::string>>();
  std::vector<std::string> cell_types2 = ct2.cast<std::vector<std::string>>();

  int n_1 = 0, n_2 = 0;

  for (auto ct : cell_types1)
  {
    auto cit = this->cell_types.find(ct);
    if (cit != this->cell_types.end())
    {
      n_1 += this->inverse_cell_type[cit->second].total_cells;
    }else{
      n_1 +=0;
    }
  }
  for (auto ct : cell_types2)
  {
    auto cit = this->cell_types.find(ct);
    if (cit != this->cell_types.end())
    {
      n_2 += this->inverse_cell_type[cit->second].total_cells;
    }else{
      n_2 += 0;
    }
  }

  for (int i = 0; i < n_genes; ++i){
    std::string gene = genes[i];
    int x_1 = 0, x_2 = 0;
    
    for (auto ct : cell_types1)
    {
      try
      {
        const EliasFano ef = this->ef_data.at(this->index.at(gene).at(this->cell_types.at(ct)));
        std::vector<int> cell_ids = eliasFanoDecoding(ef);
        x_1 += cell_ids.size();
      }
      catch(const std::out_of_range & e)
      {
        x_1 += 0;
      }
    }

    for (auto ct : cell_types2)
    {
      try
      {
        const EliasFano ef = this->ef_data.at(this->index.at(gene).at(this->cell_types.at(ct)));
        std::vector<int> cell_ids = eliasFanoDecoding(ef);
        x_2 += cell_ids.size();
      }
      catch(const std::out_of_range & e)
      {
        x_2 += 0;
      }
    }

    double p_1 = static_cast<double>(x_1) / n_1;
    double p_2 = static_cast<double>(x_2) / n_2;

    double p_value;
    std::string test_used;
    if (std::min({x_1, n_1-x_1, x_2, n_2-x_2}) < 5){
      p_value = fisher_exact_test(x_1, n_1-x_1, x_2, n_2-x_2);
      test_used = "Fisher";
    }else{
      double p_pooled = static_cast<double>(x_1 + x_2)/(n_1 + n_2);
      double se = std::sqrt(p_pooled * (1 - p_pooled) * (1.0/n_1 + 1.0/n_2));
      double z_score = (p_1 - p_2)/std::max(se, 1e-8); // avoid division by zero
      p_value = 2 * (1 - 0.5 * std::erfc(-std::abs(z_score) / std::sqrt(2)));
      test_used = "z-test";
    }
    
    if ((p_1 > min_fraction && p_1 > p_2) || (p_2 > min_fraction && p_2 > p_1)){
      py::dict gene_result;
      gene_result["gene"] = gene;
      gene_result["proportion_1"] = p_1;
      gene_result["proportion_2"] = p_2;
      gene_result["abs difference"] = abs(p_2-p_1);
      gene_result["p_value"] = p_value;
      gene_result["test_used"] = test_used;

      results.push_back(gene_result);
      
    }else{
      continue;
    }
  }

  return results;
}


const arma::sp_mat EliasFanoDB::csr_to_sp_mat(const py::object& csr_obj) {
  if (py::isinstance<py::array_t<int>>(csr_obj.attr("indptr")) &&
      py::isinstance<py::array_t<int>>(csr_obj.attr("indices")) &&
      py::isinstance<py::array_t<double>>(csr_obj.attr("data"))) {
    py::tuple shape = csr_obj.attr("shape").cast<py::tuple>();
    size_t nrows = shape[0].cast<size_t>();

    // Get csr_matrix data, indices, and indptr
    py::array_t<int> indptr = csr_obj.attr("indptr").cast<py::array_t<int>>();
    py::array_t<int> indices = csr_obj.attr("indices").cast<py::array_t<int>>();
    py::array_t<double> data = csr_obj.attr("data").cast<py::array_t<double>>();

    int* p_indptr = indptr.mutable_data();
    int* p_indices = indices.mutable_data();
    double* p_data = data.mutable_data();

    size_t nnz = data.size();  // number of non-zero elements

    arma::umat locations(2, nnz);
    arma::vec values(nnz);

    for (size_t k = 0, i = 0; i < nrows; ++i) {
      for (int j = p_indptr[i]; j < p_indptr[i + 1]; ++j) {
        locations(0, k) = i;               // row indices
        locations(1, k) = p_indices[j];    // column indices
        values(k) = p_data[j];             // values
        ++k;
      }
    }
    arma::sp_mat mat(locations, values);

    return mat;
  } else if (py::isinstance<py::array_t<int64_t>>(csr_obj.attr("indptr")) &&
             py::isinstance<py::array_t<int64_t>>(csr_obj.attr("indices")) &&
             py::isinstance<py::array_t<double>>(csr_obj.attr("data")))
  {
    py::tuple shape = csr_obj.attr("shape").cast<py::tuple>();
    size_t nrows = shape[0].cast<size_t>();

    // Get csr_matrix data, indices, and indptr
    py::array_t<int64_t> indptr = csr_obj.attr("indptr").cast<py::array_t<int64_t>>();
    py::array_t<int64_t> indices = csr_obj.attr("indices").cast<py::array_t<int64_t>>();
    py::array_t<double> data = csr_obj.attr("data").cast<py::array_t<double>>();

    int64_t* p_indptr = indptr.mutable_data();
    int64_t* p_indices = indices.mutable_data();
    double* p_data = data.mutable_data();

    int64_t nnz = data.size();  // number of non-zero elements

    arma::umat locations(2, nnz);
    arma::vec values(nnz);

    for (size_t k = 0, i = 0; i < nrows; ++i) {
      for (int64_t j = p_indptr[i]; j < p_indptr[i + 1]; ++j) {
        locations(0, k) = i;               // row indices
        locations(1, k) = p_indices[j];    // column indices
        values(k) = p_data[j];             // values
        ++k;
      }
    }
    arma::sp_mat mat(locations, values);

    return mat;
  } else {
    throw std::runtime_error("The given object is not a valid CSR matrix.");
  }
}


PYBIND11_MODULE(EliasFanoDB, m){
    py::class_<EliasFanoDB>(m, "EliasFanoDB")
    .def(py::init<>())
    .def("setQB", &EliasFanoDB::setQuantizationBits)
    .def("indexMatrix", &EliasFanoDB::encodeMatrix)
    .def("indexMatrix_dense", &EliasFanoDB::encodeMatrix_dense)
    .def("queryGenes", &EliasFanoDB::queryGenes)
    .def("zgs", &EliasFanoDB::queryZeroGeneSupport)
    .def("decode", &EliasFanoDB::decode)
    .def("mergeDB", &EliasFanoDB::mergeDB)
    .def("findCellTypes", &EliasFanoDB::findCellTypes)
    .def("efMemoryFootprint", &EliasFanoDB::dataMemoryFootprint)
    .def("dbMemoryFootprint", &EliasFanoDB::dbMemoryFootprint)
    .def("quantFootprint", &EliasFanoDB::quantizationMemoryFootprint)
    .def("findMarkerGenes", &EliasFanoDB::findMarkerGenes)
    .def("numberOfCellTypes", &EliasFanoDB::numberOfCellTypes)
    .def("getByteStream", &EliasFanoDB::getByteStream)
    .def("loadByteStream", &EliasFanoDB::loadByteStream)
    .def("getTotalCells", &EliasFanoDB::getTotalCells)
    .def("genes", &EliasFanoDB::getGenesInDB)
    .def("genesSupport", &EliasFanoDB::totalCells)
    .def("geneSupportInCellTypes", &EliasFanoDB::geneSupportInCellTypes)
    .def("cellTypeMarkers", &EliasFanoDB::findCellTypeMarkers)
    .def("getCellTypes", &EliasFanoDB::getCellTypes)
    .def("getCellMeta", &EliasFanoDB::getCellMeta)
    .def("getCellTypeExpression", &EliasFanoDB::getCellTypeMatrix)
    .def("getCellTypeMeta", &EliasFanoDB::getCellTypeMeta)
    .def("evaluateCellTypeMarkersAND", &EliasFanoDB::evaluateCellTypeMarkersAND)
    .def("evaluateCellTypeMarkers", &EliasFanoDB::evaluateCellTypeMarkers)
    .def("getCellTypeSupport", &EliasFanoDB::getCellTypeSupport)
    .def("updateDB", &EliasFanoDB::updateDB)
    .def("DEGenes", &EliasFanoDB::DEGenes);
}
