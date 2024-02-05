#include <cmath>
#include <iterator>
#include <pybind11/stl.h>
#include <numeric>
#include <functional>
#include <exception>
#include <stdexcept>

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

long EliasFanoDB::eliasFanoCoding(const std::vector<int> &ids, const arma::rowvec &values)
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

  ef_data.push_back(ef);
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

    arma::rowvec denseVector(expression_vector.n_rows, arma::fill::zeros);
    for (arma::sp_colvec::const_iterator it = expression_vector.begin(); it != expression_vector.end(); ++it)
    {
        denseVector(it.row()) = (*it);
    }

    std::deque<int> sparse_index;

    for (arma::sp_rowvec::const_iterator it = expression_vector.begin(); it != expression_vector.end(); ++it)
    {
      int col_idx = it.row();
      double value = (*it);

      if (value > 0)
      {
        current_cells[col_idx].reads += value;
        current_cells[col_idx].features++;
        sparse_index.push_back(col_idx + 1); // 1 based indexing
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

    auto ef_index = eliasFanoCoding(ids, denseVector);
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
    arma::rowvec denseVector(total_cells);

    auto dense_mat_proxy = dense_mat.unchecked<2>();
    for (int i = 0; i < total_cells; ++i) {
        denseVector(i) = dense_mat_proxy(i, gene_col);
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

    auto ef_index = eliasFanoCoding(ids, denseVector);
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

//    results[g] = gene_results;

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
  for (auto const &ct : cts)
  {
    // TODO(Nikos) fix this, otherwise we will get a nice segfault no error control
    auto cit = this->cell_types.find(ct);
    if (cit != this->cell_types.end())
    {
      ct_support.push_back(this->inverse_cell_type[cit->second].total_cells);
    }
    else
    {
      ct_support.push_back(0);
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

  // intersect cells
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
  for (auto const &ct : cts)
  {
    auto marker_genes = this->_cellTypeScore(ct, bk_cts, gene_set, mode);
    if (marker_genes.empty())
    {
      std::cerr << "Marker genes could not be found for cell type " << ct << std::endl;
      continue;
    }

    for (auto &t : marker_genes)
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
    return std::map<EliasFanoDB::GeneName, CellTypeMarker>();
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
  else if (mode == AND)
  {
    CellTypeMarker ctm_template = {0, 0, 0, 0};
    std::ostringstream imploded;
    std::copy(gene_names.begin(), gene_names.end(),
              std::ostream_iterator<std::string>(imploded, ","));
    auto dit = scores.insert(std::make_pair(imploded.str(), ctm_template));
    CellTypeMarker &gene_ctm_score = dit.first->second;

    py::dict result = this->_findCellTypes(gene_names, std::vector<CellTypeName>(active_cell_types.begin(), active_cell_types.end()));

    std::vector<CellTypeName> cell_types;
    for (auto item : result) {
      cell_types.push_back(item.first.cast<CellTypeName>());
    }

    for (auto const &ct : cell_types)
    {
      auto cells_positive = result[py::str(ct)].cast<py::list>().size();
      auto cells_negative = this->inverse_cell_type.at(this->cell_types.at(ct)).total_cells - cells_positive;
      if (ct == cell_type)
      {
        gene_ctm_score.tp = cells_positive;
        gene_ctm_score.fn += cells_negative;
      }
      else
      {
        gene_ctm_score.fp += cells_positive;
        gene_ctm_score.tn = cells_negative;
      }
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
  const CellType &ctmeta = this->inverse_cell_type[ct_it->second];

  py::dict result;
  result["total_cells"] = py::cast(ctmeta.getTotalCells());

  return result;
}

const arma::sp_mat EliasFanoDB::csr_to_sp_mat(const py::object& csr_obj) {
  if (py::isinstance<py::array_t<int>>(csr_obj.attr("indptr")) &&
      py::isinstance<py::array_t<int>>(csr_obj.attr("indices")) &&
      py::isinstance<py::array_t<double>>(csr_obj.attr("data"))) {
    py::tuple shape = csr_obj.attr("shape").cast<py::tuple>();
    int nrows = shape[0].cast<int>();

    // Get csr_matrix data, indices, and indptr
    py::array_t<int> indptr = csr_obj.attr("indptr").cast<py::array_t<int>>();
    py::array_t<int> indices = csr_obj.attr("indices").cast<py::array_t<int>>();
    py::array_t<double> data = csr_obj.attr("data").cast<py::array_t<double>>();

    int* p_indptr = indptr.mutable_data();
    int* p_indices = indices.mutable_data();
    double* p_data = data.mutable_data();

    int nnz = data.size();  // number of non-zero elements

    arma::umat locations(2, nnz);
    arma::vec values(nnz);

    arma::uword u_nrows = static_cast<arma::uword>(nrows);
    for (arma::uword k = 0, i = 0; i < u_nrows; ++i) {
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
    int64_t nrows = shape[0].cast<int64_t>();

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

    arma::uword u_nrows = static_cast<arma::uword>(nrows);
    for (arma::uword k = 0, i = 0; i < u_nrows; ++i) {
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
    .def("getCellTypeSupport", &EliasFanoDB::getCellTypeSupport);
}
