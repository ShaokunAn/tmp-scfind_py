<img src=https://scfind.sanger.ac.uk/img/scfind.png alt="scfind" height="200">

# scfind - Fast searches of large collections of single cell data

Single-cell technologies have enabled the profiling of millions of cells. However, for these vast resources to be fully leveraged, they must be easily queryable and accessible. To facilitate interactive and intuitive access to single-cell data, we have developed scfind, a search engine for cell atlases. Scfind can be utilized to evaluate marker genes, perform in silico gating, and identify both cell-type specific and housekeeping genes. An interactive interface with access to nine single-cell datasets is available at [scfind.sanger.ac.uk](https://scfind.sanger.ac.uk).

## Installation
scfind is available as a package for Python and R, with C++ extensions. The scfind R library is accessible at [our GitHub repository](https://github.com/hemberg-lab/scfind). Before installing the scfind Python package, the Armadillo library, a C++ linear algebra library, must be installed.

**Step 1: Install Armadillo**

If you have Homebrew  installed, Armadillo can be installed with the following command:

```bash
brew install armadillo
```

Alternatively, you can download the source files and compile them mannually. See the [Armadillo documentation](https://arma.sourceforge.net) for more details.

**Step 2: Install the scfind package**

```bash
git clone https://github.com/ShaokunAn/tmp-scfind_py.git
cd tmp-scfind_py
pip install -r requirements.txt
python setup.py build_ext --inplace
python setup.py sdist bdist_wheel
pip install .
```



## Tutorials

scfind offers efficient querying and access to large single-cell datasets through an interface that is both fast and user-friendly. Its primary function is to build an index, which enables user to query the dataset efficiently.

**Examples**

```
import scfind
import anndata

# Read the original AnnData
adata = anndata.read_h5ad('your/path/to/data.h5ad')

# Build the index
scfind_index = scfind.SCFind()
scfind_index.buildCellTypeIndex(adata=addata, dataset_name='your_data_name', 
cell_type_label='your_cell_type_label', 
feature_name='your_feature_name') 
```

The `cell_type_label` should correspond to a column in `adata.obs` that contains the cell type annotations. The `feature_name` should correspond to a column in `adata.var`  that contains feature annotations, like gene names.

With the built index, users can perform various queries, such as finding cell type markers, identifying housekeeping genes across cell types, and conducting hypergeometric tests to discover significantly enriched cell types for provided genes. Below are some query functions in scfind. For additional functionalities, please refer to the scfind [Nature methods](https://www.nature.com/articles/s41592-021-01076-9) paper.

```python
# Find cell type markers
cell_types = ['your_interested_cell_types']
ct_markers = scfind_index.cellTypeMarkers(cell_types=cell_types, )
print(ct_markers)

# Find housekeeping genes across cell types
hk_genes = scfind_index.findHouseKeepingGenes(cell_types=scfind_index.cellTypeNames())
print(hk_genes)

# Find significantly enriched cell types for specific genes
genes = ['your_interesed_genes']
hypQ_cts = scfind_index.hyperQueryCellTypes(genes)
print(hypQ_cts)

# Merge two indices
index1 = scfind.SCFind()
index1.buildCellTypeIndex(adata=adata1, ...) # build the first index
index2 = scfind.SCFind()
index2.buildCellTypeIndex(adata=adata2, ...)
index1.mergeDataset(index2) # now index1 contains both adata1 and adata2

# Save the index
scfind_index.saveObject("your/save/path.bin")

# Load the index
load_index = scfind.SCFind()
load_index.loadObject("your/load/path.bin")
```



## Citation
Please cite our work using the following reference:

```
@article{lee2021fast,
  title={Fast searches of large collections of single-cell data using scfind},
  author={Lee, Jimmy Tsz Hang and Patikas, Nikolaos and Kiselev, Vladimir Yu and Hemberg, Martin},
  journal={Nature methods},
  volume={18},
  number={3},
  pages={262--271},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```
