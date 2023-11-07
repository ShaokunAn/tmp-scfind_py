<img src=https://scfind.sanger.ac.uk/img/scfind.png height="200">

# scfind - Fast searches of large collections of single cell data

Single cell technologies have made it possible to profile millions of cells, but for these resources to be useful they mush be eash to query and access. To facilitate interactive and intuitive access to single cell data we have developed scfind, a search engine for cell atlas. Scfind can be used to evaluate marker genes, to perform in silico gating, and to identify both cell-type specific and housekeeping genes. An interactive interface website with 9 single cell datasets is available at https://scfind.sanger.ac.uk. 


## Installation
scfind is a package packed in python and R (the scfind R library is available at https://github.com/hemberg-lab/scfind) with cpp extensions. To install scfind python package, it's supposed to first install Armadillo library, a c++ linear algebra library.

**Step 1**

Install Armadillo: if you have installed Homebrew, you can install Armadillo just by 

```bash
brew install armadillo
```

  or you can download the source file and compile it mannually. See the documentations of Armadillo (https://arma.sourceforge.net). 

**Step 2**

Install scfind package

```bash
git clone https://github.com/ShaokunAn/tmp-scfind_py.git
cd tmp-scfind_py
pip install -r requirements.txt
pip install .
```

...

## Tutorials

...

scfind provides efficient query and access to large single cell data through an interface which is both very fast and familiar to users from any background. It's main function is to build a index. With the index, user can query and access the dataset efficiently. 

**Examples**

```
import scfind
import anndata

# Read original anndata
adata = anndata.read_h5ad('your/path/to/data.h5ad')

# Build index
scfind_index = scfind.SCFind()
scfind_index.buildCellTypeIndex(adata=addata, dataset_name='your_data_name', cell_type_label='your_cell_type_label', feature_name='your_feature_name') 
```

The `cell_type_label` should be the name of one column in `adata.obs`  which is supposed to be the cell type annotations. The `feature_name` should be the name of one column in `adata.var`  which is the feature annotations (like gene names). 

With the built index, users are allowed to perform multiple kinds of queries, including find cell type markers, find housekeeping genes across cell types, and perform hypergeometric test to find cell types that siginificantly expresse provided genes. Here're some query functions in scfind. Please check out scfind 

[Nature methods paper]: https://www.nature.com/articles/s41592-021-01076-9	"test"

for more functionalities.

```python
# find cell type markers
cell_types = ['your_interested_cell_types']
# you can get the total cell types in index by print(scfind_index.cellTypeNames())
ct_markers = scfind_index.cellTypeMarkers(cell_types=cell_types, )
print(ct_markers)

# find housekeeping genes across cell types
hk_genes = scfind_index.findHouseKeepingGenes(cell_types=scfind_index.cellTypeNames())
print(hk_genes)

# find significantly enriched cell types for specific genes
genes = ['your_interesed_genes']
hypQ_cts = scfind_index.hyperQueryCellTypes(genes)
print(hypQ_cts)

# merge two index
index1 = scfind.SCFind()
index1.buildCellTypeIndex(adata=adata1, ...) # build index first
index2 = scfind.SCFind()
index2.buildCellTypeIndex(adata=adata2, ...)
index1.mergeDataset(index2) # now index1 is the merged index containing adata1 and adata2

# save index
scfind_index.saveObject("your/save/path.bin")

# load index
load_index = scfind.SCFind()
load_index.loadObject("your/load/path.bin")
```



## Citation

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
