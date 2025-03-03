import pickle
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
import scipy.sparse
from EliasFanoDB import EliasFanoDB
from anndata import AnnData
import anndata as ad
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm



class SCFind:

    def __init__(self):
        super().__init__()

        self.index = EliasFanoDB()
        self.datasets = []
        self.serialized = bytes()
        self.metadata = {}
        self.index_exist = False
        self.datasets_map = {}
        self.datasetID_map = pd.DataFrame(columns=['dataset_id'])

    def buildCellTypeIndex(self, adata: AnnData,
                           dataset_id: str,
                           tissue: str,
                           dataset_index: Optional[int] = None,
                           feature_name: str = 'gene',
                           cell_type_label: str = 'cell_type',
                           qb: int = 2
                           ) -> None:
        """
        Build an index for cell types based on the given AnnData data.

        Parameters
        ----------
        adata: AnnData
            The annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells
            and columns to genes.

        dataset_id: str
            Dataset ID obtained from HuBMAP.

        dataset_index: Optional[int], default=None
            Index of the dataset in the SCFind object. If not provided, it will be set as 0.
            This is used to update index for HuBMAP dataset. By checking the number of datasets in old index,
            we can set the index of the new dataset to distinguish duplicate cell types.

        tissue: str
            Tissue name.

        feature_name: str, default='gene'
            The label or key in the AnnData object's variables (var) that corresponds to the feature names.

        cell_type_label: str, default='cell_type'
            The label or key in the AnnData object's observations (obs) that corresponds to the cell type.

        qb: int, default=2
            Number of bits per cell that are going to be used for quantile compression of the expression data.

        Returns
        -------
        None
            Updates are made directly to the C++ objects, enhancing search and retrieval operations.

        Raises
        ------
        ValueError
            If dataset_id contains any dots or if assay_name is not found in the AnnData object.
        """

        # Use id to separate dataset_id and tissue
        dataset_id_modified = dataset_id.replace('.', '-')  # use - to replace . in dataset_id
        dataset_id_modified = dataset_id_modified.replace('_', '-')
        if dataset_index is None:
            dataset_index = 0

        self.datasetID_map = pd.DataFrame({'dataset_id': [dataset_id_modified]}, index=[dataset_index])
        # Get cell types
        try:
            cell_types_all = adata.obs[cell_type_label].astype('category')
        except KeyError:
            raise ValueError(f"'{cell_type_label}' not found in adata.obs.")

        cell_types = cell_types_all.cat.categories.tolist()
        # Assuming tissue and cell_types are already defined, and use dataset_id as stamp to distinguish datasets
        tissue_modified = tissue.replace('.', '-')
        tissue_modified = tissue_modified.replace('_', '-')
        dataset_id_stamp = f"{tissue_modified}_{dataset_index}"
        new_cell_types = {cell_type: f"{dataset_id_stamp}.{cell_type}" for cell_type in cell_types}

        if len(cell_types) == 0:
            raise ValueError("No cell types found in the provided AnnData object.")

        print(f"Generating index for {dataset_id_modified}")

        non_zero_cell_types = []
        # Get expression data
        is_sparse = scipy.sparse.issparse(adata.X)

        ef = EliasFanoDB()
        qb_set = ef.setQB(qb)
        if qb_set == 1:
            raise ValueError("Setting the quantization bits failed")

        for cell_type in cell_types:
            inds_cell = np.where(cell_types_all == cell_type)[0]
            if len(inds_cell) < 2:
                continue
            non_zero_cell_types.append(cell_type)
            print(f"\t Indexing {cell_type} as {new_cell_types[cell_type]} with {len(inds_cell)} cells.")

            cell_type_exp = adata.X[inds_cell]

            try:
                cell_type_genes = adata.var[feature_name].tolist()
            except KeyError:
                raise ValueError(f"'{feature_name}' not found in adata.var")

            cell_type_exp = cell_type_exp.astype(np.float64)
            # Convert python matrix into
            if is_sparse:
                # Ensure the sparse matrix is csr format
                if not scipy.sparse.isspmatrix_csr(cell_type_exp):
                    cell_type_exp = cell_type_exp.tocsr()

                ef.indexMatrix(new_cell_types[cell_type], cell_type_exp, cell_type_genes)
            else:
                ef.indexMatrix_dense(new_cell_types[cell_type], cell_type_exp, cell_type_genes)

        self.index = ef
        self.datasets = [tissue_modified]
        self.datasets_map[tissue_modified] = [dataset_id_modified]
        self.index_exist = True

    def saveObject(self, file: str) -> None:
        """
        Save a serialized object to a file.

        Parameters
        ----------
        object: Dict[EliasFanoDB, str]
            The object to be serialized, containing the EliasFanoDB index and dataset name.

        file: str
            The path to the file where the serialized object will be saved.

        Returns
        -------
        None
            Updates are made directly to the C++ objects
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        # Serialize the object
        self.serialized = self.index.getByteStream()

        # Conver to a dict to store serialized object instead of the EliasFanoDB object
        saved_result = {'serialized': self.serialized,
                        'datasets': self.datasets,
                        'metadata': self.metadata,
                        'datasets_map': self.datasets_map,
                        'datasetID_map': self.datasetID_map,
                        }

        # Save the serialized object to a file
        with open(file, 'wb') as f:
            pickle.dump(saved_result, f)

        # Clear the serialized stream
        self.serialized = bytes()

    def loadObject(self,
                   file: str
                   ) -> None:
        """
        Load a serialized object from file.

        Parameters
        ----------
        file: str
            Path to the file containing the serialized SCFind object.

        Returns
        -------
        None
            Update SCFind object attributes.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """

        # Load serialized data from file
        with open(file, 'rb') as f:
            loaded_object = pickle.load(f)

        # Deserialize data
        fail = self.index.loadByteStream(loaded_object['serialized'])

        if fail:
            raise ValueError("Failed to load byte stream into index")

        self.datasets = loaded_object['datasets']
        self.serialized = None
        self.index_exist = True
        self.metadata = loaded_object['metadata']
        self.datasets_map = loaded_object['datasets_map']
        self.datasetID_map = loaded_object['datasetID_map']

    def mergeDataset(self, new_object: 'SCFind') -> None:
        """
        Merge another SCFind object into the current object.

        Parameters:
        -----------
        new_object: SCFind
            The SCFind object containing datasets to be merged.

        Returns:
        --------
        None
            Update attributes

        Raises:
        -------
        Warning
            If common dataset names exist between the two objects.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        # Update datasets_map
        all_datasets = set(self.datasets).union(new_object.datasets)
        self.datasets = all_datasets
        for d in all_datasets:
            if d not in self.datasets_map:
                self.datasets_map[d] = new_object.datasets_map[d]
            else:
                intersection = set(self.datasets_map.get(d, [])).intersection(new_object.datasets_map.get(d, []))
                if intersection:
                    raise ValueError(f"Error: {intersection} already exists in current index."
                                     f"Not allowed to update index with duplicated datasets.")

                self.datasets_map[d].extend(new_object.datasets_map.get(d, []))

        # Update datasetID_map
        duplicate_dataset_index = set(self.datasetID_map.index).intersection(new_object.datasetID_map.index)
        if duplicate_dataset_index:
            raise ValueError(f"Error: duplicate dataset index in mapping: {duplicate_dataset_index}")

        self.datasetID_map = pd.concat(
            [self.datasetID_map, new_object.datasetID_map], ignore_index=False
        )

        print(f"Merging {new_object.datasets} ... ")
        self.index.mergeDB(new_object.index)

    def mergeAnnData(self, adata: AnnData,
                     dataset_id: str,
                     tissue: str,
                     feature_name: str = 'gene',
                     cell_type_label: str = 'cell_type',
                     qb: int = 2
                     ) -> None:
        """
        Merge index from an AnnData object into the current SCFind object

        Parameters
        ----------
        adata: AnnData
            The annotated data matrix of shape (n_obs, n_vars)

        dataset_id: str
            Dataset ID obtained from HuBMAP.

        tissue: str
            Tissue name.

        feature_name: str, default='feature_name'
            Name of the feature to be used, typically indicate gene names.

        cell_type_label: str, default='cell_type1'
            The label or key in the AnnData object's observations (obs) that corresponds to the cell type.

        qb: int, default=2
            Number of bits per cell that are going to be used for quantile compression of the expression data.


        Returns
        -------
        None
            Update attributes

        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        dataset_index = len(self.datasetID_map)

        new_object = SCFind()._buildCellTypeIndex(
            adata=adata,
            dataset_id=dataset_id,
            tissue=tissue,
            dataset_index=dataset_index,
            feature_name=feature_name,
            cell_type_label=cell_type_label,
            qb=qb,
            )
        self.mergeDataset(new_object)

    def markerGenes(self,
                    gene_list: Union[str, List[str]],
                    datasets: Optional[Union[str, List[str]]] = None,
                    exhaustive: bool = False,
                    support_cutoff: int = -1,
                    ) -> pd.DataFrame:
        """
        Find marker genes in the given datasets.

        Parameters
        ----------
        gene_list: str or list of str,
            Gene or a list of genes existing in the database.

        datasets: str or a list of str, optional (default=None)
            Dataset or a list of datasets to be searched in. If datasets=None, search all datasets.

        exhaustive: bool, default=False
            Whether to perform an exhaustive search instead of FP-growth.

        support_cutoff: int, default=-1
            Minimum support in cells. By default, it is -1, which defaults to estimating
            the cutoff value for a given gene set

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the search.

        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        datasets = self._select_datasets(datasets)  # Identify valid datasets (no stamp)
        if not datasets:
            raise ValueError(f"No valid dataset identified in input. Use index.datasets to check valid datasets.")
        
        datasets_name_stamp = [ds_stamp for d in datasets for ds_stamp in self.datasets_map[d]]

        try:
            results = self.index.findMarkerGenes(
                self._case_correct(gene_list),
                datasets_name_stamp,
                exhaustive,
                support_cutoff
            )
            results_df = pd.DataFrame.from_dict(results)
            return results_df
        except ValueError:
            # Log the error if necessary
            # print(f"Error: {err}")
            return pd.DataFrame(columns=["Genes", "Query", "TF-IDF", "Number of Cells"])

    def getCellTypeExpression(self,
                              cell_type: str,
                              ) -> AnnData:
        """
        Retrieve expression matrix of provided cell types.

        Parameters
        ----------
        cell_type: str
            The cell type for which we want to retrieve the expression data.
            The format should be tissue(dataset).cell_type

        Returns
        -------
        AnnData: 
            Return the sparse expression matrix.
        """
        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")
        
        cell_type_stamp = self._select_celltype_stamp(cell_type)

        adatas = []
        for ct in cell_type_stamp:
            result = self.index.getCellTypeExpression(ct)
            
            if len(result) == 5:  # sparse matrix
                values, row_indices, col_indices, n_cells, feature_names = result
                n_features = len(feature_names)
                sp_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_cells, n_features))
                adata = AnnData(X=sp_matrix)
                adata.var_names = feature_names
            elif len(result) == 2:  # dense matrix
                mat, feature_names = result
                adata = AnnData(X=mat)
                adata.var_names = feature_names
        
            adatas.append(adata)
        
        adata_all = ad.concat(adatas, join='outer')

        return adata_all

    def cellTypeMarkers(self,
                        cell_types: Union[str, List[str]],
                        background_cell_types: Optional[Union[str, List[str]]] = None,
                        top_k: int = 5,
                        sort_field: str = 'f1',
                        include_prefix: bool = True
                        ) -> pd.DataFrame:
        """
        Find marker genes for a specific cell type.

        Parameters
        ----------
        cell_types: str or list of str
            The cell types for which to extract the marker genes.
            The format should be tissue(dataset).cell_type

        background_cell_types: str or list of str, optional (default=None)
            The universe of cell types to consider. If not provided,
            it defaults to all cell types in the SCFind object. The format should be
            tissue(dataset).cell_type

        top_k: int, default=10
            How many genes to retrieve. Defaults to 10.

        sort_field: str, default='f1'
            The dataframe will be sorted according to this field. Defaults to 'f1'.

        include_prefix: bool, default=True
            If True, include the dataset name as prefix in the cellType.
            If False, only include cell type name.

        Returns
        -------
        DataFrame
            A dataframe where each row represents a gene score for a specific cell type.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        if background_cell_types is None:
            # print("Considering the whole database.")
            background_cell_types = self.cellTypeNames()

        if isinstance(background_cell_types, str):
            background_cell_types = [background_cell_types]

        if isinstance(cell_types, str):
            cell_types = [cell_types]

        background_cell_types_set = set(background_cell_types)
        background_cell_types_set.update(set(cell_types))

        background_cell_types = list(background_cell_types_set)

        cell_types_stamp = self._select_celltype_stamp(cell_types)
        background_cell_types_stamp = self._select_celltype_stamp(background_cell_types)

        all_cell_types = self.index.cellTypeMarkers(cell_types_stamp, background_cell_types_stamp)
        all_cell_types = pd.DataFrame(all_cell_types)

        if sort_field not in all_cell_types.keys():
            print(f"Column {sort_field} not found. F1 score would be used.")
            sort_field = 'f1'

        all_cell_types = all_cell_types.sort_values(by=[sort_field, 'genes'], ascending=[False, True],
                                                    ignore_index=True)
        all_cell_types = all_cell_types.head(top_k)

        if not include_prefix:
            all_cell_types['cellType'] = all_cell_types['cellType'].str.split('.').str[-1]

        return all_cell_types

    def cellTypeNames(self, datasets: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Retrieve the names of cell types in datasets from the SCFind object.

        Parameters
        ----------
        datasets: str or list of str, optional (default=None)
            Dataset names to filter the cell types.
            If not provided, all cell types from the SCFind object will be returned.

        Returns
        -------
        List[str]
            A list of cell type names. If 'datasets' is provided, only the cell types
            corresponding to the specified datasets will be returned. The format is [tissue(dataset).cell_type]
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        all_cell_types_stamp = self.index.getCellTypes()
        all_cell_types = [f"{name.split('_')[0]}.{name.split('.')[1]}" for name in all_cell_types_stamp]
        all_cell_types = list(set(all_cell_types))  # remove duplicate datasets distinguished by stamp for better unstanding to users

        if datasets is None:
            return all_cell_types
        else:
            datasets = self._select_datasets(datasets)

            filtered_cell_types = [cell_type for cell_type in all_cell_types
                                   if cell_type.split('.')[0] in datasets]
            return filtered_cell_types

    def evaluateMarkers(self,
                        gene_list: Union[str, List[str]],
                        cell_types: Union[str, List[str]],
                        background_cell_types: Optional[Union[str, List[str]]] = None,
                        sort_field: str = 'f1',
                        include_prefix: bool = True
                        ) -> pd.DataFrame:
        """
        Evaluate a user-specific query by calculating the precision recall metrics.

        This method evaluates a list of genes for specific cell types and returns
        a DataFrame where each row represents a gene score for a specific cell type.

        Parameters
        ----------
        gene_list: str or list of str
            Genes to be evaluated.

        cell_types: str or list of str
            Cell types to be evaluated. The format should be tissue(dataset).cell_type

        background_cell_types: str or list of str, optional (default=None)
            The universe of cell types to consider. If not provided, all cell types
            from the SCFind object will be considered. The format should be tissue(dataset).cell_type

        sort_field: str, default='f1'
            The DataFrame will be sorted according to this field.

        include_prefix: bool, default=True
            If True, include the dataset name as prefix in the cellType.
            If False, only include cell type name.

        Returns
        -------
        DataFrame
            A DataFrame where each row represents a gene score for a specific cell type.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        if background_cell_types is None:
            print("Considering the whole database.")
            background_cell_types = self.cellTypeNames()

        if isinstance(background_cell_types, str):
            background_cell_types = [background_cell_types]

        if isinstance(cell_types, str):
            cell_types = [cell_types]

        background_cell_types_set = set(background_cell_types)
        background_cell_types_set.update(set(cell_types))

        background_cell_types = list(background_cell_types_set)

        cell_types_stamp = self._select_celltype_stamp(cell_types)
        background_cell_types_stamp = self._select_celltype_stamp(background_cell_types)

        all_cell_types = self.index.evaluateCellTypeMarkers(
            cell_types_stamp,
            self._case_correct(gene_list),
            background_cell_types_stamp,
        )
        all_cell_types = pd.DataFrame(all_cell_types)

        if sort_field not in all_cell_types.columns:
            print(f"Column {sort_field} not found. F1 score would be used.")
            sort_field = 'f1'

        all_cell_types = all_cell_types.sort_values(by=sort_field, ascending=False, ignore_index=True)

        if not include_prefix:
            all_cell_types['cellType'] = all_cell_types['cellType'].str.split(".").str[-1]

        return all_cell_types

    def hyperQueryCellTypes(self,
                            gene_list: Union[str, List[str]],
                            datasets: Optional[Union[str, List[str]]] = None,
                            include_prefix: bool = True
                            ) -> pd.DataFrame:
        """
        Perform a hypergeometric test on cell types based on gene_list.

        This method searches for cell types based on a gene list and then performs
        a hypergeometric test on the results.

        Parameters
        ----------
        gene_list: str or list of str
            Genes to be searched in the gene index. Operators can be used:
            "-gene" to exclude "gene", "*gene" if either "gene" is epxressed,
            "*-gene" if either gene is expressed to be excluded.

        datasets: str or list of str, optional (default=None)
            The datasets vector that will be tested as background for the hypergeometric test.
            If datasets=None, use all datasets as background.

        include_prefix: bool, default=True
            If True, include the dataset name as prefix in the cell_type.
            If False, only include cell type name.

        Returns
        -------
        pd.DataFrame
            A DataFrame that contains all cell types with their respective cell cardinality and
            the hypergeometric test results.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        result = self.findCellTypes(gene_list, datasets)  # ids of each cellcell_stamp
        if result:
            df = self._phyper_test(result)  # merge celltype by removing stamp
            df = df.sort_values(by='adj-pval', ascending=True, ignore_index=True)
            if not include_prefix:
                # Split the 'cell_type' column and keep only the suffix
                df['cell_type'] = df['cell_type'].str.split('.').str[-1]
            return df[df['adj-pval']<0.05]
        else:
            print("No Cell Is Found!")
            return pd.DataFrame({'cell_type': [], 'cell_hits': [],
                                 'total_cells': [], 'pval': [],
                                 'adj-pval': [],
                                 })

    def findCellTypes(self,
                      gene_list: Union[str, List[str]],
                      datasets: Optional[Union[str, List[str]]] = None
                      ) -> Dict[str, List[int]]:
        """
        Find cell types and the cells associated with given gene_list.
        All returned cells express all genes in the given gene_list.

        Parameters
        ----------
        gene_list: str or list of str
            Genes to be searched in the gene index. Operators can be used:
            "-gene" to exclude "gene", "*gene" if either "gene" is expressed,
            "*-gene" if either gene is expressed to be excluded.

        datasets: str or list of str, optional (default=None)
            The datasets that will be considered. If datasets=None, all datasets
            from the SCFind object will be considered. 

        Returns
        -------
        Dict[str, List[int]]
            A dictionary where keys are cell type names and values are lists of integers representing cell IDs.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        datasets = self._select_datasets(datasets)
        datasets_stamp = [d for ds in datasets for d in self.datasets_map[ds]]

        if isinstance(gene_list, str):
            gene_list = [gene_list]

        regular_genes = [gene for gene in gene_list if gene.startswith("-") or gene.startswith("*")]
        if len(regular_genes) == 0:
            sanitized_genes = self._case_correct(gene_list)
            if not sanitized_genes:
                # if no valid genes returned, return {}
                print(f"All input genes are not valid.")
                return {}
            if all(isinstance(gene, str) for gene in sanitized_genes):
                cts = self.index.findCellTypes(sanitized_genes, datasets_stamp)
                # in python, index starts at 0
                cts = {key: [cell_id - 1 for cell_id in value] for key, value in cts.items()}
                return cts
            else:
                return {}
        else:
            # Extract genes based on operators
            pos_genes = [gene for gene in gene_list if not (gene.startswith("-") or gene.startswith("*"))]
            pos_genes = self._case_correct(pos_genes)
            excl_or_genes = [gene for gene in gene_list if gene.startswith("*-") or gene.startswith("-*")]
            or_genes = [gene.replace("*", "") for gene in gene_list if gene.startswith("*")
                        and gene not in excl_or_genes]
            or_genes = self._case_correct(or_genes)
            excl_genes = [gene.replace("-", "") for gene in gene_list if gene.startswith("-")
                          and gene not in excl_or_genes]
            excl_genes = self._case_correct(excl_genes)
            excl_or_genes = [gene.replace("*-", "").replace("-*", "") for gene in excl_or_genes]
            excl_or_genes = self._case_correct(excl_or_genes)

            # Checking intersections
            intersections = (
                    set(pos_genes) & set(or_genes) or
                    set(pos_genes) & set(excl_genes) or
                    set(pos_genes) & set(excl_or_genes) or
                    set(or_genes) & set(excl_genes) or
                    set(or_genes) & set(excl_or_genes) or
                    set(excl_genes) & set(excl_or_genes)
            )

            if intersections:
                print("Warning: Same gene labeled with different operators!")
                print("There is a priority to handle operators:")

                pos_msg = f"Cells with {' ^ '.join(pos_genes)} expression will be included."
                or_msg = f"Then cells with {' v '.join(or_genes)} expression will be included." if or_genes else ""
                excl_msg = f"The result will be excluded by {' ^ '.join(excl_genes)}"
                excl_or_msg = f"and further be excluded by {' v '.join(excl_or_genes)}" if excl_or_genes else ""

                print(pos_msg, or_msg)
                print(excl_msg, excl_or_msg)
                print("\n")

            cell_to_id = []
            if len(pos_genes) == 0 and len(or_genes) == 0 and (len(excl_genes) != 0 or len(excl_or_genes) != 0):
                datasets_stamp = [d for ds in datasets for d in self.datasets_map[ds]]
                all_cell_types_stamp = self.index.getCellTypes()
                cell_types_stamp = [ct for ct in all_cell_types_stamp if ct.split('_')[0] in datasets]
                cell_to_id = self.index.getCellTypeSupport(cell_types_stamp, False)
                cell_to_id = {name: list(range(cells)) for name, cells in 
                              zip(cell_types_stamp, cell_to_id)}

                cell_to_id = SCFind._pair_id(cell_to_id)  # celltype_stamp#id format

            if len(or_genes) != 0:
                gene_or = []
                for i in range(len(or_genes)):
                    tmp_id = SCFind._pair_id(self.index.findCellTypes(pos_genes + [or_genes[i]], datasets_stamp))  # celltype_stamp#id

                    if len(pos_genes) != 0 and tmp_id is not None:
                        print(f"Found {len(tmp_id)} {'cells' if len(tmp_id) > 1 else 'cell'} co-expressing "
                              f"{' and '.join(pos_genes + [or_genes[i]])}")
                    if tmp_id is not None:
                        cell_to_id = list(set(cell_to_id + tmp_id))
                        gene_or.append(or_genes[i])
                    else:
                        cell_to_id = cell_to_id

                if len(pos_genes) == 0 and len(gene_or) != 0:
                    print(
                        f"Found {len(cell_to_id)} {'cells' if len(cell_to_id) > 1 else 'cell'} expressing "
                        f"{' or '.join(gene_or)}")
            else:
                if len(pos_genes) != 0:
                    cell_to_id = SCFind._pair_id(self.index.findCellTypes(pos_genes, datasets_stamp))  #celltype_stamp#id
                    print(
                        f"Found {len(cell_to_id)} {'cells co-expressing' if len(pos_genes) > 1 else 'cell expressing'} "
                        f" {' and '.join(pos_genes)}")

            count_cell = len(cell_to_id)
            gene_excl = []

            if len(excl_or_genes) != 0:
                # Negative select cell in OR condition
                for i in range(len(excl_or_genes)):
                    ex_tmp_id = SCFind._pair_id(self.index.findCellTypes(excl_genes + [excl_or_genes[i]], datasets_stamp))

                    num_excluded = sum(item in ex_tmp_id for item in cell_to_id)
                    excl_message = f"Excluded {num_excluded} {'cells' if num_excluded > 1 else 'cell'}"
                    if len(excl_genes) != 0:
                        excl_message += f" co-expressing {' and '.join(excl_genes + [excl_or_genes[i]])}"
                    else:
                        excl_message += f" expressing {excl_or_genes[i]}"
                    print(excl_message)

                    if ex_tmp_id:
                        cell_to_id = list(set(cell_to_id) - set(ex_tmp_id))
                        gene_excl.append(excl_or_genes[i])
                    else:
                        cell_to_id = cell_to_id

                count_cell -= len(cell_to_id)
                if count_cell > 0 and len(gene_excl) == 0:
                    print(
                        f"Excluded {count_cell} {'cells' if count_cell > 1 else 'cell'} "
                        f"expressing {' and '.join(excl_genes)}")

            else:
                if len(excl_genes) != 0:
                    # Negative selection
                    cell_to_id = list(
                        set(cell_to_id) - set(SCFind._pair_id(self.index.findCellTypes(excl_genes, datasets_stamp))))
                    count_cell -= len(cell_to_id)
                    if count_cell > 0:
                        excl_message = (
                            f"Excluded {count_cell} {'cells co-expressing' if len(excl_genes) > 1 else 'cell expressing'} "
                            f" {' and '.join(excl_genes)}")
                        print(excl_message)
                    else:
                        print("No Cell Is Excluded!")

            df = pd.DataFrame([x.split('#') for x in cell_to_id])
            if not df.empty:
                result = {key: sorted(list(map(int, group[1]))) for key, group in df.groupby(0)}

                if len(df[0].unique()) == len(df):
                    pass  # result is already in the desired format
                else:
                    if len(set(result.keys())) == 1:
                        unique_key = list(result.keys())[0]
                        tmp = {unique_key: sorted([item for sublist in result.values() for item in sublist])}
                        result = tmp
                    else:
                        result = {k: sorted(map(int, v)) for k, v in df.groupby(0)[1]}
                # result = {key: [cell_id - 1 for cell_id in value] for key, value in result.items()}
                return result
            else:
                print("No Cell Is Found!")
                return {}

    @property
    def scfindGenes(self, ) -> List[str]:
        """
        Get all genes in the database

        Returns
        -------
        List of Genes in database.
        """
        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        return self.index.genes()

    def findCellTypeSpecificities(self,
                                  gene_list: Optional[Union[str, List[str]]] = None,
                                  datasets: Union[str, List[str]] = None,
                                  min_cells: int = 10,
                                  min_fraction: float = 0.25
                                  ) -> Dict[str, List[int]]:
        """
        Determine the count of cell types that 'express' a gene, where 'express' is defined as the proportion of cells
        expressing the gene exceeding a specified minimum fraction.

        Parameters
        ----------
        gene_list: str or list of str, optional (default=None)
            Genes to be searched in the gene.index. If gene_list=None, use all genes.

        datasets: str or list of str, default=None
            The datasets that will be searched in.

        min_cells: int, default 10
            Threshold of cell hit of a cell type.

        min_fraction: float, default 0.25
            Portion of total cell as threshold.

        Returns
        -------
        Dict[str, List[int]]
            A dictionary with the list of cell type for each gene.

        Raises
        ------
        ValueError:
            If min_fraction is not in the range (0, 1).
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        if not (0 < min_fraction < 1):
            raise ValueError("min_fraction reached limit, please use values > 0 and < 1.0.")

        print("Calculating cell-types for each gene...")

        datasets = self._select_datasets(datasets)
        datasets_stamp = [d for ds in datasets for d in self.datasets_map[ds]]

        if gene_list is None:
            res = self.index.geneSupportInCellTypes(self.index.genes(), datasets_stamp)
        else:
            gene_list = self._case_correct(gene_list)
            res = self.index.geneSupportInCellTypes(gene_list, datasets_stamp)

        res_tissue = {key.replace(".", "#"): value for key, value in res.items()}

        # extract elements from dict
        res_df = pd.DataFrame({
            'res_values': [item for sublist in res.values() for item in sublist.values()],
            'res_ind': [key for key, sublist in res.items() for _ in sublist]
        })

        res_tissue_df = pd.DataFrame({
            'tissue_values': [item for sublist in res_tissue.values() for item in sublist.values()],
            'tissue_ind': [f"{key}.{item}" for key, sublist in res_tissue.items() for item in sublist.keys()]
        })

        df = pd.concat([res_df, res_tissue_df], axis=1)

        if df.empty:
            return {gene: 0 for gene in gene_list}
        else:
            df.iloc[:, 0] = df.iloc[:, 3].str.replace(r'^[^.]+\.', '', regex=True).apply(
                lambda x: np.sum(self.index.getCellTypeSupport(self._select_celltype_stamp(x), True)) * min_fraction)

            df.loc[df.iloc[:, 0] < min_cells, df.columns[0]] = min_cells
            df = df[df.iloc[:, 2] > df.iloc[:, 0]]
            if df.empty:
                return {gene: 0 for gene in gene_list}
            else:
                return df.iloc[:, 1].value_counts().to_dict()

    def findTissueSpecificities(self,
                                gene_list: Optional[Union[str, List[str]]] = None,
                                min_cells: int = 10
                                ) -> Dict[str, int]:
        """
        Find out how many tissues (datasets) each gene is found in.

        Parameters
        ----------
        gene_list: str or list of str, optional (default=None)
            Genes to be searched in the gene.index. If gene_list=None, use all genes.

        min_cells: int, default 10
            Threshold of cell hit of a tissue.

        Returns
        -------
        Dict[str, int]
            A dictionary with the number of tissues (datasets) for each gene.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        if len(self.datasets) <= 1:
            raise ValueError("Index contains 1 dataset only. No need to detect genes in tissues (datasets).")
        print("Calculating tissues for each gene...")

        datasets_stamp = [d for ds in self.datasets_map.values() for d in ds]
        if gene_list is None:
            res = self.index.geneSupportInCellTypes(self.index.genes(), datasets_stamp)
        else:
            gene_list = self._case_correct(gene_list)
            res = self.index.geneSupportInCellTypes(gene_list, datasets_stamp)

        if not res:
            return {gene: 0 for gene in gene_list}
        else:
            res_tissue = res

        res_df = pd.DataFrame({
            'res_values': [item for sublist in res.values() for item in sublist.values()],
            'res_ind': [key for key, sublist in res.items() for _ in sublist]
        })

        res_tissue_df = pd.DataFrame({
            'tissue_values': [item for sublist in res_tissue.values() for item in sublist.values()],
            'tissue_ind': [f"{key}.{sub_key}" for key, sublist in res_tissue.items() for sub_key in sublist.keys()]
        })

        df = pd.concat([res_df, res_tissue_df], axis=1)

        df['tissue'] = df['tissue_ind'].str.extract("^[^.]*\\.([^.]*)\\..*$")[0]
        df = df.groupby([df.columns[4], df.columns[1]])[df.columns[0]].sum().reset_index()

        df = df[df['res_values'] > min_cells]

        if not df.empty:
            return df['res_ind'].value_counts().to_dict()
        else:
            return {gene: 0 for gene in gene_list}

    def findHouseKeepingGenes(self,
                              cell_types: List[str],
                              min_recall: float = 0.5,
                              max_genes: int = 1000
                              ) -> List[str]:
        """
        Find the set of genes that are ubiquitously expressed in a query of cell types.

        Parameters
        ----------
        cell_types: list of str
            A list of cell types to be evaluated. The format should be tissue(dataset).cell_type

        min_recall: float, default=0.5
            Threshold of minimum recall value. Defaults to 0.5.
            Must be a value between 0 and 1 (exclusive).

        max_genes: int, default=1000
            Threshold of number of genes to be considered for each cell type.
            Defaults to 1000. Must be a positive integer.

        Returns
        -------
        list of str
            A list of genes that are ubiquitously expressed in a query of cell types.

        Raises
        ------
        ValueError
            If number of cell_types less than 2, or if min_recall is not between 0 and 1 (exclusive), or
            if max_genes exceeds the limit.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        if len(cell_types) < 2:
            raise ValueError("Should input more than 2 cell types to identify housekeeping genes.")

        if not (0 < min_recall < 1):
            raise ValueError("min_recall reached limit, please use values > 0 and < 1.0.")

        if max_genes > len(self.index.genes()):
            raise ValueError(f"max.genes exceeded limit, please use values > 0 and < {len(self.index.genes())}")

        print("Searching for house keeping genes...")

        df = self.cellTypeMarkers(cell_types[0], top_k=max_genes, sort_field="recall")
        house_keeping_genes = df['genes'][df['recall'] > min_recall].tolist()
        house_keeping_genes = sorted(house_keeping_genes)

        for i, cell_type in tqdm(enumerate(cell_types[1:], 1)):
            df = self.cellTypeMarkers(cell_type, top_k=max_genes, sort_field="recall")
            current_genes = df['genes'][df['recall'] > min_recall].tolist()
            house_keeping_genes = list(set(house_keeping_genes).intersection(current_genes))

            if not house_keeping_genes:
                print("No house keeping gene is found.")
                return []

        house_keeping_genes = sorted(house_keeping_genes)
        return house_keeping_genes

    def findGeneSignatures(self,
                           cell_types: Optional[Union[str, List[str]]] = None,
                           max_genes: int = 1000,
                           min_cells: int = 10,
                           max_pval: float = 0.05
                           ) -> Union[Dict[str, List[str]], str]:
        """
        Find the list of gene signatures in a query of cell types.

        Parameters
        ----------
        cell_types: str or list of str, optional (default=None)
            Cell types to be evaluated.
            If not provided, all cell types from the index will be used.

        max_genes: int, default=1000
            Threshold of number of genes to be considered for each cell type. Defaults to 1000.

        min_cells: int, default=10
            Threshold of cell hit of a tissue. Defaults to 10.

        max_pval: float, default=0.05
            Threshold of p-value.

        Returns
        -------
        Union of dict of str or str
            A dictionary with cell types as keys and their corresponding gene signatures as values.
            If no cell types match the given list, a message string is returned.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling "
                             "object.buildCellTypeIndex().")

        print("Searching for gene signatures...")

        if cell_types is None:
            cell_types_all_stamp = self.index.getCellTypes()
            datasets = [d.split('_')[0] for d in cell_types_all_stamp]
            cts = [d.split('.')[1] for d in cell_types_all_stamp]
            cell_types_all = list(set([f"{d}.{ct}" for d, ct in zip(datasets, cts)]))

        else:
            if isinstance(cell_types, str):
                cell_types = [cell_types]

            cell_types_all = [cell_type for cell_type in self.cellTypeNames() if
                              cell_type.lower() in map(str.lower, cell_types)]

        signatures = {}

        try:
            if not cell_types_all:
                raise ValueError(f"Ignored {', '.join(cell_types)}. Cell type not found in index.")

            for cell_type in tqdm(cell_types_all, total=len(cell_types_all), desc="Processing cell types"):
                signatures[cell_type] = self._find_signature(cell_type, max_genes=max_genes, min_cells=min_cells,
                                                             max_pval=max_pval)
            return signatures

        except ValueError as e:
            return str(e)

    def findSimilarGenes(self,
                         gene_list: Union[str, List[str]],
                         datasets: Optional[Union[str, List[str]]] = None,
                         top_k: int = 5
                         ) -> pd.DataFrame:
        """
        Look at all other genes and rank them based on the similarity of their expression pattern
        to the pattern defined by the gene query.

        Parameters
        ----------
        gene_list : str or list of str
            Genes to be searched in the gene index.

        datasets : str or list of str, optional (default=None)
            The datasets that will be searched in.
            If datasets=None, all datasets from the index will be used.

        top_k : int, default=5
            How many genes to retrieve.

        Returns
        -------
        pd.DataFrame
            A dataframe containing genes and their similarities presented in Jaccard indices.
            The "overlap" column indicates the count of overlapping cell types that expressed both the specified
            genes and those from provided gene_list.
            The "n" column indicates the total number of cells that express both the specific gens and the genes
            in the provided gene_list.
        """

        if not self.index_exist:
            raise ValueError("SCFind index is not built. Please build index first by calling \
            object.buildCellTypeIndex().")

        print("Searching for genes with similar pattern...")

        if not datasets:
            datasets = self.datasets
        else:
            datasets = self._select_datasets(datasets)

        e = self.findCellTypes(gene_list, datasets)
        n_e = sum(len(sublist) for sublist in e.values())

        if n_e > 0:
            gene_names = list(set(self.index.genes()) - set(self._case_correct(gene_list)))
            similarities = [0] * len(gene_names)
            ns = [0] * len(gene_names)
            ms = [0] * len(gene_names)

            for i, gene_name in enumerate(tqdm(gene_names, desc="Processing genes", total=len(gene_names))):
                f = self.findCellTypes([gene_name], datasets)
                if f:
                    m = [len(set(e[name]) & set(f.get(name, []))) for name in e.keys()]
                    n_f = sum(len(sublist) for sublist in f.values())
                    similarities[i] = sum(m) / (n_e + n_f - sum(m))
                    ns[i] = n_f
                    ms[i] = sum(m)

            # Sorting the results and getting the top_k genes
            sorted_indices = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)[:top_k]
            res = pd.DataFrame({
                "gene": [gene_names[i] for i in sorted_indices],
                "Jaccard": [similarities[i] for i in sorted_indices],
                "overlap": [ms[i] for i in sorted_indices],
                "n": [ns[i] for i in sorted_indices]
            })
            return res

        else:
            print(f"Cannot find cells expressing {', '.join(gene_list)} in the index.")
            return pd.DataFrame(columns=['gene', 'Jaccard', 'overlap', 'n'])

    def de_genes(
            self,
            cell_type1: str,
            cell_type2: str,
            genes: Union[str, List[str]] = None,
            alpha: float = 0.05,
            min_fraction: float = 0.25,
            ) -> pd.DataFrame:
        """
        Identify differentially expressed (DE) genes between two cell types.
        For each gene, this function tests if the proportion of cells expressing the gene is significantly different
        between the two cell types.

        Parameters
        ----------
        cell_type1: str
            the first cell type to study DE genes

        cell_type2: str
            the second cell type to study DE genes

        genes: str or list of str, optional
            List of genes to analyze. If None, all genes are used.

        alpha : float, optional
            Significance level for statistical tests. Default is 0.05.

        min_fraction: float, default 0.25
            Portion of total cell expressed DE genes as threshold.

        Returns
        -------
        A dataframe containing the DE genes.
        """
        valid_cell_types = self.cellTypeNames()
        if not (cell_type1 in valid_cell_types and cell_type2 in valid_cell_types):
            raise ValueError("Input cell types are not valid in index. \n"
                             "Check all cell type names by index.cellTypeNames()")
        if genes is not None:
            valid_genes = self._case_correct(genes)
        else:
            valid_genes = self.scfindGenes

        cell_type1_stamp = self._select_celltype_stamp(cell_type1)
        cell_type2_stamp = self._select_celltype_stamp(cell_type2)

        results = self.index.DEGenes(cell_type1_stamp, cell_type2_stamp, valid_genes, min_fraction)
        if len(results) == 0:
            print("No significant genes identified between given cell types.")
            return pd.DataFrame(
                columns=["gene", "proportion_1", "proportion_2", "abs",
                         "difference", "p_value", "test_used", "adj_p_value", "de_in"
                         ])
        else:
            results_df = pd.DataFrame(results).sort_values('p_value', ignore_index=True)
            adjusted_pvals = multipletests(results_df['p_value'], method='holm')[1]
            results_df['adj_p_value'] = adjusted_pvals
            results_df = results_df[results_df['adj_p_value'] < alpha]
            results_df['de_in'] = np.where(
                (results_df['proportion_1'] > results_df['proportion_2']),
                cell_type1,
                np.where(
                    (results_df['proportion_2'] > results_df['proportion_1']),
                    cell_type2,
                    None
                )
            )
            return results_df


















    @staticmethod
    def _buildCellTypeIndex(adata: AnnData,
                            dataset_id: str,
                            tissue: str,
                            dataset_index: Optional[int] = None,
                            feature_name: str = 'feature_name',
                            cell_type_label: str = 'cell_type',
                            qb: int = 2,
                            ) -> 'SCFind':
        """
        Build a SCFind index.

        Parameters
        ----------
        adata: AnnData
            The annotated data matrix of shape (n_obs, n_vars). Rows correspond to cells
            and columns to genes.

        dataset_id: str
            Dataset ID obtained from HuBMAP.

        tissue: str
            Tissue name.

        dataset_index: Optional[int], default=None
            Index of the dataset in the SCFind object. If not provided, it will be set as 0.
            This is used to update index for HuBMAP dataset. By checking the number of datasets in old index,
            we can set the index of the new dataset to distinguish duplicate cell types.

        feature_name: str, default='feature_name'
            The label or key in the AnnData object's variables (var) that corresponds to the feature names.

        cell_type_label: str, default='cell_type'
            The label or key in the AnnData object's observations (obs) that corresponds to the cell type.

        qb: int, default=2
            Number of bits per cell that are going to be used for quantile compression of the expression data.

        Returns
        -------
        A SCFind object.
        """
        scf_object = SCFind()
        scf_object.buildCellTypeIndex(
            adata=adata, 
            dataset_id=dataset_id,
            tissue=tissue,
            dataset_index=dataset_index,
            feature_name=feature_name,
            cell_type_label=cell_type_label, 
            qb=qb)
        return scf_object

    def _select_datasets(self,
                         datasets: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Validates and selects datasets for further operations.

        Parameters
        ----------
        datasets: string or list of strings, optional
            List of datasets to validate against the available datasets in the class.
            If none, all available datasets are selected.

        Returns
        -------
            List[str]: the list of validated datasets to be used.

        Raises
        ------
        Exception:
            If none of the specified datasets exists in the available datasets list.
        """
        if datasets is None:
            return self.datasets
        elif isinstance(datasets, str):
            datasets = [datasets]

        missing_datasets = set(datasets).difference(self.datasets)
        if missing_datasets:
            print(f"Dataset(s) {', '.join(missing_datasets)} do not exist in the database. Ignore it.")
        
        datasets = list(set(datasets).intersection(self.datasets))

        if len(datasets) == 0:
            raise ValueError("None of the input datasets are valid. Please check valid datasets by index.datasets")

        return datasets
    
    def _select_celltype_stamp(self,
                               cell_type: Union[str, List[str]] = None,
                               ):
        """
        Select dataset_name_stamp.celltype for input cell_type (format: dataset.cell_type)

        Parameters
        ----------
        cell_type: string or list of strings, optional
            List of cell types without stamp
        
        Returns
        -------
            List of cell types with stamp

        Raises
        ------
        Exception:
            If none of the specified datasets exists in the available datasets list.
        """
        if cell_type is None:
            cell_type_stamp = self.index.getCellTypes()
            return cell_type_stamp

        if isinstance(cell_type, str):
            cell_type = [cell_type]
        
        datasets = [d.split('.')[0] for d in cell_type]
        datasets = self._select_datasets(datasets)

        cts = [ct.split('.')[1] for ct in cell_type]

        dataset_stamp = [d for ds in datasets for d in self.datasets_map[ds]]
        dataset_id = self.datasetID_map['dataset_id'].isin(dataset_stamp)
        dataset_id = self.datasetID_map[dataset_id].index
        cell_type_stamp = self.index.getCellTypes()
        filter_cell_type_stamp = [ct for ct in cell_type_stamp if ct.split('.')[1] in cts and ct.split('.')[0] in dataset_id]

        return filter_cell_type_stamp

    def _case_correct(self,
                      gene_list: Union[str, List[str]],
                      if_print: bool = True,
                      ) -> List[str]:
        """
        Corrects the gene list to match the genes in index.

        Parameters
        ----------
        gene_list: str or list of str
            Gene of a list of genes to be corrected.

        if_print: bool, default=True
            Whether print gene searching information

        Returns
        -------
        A list of corrected gene names.
        """

        if isinstance(gene_list, str):
            gene_list = [gene_list]

        # Filtering malicious inputs
        if all(isinstance(item, str) for item in gene_list):

            db_genes = self.index.genes()

            # Convert to lowercase
            normalized_query_genes = [gene.lower() for gene in gene_list]
            normalized_db_genes = [gene.lower() for gene in db_genes]

            # Identify matches and misses
            matches = [db_gene for db_gene in db_genes if db_gene.lower() in normalized_query_genes]
            misses = [gene for gene in gene_list if gene.lower() not in normalized_db_genes]

            if matches:
                if misses and if_print:
                    print(f"Ignoring {', '.join(misses)}. Not valid gene name(s).")
                return list(set(matches))

        return []

    @staticmethod
    def _pair_id(cell_list: Dict[str, List[Union[str, int]]]) -> List[str]:
        """
        Generate a list of strings by combining the dictionary keys with their corresponding values.

        Parameters
        ----------
        cell_list:
            A dictionary where keys are strings and values are lists of strings or integers.

        Returns
        -------
        List[str]:
            A list of strings where each string is formed by concatenating a key from the dictionary with a value from
            its corresponding list, separated by a "#".
        """

        if not cell_list:
            return []

        pair_vec = [(key, value) for key, values in cell_list.items() for value in values]
        return [f"{key}#{value}" for key, value in pair_vec]

    def _find_signature(self,
                        cell_type: str,
                        max_genes: int = 1000,
                        min_cells: int = 10,
                        max_pval: float = 0.05
                        ) -> List[str]:
        """
        Use this method to find a gene signature for a cell type.
        We do this by ranking genes by recall and then adding genes to the query until we exceed a target p-value
        threshold or until a minimum number of cells is returned from the query.

        Parameters
        ----------
        cell_type: str
            Cell type name without stamp.

        max_genes: int, default=1000
            The maximum number of genes. Default is 1000.

        min_cells: int, default=10
            The minimum number of cells. Default is 10.

        max_pval: float, default=0.05
            The maximum p-value. Default is 0.05.

        Returns
        -------
        A list of genes.
        """

        df = self.cellTypeMarkers([cell_type], top_k=max_genes, sort_field="recall")
        genes = [str(gene) for gene in df['genes']]
        genes_list = []
        cell_type_stamp = self._select_celltype_stamp(cell_type)
        total_cells = np.sum([self.index.getCellTypeMeta(ct)['total_cells'] for ct in cell_type_stamp])
        thres = min(min_cells, total_cells)

        for j in range(len(df)):

            res = self.hyperQueryCellTypes(genes_list + [genes[j]])

            if not res.empty:
                ind = res['cell_type'] == cell_type

                if not ind.any():
                    break
                else:
                    if (res[ind].loc[:, 'adj-pval'] > max_pval).all() or (res[ind].iloc[:, 1] < thres).any():
                        break

            genes_list.append(genes[j])

        return genes_list

    def _phyper_test(self,
                     result: Dict[str, List[int]],
                     ) -> pd.DataFrame:
        """
        Performs a hypergeometric test to assess the statistical significance of the number of observed non-zero

        Parameters
        ----------
        result:
            A dictionary containing the cells types as keys and list of cell IDs as values.

        Returns
        -------
        A dataframe with hypergeometric test result.

        """

        # Convert the result to a dataframe
        df = self._result_to_dataframe(result)

        # Aggregate by cell_type_stamp
        cell_types_df = df.groupby('cell_type').size().reset_index(name='cell_hits')
        cell_types_df['cell_type'] = cell_types_df['cell_type'].str.split('_').str[0] + "." + cell_types_df['cell_type'].str.split('.').str[1]
        cell_types_df = cell_types_df.groupby('cell_type', as_index=False)['cell_hits'].sum()


        # Get total_cells for each cell type
        cell_types_df['total_cells'] = cell_types_df['cell_type'].apply(lambda x: np.sum(self.index.getCellTypeSupport(self._select_celltype_stamp(x), True)))

        query_hits = len(df)

        # Calculate the hypergeometric test p-values
        cell_types_df['pval'] = 1 - hypergeom.cdf(
            cell_types_df['cell_hits'],
            cell_types_df['total_cells'].sum(),
            cell_types_df['total_cells'],
            query_hits
        )

        # Adjust p-values using Holm adjustment method
        adjusted_pvals = multipletests(cell_types_df['pval'], method='holm')[1]
        cell_types_df['adj-pval'] = adjusted_pvals

        return cell_types_df

    @staticmethod
    def _result_to_dataframe(result: Dict[str, Union[int, List[int]]]) -> pd.DataFrame:
        """
        Converts a query result into a pandas DataFrame.
        This function takes a query result, which can be a dictionary with cell types as keys and lists of cell IDs as
        values, or an existing pandas DataFrame, and converts it into a structured DataFrame with 'cell_type' and
        'cell_id' columns.

        Parameters
        ----------
        result:
            A dictionary containing the cells types as keys and list of cell IDs as values.
            It should have cell types as keys and lists of cell IDs as values. If it's a DataFrame, it will be returned
            as is.

        Returns
        -------
            pd.DataFrame: A DataFrame with two columns: 'cell_type' and 'cell_id'.

        Raises:
        ------
            TypeError: If 'result' is neither a dictionary nor a pandas DataFrame.

        """

        # If query_result is already a dataframe
        if isinstance(result, pd.DataFrame):
            return result

        # If result is empty
        if len(result) == 0:
            return pd.DataFrame({'cell_type': [], 'cell_id': []})

        # Else, process the result to convert to a dataframe
        else:
            cell_type = []
            cell_id = []

            for key, values in result.items():
                cell_type.extend([key] * len(values))
                cell_id.extend(values)

            return pd.DataFrame({'cell_type': cell_type, 'cell_id': cell_id})
