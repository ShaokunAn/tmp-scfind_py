import scfind

# Load index (HuBMAP index)
with open('/Users/sa3520/BWH/scfind_py_dev/test/index_serialized.bin', 'rb') as f:
    index_serialized = f.read()

index = scfind.SCFind()
index.index.loadByteStream(index_serialized)
index.datasets = ['Kidney', 'Lung']
index.index_exist = True

# Load word2vec model
w2v_path = "/Users/sa3520/BWH/word2vec/w2v/PubMed-w2v.bin"
dic_path = "/Users/sa3520/BWH/word2vec/w2v/scfind_dictionary_hs_v1.h5"

dic = scfind.read_all_dictionaries(w2v_path, dic_path)

# Test free text search function
query = "diabetes and TNF, rs1826906"
query = "TNF, TP53, CD44"
query = "TNF, TP63, MDM2"
query = "TNF or rs3218009"
query = "MESH:C556802"
query = "MESH:C022139,MESH:C023489,MESH:C059597,MESH:C000231"
print(scfind.query2genes(index=index, dictionary=dic, query=query, strict=False))

