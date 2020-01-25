import numpy as np
import networkx as nx


#extract the barcode from the main file
def extractBarcode(ndarray):
    first_line = ndarray[0]
    barcode = np.delete(first_line, 0)
    return barcode
#extract the ID of the RNAseq from the main file
def extractID(ndarray):        
    first_column = ndarray[:,0]
    ID = np.delete(first_column, 0)
    return ID
#extract the RNAseq data from the main file
def rnaseqData(ndarray):        
    remove_first_row = np.delete(ndarray, 0, axis=0)
    remove_first_column = np.delete(remove_first_row, 0, axis=1)
    RNAseqData = remove_first_column.astype(np.float)
    return RNAseqData
#divide the string value from the numerical value of the ID
def id_Division(ndarray):        
    geneS = []
    geneI = []
    for gene in ndarray:
        token = gene.split("|")
        geneS.append(token[0])
        geneI.append(token[1])
    return np.array(geneS), np.array(geneI)
#extract from the RNAseq data the matching column of a 4-way-data file
def extractMatchingColumn(pats, listP, RNAseqData):
    pos = np.isin(pats, listP)
    data = RNAseqData[:,pos]
    return data
#calculating the fold-change and its mean of the cancer patients vs the control patients, adding 1 to both lists to avoid log0
def foldchange(cancer, control):
    fold_change = np.log2(np.divide((cancer+1), (control+1)))
    mean_fold_change = np.mean(fold_change, axis=1)
    return fold_change, mean_fold_change
#replace zeros with np.nan
def nan(ndarray):        
    ndarray[ndarray == 0] = np.nan
    return ndarray
#replace np.nan with zeros
def zeros(ndarray):        
    array_with_zeros = np.where(np.isnan(ndarray), 0, ndarray)
    return array_with_zeros
#normalization of the data
def normalization(ndarray):        
    normalized_data = np.log10(ndarray)
    return normalized_data
#sorting the value of a list but returning a collection of index vectors for any of the previous syntaxes.
def  sortDictValue(ListOrDict):
	if type(ListOrDict) is list:
		ListOrDict = dict(enumerate(ListOrDict, start=0))
    sorted_idxdict = {k: v for k, v in sorted(ListOrDict.items(), key=lambda item: item[1])}
    sorted_value = list(sorted_idxdict.keys())
    return sorted_value

#taking the first M element of an array, based on the first M element of another, checking if the array is mono or multidimensional
def takeM(ndarray, index, M: int):
    if ndarray.ndim == 1:
        data = [ndarray[j] for j in index[:M]]
    else: data = [ndarray[j,:] for j in index[:M]]
    return data
#create a correlation matrix row-wise
def corrMatrixRow(ndarray):        
    corrMatrix = np.corrcoef(ndarray, rowvar=True)
    return corrMatrix
    #create a correlation matrix column-wise
def corrMatrixColumn(ndarray):        
    corrMatrix = np.corrcoef(ndarray, rowvar=False)
    return corrMatrix
#creating an adjacency matrix replacing any element of the correlation matrix that is less or equal the 99.9 percentile
#of its row, and with 1 if is strictly greater
def adjacencyMatrix(ndarray, ndarraytovect, perc):
    ndarray_zero = np.where(np.isnan(ndarray), 0, ndarray)
    ndarray_zero[ndarray_zero > np.nanpercentile(ndarraytovect.flatten(), perc)] = 1
    ndarray_zero[ndarray_zero <= np.nanpercentile(ndarraytovect.flatten(), perc)] = 0
    return ndarray_zero
#create a subgraph of the biggest connected component of a graph
def createSubgraph(graph):
    if not nx.is_connected(graph):
        sub_graph = [graph.subgraph(c) for c in nx.connected_components(graph)]
        main_graph = sub_graph[0]
        for sg in sub_graph:
            if len(sg.nodes()) > len(main_graph.nodes()):
                main_graph = sg
        subgraph = main_graph
        return subgraph