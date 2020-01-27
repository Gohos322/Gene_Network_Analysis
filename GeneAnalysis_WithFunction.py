from multipy.fdr import qvalue
from scipy import stats
import modeling.GNA as gna
import networkx as nx
import numpy as np


#extract the patient barcodes, the gene IDs and the RNAseq datas from the main file
AllData = np.genfromtxt("C:/Users/Lorenzo/Documents/Uni/Modeling/Matrici_TCGA 2014/brca/brca_RNASeq/matrice__brca_RNASeq.txt", delimiter='\t', dtype=None, encoding=None)
#extract the patient barcodes from the 4-way-data patients list
listPN = np.genfromtxt("C:/Users/Lorenzo/Documents/Uni/Modeling/Matrici_TCGA 2014/brca/Patients_with_4wayData/Lista__RNASeq_Normal__brca__4wayData.txt", delimiter='\n', dtype=None, encoding=None)
listPC = np.genfromtxt("C:/Users/Lorenzo/Documents/Uni/Modeling/Matrici_TCGA 2014/brca/Patients_with_4wayData/Lista__RNASeq_Tumor__brca__4wayData.txt", delimiter='\n', dtype=None, encoding=None)


#extract the barcode
pats = gna.extractBarcode(AllData)
#extract the gene IDs
gene_list = gna.extractID(AllData)
#extract in geneS the string section of the gene ID, and in geneI the integer part
geneS, geneI = gna.id_Division(gene_list)
#extract the RNAseq datas
RNAseqData = gna.rnaseqData(AllData)
#from the RNAseq datas extract the columns that match the one of the 4-way-data patients list
dataN = gna.extractMatchingColumn(pats, listPN, RNAseqData)
dataC = gna.extractMatchingColumn(pats, listPC, RNAseqData)
#computing the Fold Change and its mean
FC, mFC = gna.foldchange(dataC, dataN)
#replacing the zeros with Not-a-Number (nan)
dataN = gna.nan(dataN)
dataC = gna.nan(dataC)
#normalization of the data via log10
dataNlg = gna.normalization(dataN)
dataClg = gna.normalization(dataC)
#two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. 
#This test assumes that the populations have identical variances by default
h,pvals = stats.ttest_ind(dataNlg, dataClg, axis=1, nan_policy='omit')
#estimating q-values from p-values using the Storey-Tibshirani q-value method
significants, qvals = qvalue(pvals, threshold=0.05, verbose=False)
#sorting the (q)value of a list but returning a collection of index vectors for any of the previous syntaxes
index = gna.sortDictValue(qvals)
#taking the first M element of an array, based on the first M element of another, checking if the array is mono or multidimensional
geneSb = gna.takeM(geneS, index, 5000)
geneIb = gna.takeM(geneI, index, 5000)
dataNb = gna.takeM(dataN, index, 5000)
dataCb = gna.takeM(dataC, index, 5000)
#Return Pearson product-moment correlation coefficients
corrN = gna.corrMatrixRow(dataNb)
corrC = gna.corrMatrixRow(dataCb)
#creating an adjacency matrix replacing any element of the correlation matrix that is less or equal the 99.9 percentile
#of its row, and with 1 if is strictly greater
corrN_adj = gna.adjacencyMatrix(corrN, corrN, 99.9)
corrC_adj = gna.adjacencyMatrix(corrC, corrC, 99.9)
#creating the graph of the adjacency matrix
GN = nx.Graph(corrN_adj)
GC = nx.Graph(corrC_adj)
#creating a subgraph of the biggest connected component of the main graph
SGN = gna.createSubgraph(GN)
SGC = gna.createSubgraph(GC)
#estimating the clustering coefficient
clustN = nx.average_clustering(SGN)
clustC = nx.average_clustering(SGC)
#measuring the transitivity. Higher values of clustering coefficients and transitivity may indicate 
#that the graph exhibits the small-world effect
transN = nx.transitivity(SGN)
transC = nx.transitivity(SGC)
#measuring the betweennes, or the centrality in our graph based on shortest paths. Higher degree on a vertex
#means that that point is hightly crossed by s.p.
bet_cenN = nx.betweenness_centrality(SGN, endpoints = True)
bet_cenC = nx.betweenness_centrality(SGC, endpoints = True)
#measuring centrality of the vertices. The most important ones are the ones with higher value
centralityN = nx.degree_centrality(SGN)
centralityC = nx.degree_centrality(SGC)
#measuring the eigencentrality. This is the degree of influence of a point in the graph.
#Could be seen as a conceptual sum of centrality and betweennes
eigencentN = nx.eigenvector_centrality(SGN)
eigencentC = nx.eigenvector_centrality(SGC)
#it's going to create a dict by returning the eigencentrality values sorted crescently,
#keeping the previous index of each value in the new and "correct" position
eigenListN = gna.sortDictValue(eigencentN)
eigenListC = gna.sortDictValue(eigencentC)
#measuring the shortest paths of the most influecing vertices of the graph, after putting them into
#a list(required by the nx function)
spN = nx.shortest_path(SGN, eigenListN[-1], eigenListN[-2])
spC = nx.shortest_path(SGC, eigenListC[-1], eigenListC[-2])
#measuring the reciprocal of the sum of the length of the shortest paths
#between the node and all other nodes in the graph.
#In other words, the closer it is to all other nodes.
closenessN = nx.closeness_centrality(SGN)
closenessC = nx.closeness_centrality(SGC)