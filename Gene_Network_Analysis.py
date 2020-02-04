from multipy.fdr import qvalue  #At the moment i'm writing this, this library installed with pip is not compatible with python3, but the version on github is
from scipy import stats
import modeling.GNA as gna
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import warnings


#remove the warnings due to the NaN
warnings.filterwarnings("ignore", category=RuntimeWarning) 
#------------------------------------------------------------------------------ 

#creating a progressbar
bar = progressbar.ProgressBar(maxval=20, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

#------------------------------------------------------------------------------ 

#extract the patient barcodes, the gene IDs and the RNAseq datas from the main file
AllData = np.genfromtxt("insert here the cancer RNA-seq matrix. It should be a tsv, txt file", delimiter='\t', dtype=None, encoding=None)
#extract the patient barcodes from the 4-way-data patients list
listPN = np.genfromtxt("insert here the barcode list of Normal patients. It should be a txt file", delimiter='\n', dtype=None, encoding=None)
listPC = np.genfromtxt("insert here the barcode list of Cancer patients. It should be a txt file", delimiter='\n', dtype=None, encoding=None)

bar.update(1)

#extract the barcode
pats = gna.extractBarcode(AllData)
#extract the gene IDs
gene_list = gna.extractID(AllData)

bar.update(2)

#extract in geneS the string section of the gene ID, and in geneI the integer part
geneS, geneI = gna.id_Division(gene_list)
#extract the RNAseq datas
RNAseqData = gna.rnaseqData(AllData)

bar.update(3)

#from the RNAseq datas extract the columns that match the one of the 4-way-data patients list
dataN = gna.extractMatchingColumn(pats, listPN, RNAseqData)
dataC = gna.extractMatchingColumn(pats, listPC, RNAseqData)

bar.update(4)

#computing the Fold Change and its mean
FC, mFC = gna.foldchange(dataC, dataN)
#replacing the zeros with Not-a-Number (nan) to avoid impropriate correlation
dataN = gna.nan(dataN)
dataC = gna.nan(dataC)

bar.update(5)

#normalization of the data via log10
dataNlg = gna.normalization(dataN)
dataClg = gna.normalization(dataC)

bar.update(6)

#two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. 
#This test assumes that the populations have identical variances by default
h,pvals = stats.ttest_ind(dataNlg, dataClg, axis=1, nan_policy='omit')
#estimating q-values from p-values using the Storey-Tibshirani q-value method
significants, qvals = qvalue(pvals, threshold=0.05, verbose=False)

bar.update(7)

#sorting the (q)value of a list but returning a collection of index vectors for any of the previous syntaxes
index = gna.sortDictValue(qvals)
#taking the first M element of an array, based on the first M element of another, checking if the array is mono or multidimensional
geneSb = gna.takeM(geneS, index, 5000)
geneIb = gna.takeM(geneI, index, 5000)
dataNb = gna.takeM(dataN, index, 5000)
dataCb = gna.takeM(dataC, index, 5000)

bar.update(8)

#Return Pearson product-moment correlation coefficients
corrN = gna.corrMatrixRow(dataNb)
corrC = gna.corrMatrixRow(dataCb)

bar.update(9)

#creating an adjacency matrix replacing any element of the correlation matrix that is less or equal the 99.9 percentile
#of its row, and with 1 if is strictly greater
corrN_adj = gna.adjacencyMatrix(corrN, corrN, 99.9)
corrC_adj = gna.adjacencyMatrix(corrC, corrC, 99.9)

bar.update(10)

#creating the graph of the adjacency matrix
GN = nx.Graph(corrN_adj)
GC = nx.Graph(corrC_adj)

bar.update(11)

#creating a subgraph of the biggest connected component of the main graph
SGN = gna.createSubgraph(GN)
SGC = gna.createSubgraph(GC)

bar.update(12)

#estimating the clustering coefficient
clustN = nx.average_clustering(SGN)
clustC = nx.average_clustering(SGC)

bar.update(13)

#measuring the transitivity. Higher values of clustering coefficients and transitivity may indicate 
#that the graph exhibits the small-world effect
transN = nx.transitivity(SGN)
transC = nx.transitivity(SGC)

bar.update(14)

#measuring the betweennes, or the centrality in our graph based on shortest paths. Higher degree on a vertex
#means that that point is hightly crossed by s.p.
bet_cenN = nx.betweenness_centrality(SGN, endpoints = True)
bet_cenC = nx.betweenness_centrality(SGC, endpoints = True)

bar.update(15)

#measuring centrality of the vertices. The most important ones are the ones with higher value
centralityN = nx.degree_centrality(SGN)
centralityC = nx.degree_centrality(SGC)

bar.update(16)

#measuring the eigencentrality. This is the degree of influence of a point in the graph.
#Could be seen as a conceptual sum of centrality and betweennes
eigencentN = nx.eigenvector_centrality(SGN)
eigencentC = nx.eigenvector_centrality(SGC)

bar.update(17)

#it's going to create a dict by returning the eigencentrality values sorted crescently,
#keeping the previous index of each value in the new and "correct" position
eigenListN = gna.sort_dict(eigencentN)
eigenListC = gna.sort_dict(eigencentC)

bar.update(18)

#measuring the shortest paths of the most influecing vertices of the graph, after putting them into
#a list(required by the nx function)
spN = nx.shortest_path(SGN, eigenListN[-1], eigenListN[-2])
spC = nx.shortest_path(SGC, eigenListC[-1], eigenListC[-2])

bar.update(19)

#measuring the reciprocal of the sum of the length of the shortest paths
#between the node and all other nodes in the graph.
#In other words, the closer it is to all other nodes.
closenessN = nx.closeness_centrality(SGN)
closenessC = nx.closeness_centrality(SGC)
bar.finish()

#------------------------------------------------------------------------------ 

#drawing a scatterplot of the betweeness centrality values
lists = sorted(bet_cenN.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
fig, axes = plt.subplots(1, 2)
fig.suptitle("Betweeness Centrality of SGN and SGC")
axes[0].scatter(x,y)
axes[0].set(xlabel="Nodes", ylabel="Values")

lists = sorted(bet_cenC.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
axes[1].scatter(x,y)
axes[1].set(xlabel="Nodes", ylabel="Values") 
plt.show()

#------------------------------------------------------------------------------ 

#drawing a scatter plot of the eigenvalues
lists = sorted(eigencentN.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
fig, axes = plt.subplots(1, 2)
fig.suptitle("Eigenvalues of SGN and SGC")
axes[0].scatter(x,y)
axes[0].set(xlabel="Nodes", ylabel="Values")

lists = sorted(eigencentC.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
axes[1].scatter(x,y)
axes[1].set(xlabel="Nodes", ylabel="Values") 
plt.show()

#------------------------------------------------------------------------------ 

#drawing a scatterplot of the degree centrality values
lists = sorted(centralityN.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
fig, axes = plt.subplots(1, 2)
fig.suptitle("Degree Centrality of SGN and SGC")
axes[0].scatter(x,y)
axes[0].set(xlabel="Nodes", ylabel="Values")

lists = sorted(centralityC.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
axes[1].scatter(x,y)
axes[1].set(xlabel="Nodes", ylabel="Values") 
plt.show()

#------------------------------------------------------------------------------ 

#drawing a scatterplot of the closeness centrality values
lists = sorted(closenessN.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
fig, axes = plt.subplots(1, 2)
fig.suptitle("Closeness Centrality of SGN and SGC")
axes[0].scatter(x,y)
axes[0].set(xlabel="Nodes", ylabel="Values")

lists = sorted(closenessC.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
axes[1].scatter(x,y)
axes[1].set(xlabel="Nodes", ylabel="Values") 
plt.show()

#------------------------------------------------------------------------------ 

#setting options for visualizations and layout style of the graph
options = {"node_size": 100, "alpha": 0.8}
layoutN = nx.spring_layout(SGN)
layoutC = nx.spring_layout(SGC)

#drawing the SGN subgraph
nx.draw_networkx_nodes(SGN, layoutN, node_color="b" , **options)
path_edges = [tuple(spN)]
nx.draw_networkx_edges(SGN, layoutN, width=1.0, alpha=0.5)
nx.draw_networkx_edges(SGN, layoutN, edgelist=path_edges, width=5, alpha=0.5, edge_color="r")
nx.draw_networkx_labels(SGN, layoutN, font_size=10)
plt.axis('off')
textstr = '\n'.join((
    'Clustering Coefficient= %.4f ' % (clustN, ),
    'Transitivity = %.4f' % (transN, )))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.50, 0.90, textstr, fontsize=8, verticalalignment='top', bbox=props)
plt.show()

#drawing the SGC subgraph
nx.draw_networkx_nodes(SGC, layoutC, node_color="b" , **options)
path_edges = [tuple(spC)]
nx.draw_networkx_edges(SGC, layoutC, width=1.0, alpha=0.5)
nx.draw_networkx_edges(SGC, layoutC, edgelist=path_edges, width=5, alpha=0.5, edge_color="r")
nx.draw_networkx_labels(SGC, layoutC, font_size=10)
plt.axis('off')
textstr = '\n'.join((
    'Clustering Coefficient= %.4f ' % (clustC, ),
    'Transitivity = %.4f' % (transC, )))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.50, 0.90, textstr, fontsize=8, verticalalignment='top', bbox=props)
plt.show()
