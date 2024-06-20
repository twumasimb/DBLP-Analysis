import networkx as nx

def compute_closeness_centrality(network, node):
    
    closeness_centrality = nx.closeness_centrality(network)
    return closeness_centrality[node] * 100

def compute_degree_centrality(network, node):
    
    degree_centrality = nx.degree_centrality(network)
    return degree_centrality[node] * 100

def compute_eigenvector_centrality(network, node):
    
    eigenvector_centrality = nx.eigenvector_centrality(network)
    return eigenvector_centrality[node] * 100

def compute_pagerank(network, node):
    
    pagerank = nx.pagerank(network)
    return pagerank[node] * 100