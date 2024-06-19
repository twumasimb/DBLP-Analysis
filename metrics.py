import networkx as nx

def compute_betweenness_centrality(network, node):
    
    betweenness_centrality = nx.betweenness_centrality(network)
    return betweenness_centrality[node]

def compute_closeness_centrality(network, node):
    
    closeness_centrality = nx.closeness_centrality(network)
    return closeness_centrality[node]

def compute_degree_centrality(network, node):
    
    degree_centrality = nx.degree_centrality(network)
    return degree_centrality[node]

def compute_eigenvector_centrality(network, node):
    
    eigenvector_centrality = nx.eigenvector_centrality(network)
    return eigenvector_centrality[node]

def compute_katz_centrality(network, node):
    
    katz_centrality = nx.katz_centrality(network)
    return katz_centrality[node]

def compute_pagerank(network, node):
    
    pagerank = nx.pagerank(network)
    return pagerank[node]   

def compute_clustering_coefficient(network, node):  
    
    clustering_coefficient = nx.clustering(network)
    return clustering_coefficient[node]