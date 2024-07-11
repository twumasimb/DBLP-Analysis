import networkx as nx


def compute_closeness_centrality(network, node=None):
    if node is None:
        closeness_centrality = nx.closeness_centrality(network, distance="weight")
        return closeness_centrality
    else:
        closeness_centrality = nx.closeness_centrality(network, distance="weight")
        return closeness_centrality[node] * 100


def compute_degree_centrality(network, node):
    if node is None:
        degree_centrality = nx.degree_centrality(network)
        return degree_centrality
    else:
        degree_centrality = nx.degree_centrality(network)
        return degree_centrality[node] * 100


def compute_betweeness_centrality(network, node=None):
    if node is None:
        closeness_centrality = nx.betweenness_centrality(network)
        return closeness_centrality
    else:
        closeness_centrality = nx.betweenness_centrality(network)
        return closeness_centrality[node] * 100


def compute_eigenvector_centrality(network, node):
    if node is None:
        eigenvector_centrality = nx.eigenvector_centrality(network)
        return eigenvector_centrality
    else:
        eigenvector_centrality = nx.eigenvector_centrality(network)
        return eigenvector_centrality[node] * 100


def compute_pagerank(network, node):
    if node is None:
        pagerank = nx.pagerank(network)
        return pagerank
    else:
        pagerank = nx.pagerank(network)
        return pagerank[node] * 100
