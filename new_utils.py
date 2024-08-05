import pickle
import random
import networkx as nx
random.seed(42)
import math as mt

# Get the subgraph of a given node in G based on the labels in P
def subgraph_project(G, P, node_g):
    # Get the label of the given node in G
    label_g = G.nodes[node_g]['label']
    
    # Find all connected labels in P
    connected_labels = set()
    for edge in P.edges(label_g):
        connected_labels.add(edge[1])
    for edge in P.edges():
        if edge[1] == label_g:
            connected_labels.add(edge[0])
    
    # Add the original label to the set
    # connected_labels.add(label_g) # I am taking this out but I will add the node to the set
    
    # Select all nodes in G with labels in connected_labels
    selected_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] in connected_labels]
    selected_nodes.append(node_g) # Add the node to the network but ignoring all the people from his group.
    
    # Generate the subgraph of the selected nodes
    subgraph = G.subgraph(selected_nodes).copy()
    
    return subgraph


def subgraph_by_same_label(G, node_g):
    # Get the label of the given node in G
    label_g = G.nodes[node_g]['label']
    
    # Select all nodes in G with the same label as node_g
    selected_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] == label_g]
    
    # Generate the subgraph of the selected nodes
    subgraph = G.subgraph(selected_nodes).copy()
    
    return subgraph

def inteamEff(G, source):
    """
    Returns the inverse closeness centrality (the higher the better)
    """
    return round(G.nodes[source]['closeness_centrality'], 4)

def crossTeamEff(G, node, target_nodes:list):
    total = 0.0
    for target in target_nodes:
        shortest_path = nx.shortest_path(G, node, target)
        len_shortest_path = len(shortest_path)
        sumDistance = nx.dijkstra_path_length(G, node, target, weight='weight')
        closeness = (len_shortest_path)/sumDistance
        total += closeness
        # print(f"{node} --> {target}: length: {len_shortest_path}, distance: {sumDistance}, closeness: {closeness}")
    return round(total/len(target_nodes), 4) 

# def get_top_node_from_each_group(graph_G, graph_P):
#     top_nodes = []
#     for label in graph_P.nodes:
#         team_members = [n for n, attr in graph_G.nodes(
#             data=True) if attr['label'] == label]
#         team_graph = graph_G.subgraph(team_members)
#         centrality = nx.closeness_centrality(team_graph)
#         ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
#         # print(ranked_nodes) #prints out the centrality of the nodes 
#         top_node = ranked_nodes[0] if ranked_nodes else None
#         top_nodes.append(top_node)

#     return top_nodes

def influencial_nodes(G):
    """Top nodes have centrality of 1.0"""
    return [node for node in G.nodes if G.nodes[node]['closeness_centrality'] == 1.0]

def Greedy(graph_G, teams:list, seed_node):
    if graph_G is None:
        RuntimeError("One or Both of the graphs is None! ")

    subset = set()
    subset.add(seed_node)
    labels = []
    labels.append(graph_G.nodes[seed_node]['label'])
    communication_efficiency = 0.0

    while len(subset) < len(teams):
        best_node = None
        max_eff = float('-inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['label'] not in labels:
                temp_subset.add(node)

                # In-team efficiency
                in_team_eff = inteamEff(graph_G, node)

                #cross-team efficiency
                cross_team_eff = crossTeamEff(graph_G, node, list(subset))

                total_node_eff = in_team_eff + (cross_team_eff * 10)

                if total_node_eff > max_eff:
                    max_eff = total_node_eff
                    best_node = node
                    # print(f"node: {best_node}, inteam score: {in_team_eff}, cross-team score: {cross_team_eff}")

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += max_eff
    
    # return subset, sum_edge_weights(graph_G.subgraph(subset))
    return subset, round(communication_efficiency, 4)

