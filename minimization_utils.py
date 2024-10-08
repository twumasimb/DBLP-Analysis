import pickle
import random
import networkx as nx
random.seed(42)
import math as mt

def get_node_rank(graph, node) -> int:
    label = graph.nodes[node]['label']
    node_ranks = {}
    for n in graph.nodes():
        if graph.nodes[n]['label'] == label:
            rank = average_weight_of_adjacent_nodes(graph, n)
            node_ranks[n] = rank

    ranked_nodes = sorted(node_ranks, key=node_ranks.get, reverse=True)
    node_rank = ranked_nodes.index(node) + 1
    # print(ranked_nodes)
    return int(node_rank)

def get_top_ranked_node_each_group(graph):
    # Calculate the rank of all nodes once
    node_ranks = {node: average_weight_of_adjacent_nodes(graph, node) for node in graph.nodes()}

    # Get all unique labels in the graph
    labels = set(data['label'] for node, data in graph.nodes(data=True))

    top_ranked_nodes = {}
    for label in labels:
        # Get the nodes with the same label
        label_nodes = [node for node in graph.nodes() if graph.nodes[node]['label'] == label]

        # Get the node with the highest rank
        top_ranked_node = max(label_nodes, key=node_ranks.get)
        top_ranked_nodes[label] = top_ranked_node
    # print(top_ranked_nodes)
    return list(top_ranked_nodes.values())


def average_weight_of_adjacent_nodes(graph, node) -> float:
    """
    Calculate the average weight of adjacent nodes with the same label as the given node in a graph.

    Parameters:
    graph (networkx.Graph): The graph containing the nodes and edges.
    node: The node for which to calculate the average weight of adjacent nodes.

    Returns:
    float: The average weight of adjacent nodes with the same label as the given node.
    """

    label = graph.nodes[node]['label']
    adjacent_nodes = graph.neighbors(node)
    total_weight = 0
    count = 0

    for adjacent_node in adjacent_nodes:
        if graph.nodes[adjacent_node]['label'] == label:
            total_weight += graph[node][adjacent_node]['weight']
            count += 1

    if count == 0:
        return 0
    else:
        return round(total_weight / count, 4)


def sum_edge_weights(graph) -> float:
    total_weight = 0.0

    for _, _, data in graph.edges(data=True):
        if 'weight' in data:
            total_weight += data['weight']

    return round(total_weight, 4)


def randomAlgo(network):
    """
    Randomly selects a node from each unique label in the given network.

    Parameters:
    - network (networkx.Graph): The network graph containing nodes with 'label' attribute.

    Returns:
    - list: A list of selected nodes, one for each unique venue in the network.
    """

    # Assuming 'label' attribute is a single value, not a list
    label_dict = nx.get_node_attributes(network, 'label')
    
    # Get unique venues
    unique_labels = set(label_dict.values())

    selected_nodes = {}
    for label in unique_labels:
        # get nodes with this venue
        nodes = [n for n, v in label_dict.items() if v == label]
        
        # randomly select a node
        if nodes:  # check if nodes list is not empty
            selected_node = random.choice(nodes)
            selected_nodes[label] = selected_node

    return list(selected_nodes.values())


def randomMonteCarlo(graph, num_iter):
    """
        We calculate the weight of the adjacent nodes the selected node is connected to in its team.
        We then calculate the weight of the subgraph formed by the selected nodes.(iterteam communication)
    """
    total_weight = 0

    for _ in range(num_iter):
        local_weight = 0
        for node in randomAlgo(graph):
            local_weight += (average_weight_of_adjacent_nodes(graph, node))
        
        total_weight = total_weight + local_weight + sum_edge_weights(graph.subgraph(randomAlgo(graph)))

    avg_weight = round(total_weight / num_iter, 2)
    # print(f"Using Random : {avg_weight}")
    return avg_weight


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
    Returns the inverse closeness centrality (the lower the better)
    """
    return (1/(nx.closeness_centrality(G, source, distance='weight')))/mt.sqrt(G.number_of_nodes())

def crossteamEff(G, P, source, target:list):
    """
    Returns the average shortest path between the node and the selected nodes

    """
    cross_net = subgraph_project(G, P, source)
    label_list = list(set([node['label'] for _, node in cross_net.nodes(data=True)]))
    crossEff = 0.0
    for t in target:
        if G.nodes[t]['label'] in label_list:
            if nx.has_path(cross_net, source, t):
                temp = nx.dijkstra_path_length(cross_net, source, t, weight="weight")/mt.sqrt(cross_net.number_of_nodes())
                
            else:
                temp = 10000
            
            crossEff = crossEff + temp
        
        else:
            crossEff = 10000

    return crossEff/len(target)


def get_top_node_from_each_group(graph_G, graph_P):
    top_nodes = []
    for label in graph_P.nodes:
        team_members = [n for n, attr in graph_G.nodes(
            data=True) if attr['label'] == label]
        team_graph = graph_G.subgraph(team_members)
        centrality = nx.closeness_centrality(team_graph)
        ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
        # print(ranked_nodes) #prints out the centrality of the nodes 
        top_node = ranked_nodes[0] if ranked_nodes else None
        top_nodes.append(top_node)

    return top_nodes

def Greedy(graph_G, graph_P, seed_node):
    if graph_G is None or graph_P is None:
        RuntimeError("One or Both of the graphs is None! ")

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    subset = set()
    subset.add(seed_node)
    labels = []
    labels.append(graph_G.nodes[seed_node]['label'])
    communication_efficiency = 0.0

    while len(subset) < len(graph_P.nodes):
        best_node = None
        min_eff = float('inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['label'] not in labels:
                temp_subset.add(node)

                # In-team efficiency
                local_net = subgraph_by_same_label(graph_G, node)
                in_team_eff = inteamEff(local_net, node)

                #cross-team efficiency
                cross_team_eff = crossteamEff(graph_G, graph_P, node, list(temp_subset))

                total_eff = in_team_eff + cross_team_eff

                if total_eff < min_eff:
                    min_eff = total_eff
                    best_node = node
                    print(f"node: {best_node}, inteam score: {in_team_eff}, cross-team score: {cross_team_eff}")

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += min_eff

    # return subset, sum_edge_weights(graph_G.subgraph(subset))
    return subset, round(communication_efficiency, 4)

