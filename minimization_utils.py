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

def inteamEff(G, source):
    """
    Returns the inverse closeness centrality (the lower the better)
    """
    return 1/(nx.closeness_centrality(G, source, distance='weight'))

def crossteamEff(G, source, target):
    """
    Returns the shortest path between two nodes
    """
    return nx.shortest_simple_paths(G, source, target, weight="weight")

def Greedy(graph_G, graph_P, seed_node, beta=None):
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
        max_total_edge_weight = float('inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['label'] not in labels:
                temp_subset.add(node)

                # Calculate the total edge weight in the subgraph
                if beta is None:
                    total_edge_weight = sum_edge_weights(
                        graph_G.subgraph(temp_subset)) + (average_weight_of_adjacent_nodes(graph_G, node))
                else:
                    total_edge_weight = beta*sum_edge_weights(
                        graph_G.subgraph(temp_subset)) + (1-beta)*(average_weight_of_adjacent_nodes(graph_G, node))

                # Update the best node if the current node maximizes the total edge weight
                if total_edge_weight < max_total_edge_weight:
                    max_total_edge_weight = total_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += max_total_edge_weight

    # print(f"Coordinators Communication Efficiency: {sum_edge_weights(graph_G.subgraph(subset))}")
    # print(f"Coordinators : {subset}")

    # return subset, sum_edge_weights(graph_G.subgraph(subset))
    return subset, round(communication_efficiency, 4)

