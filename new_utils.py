import pickle
import random
import networkx as nx
random.seed(42)
import math as mt
from typing import Tuple, List
import itertools
from collections import defaultdict

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

def find_best_set_of_leaders(G, top_nodes, teams=['DM', 'T', 'DB', 'AI']):
    """
    Finds the best set of leaders based on the given graph, top nodes, and teams.

    Parameters:
    - G (graph): The graph representing the network.
    - top_nodes (list): The list of top nodes in the network.
    - teams (list): The list of teams.

    Returns:
    - best_set (set): The best set of leaders.
    - max_eff (float): The efficiency of the best set of leaders.
    """

    max_eff = float('-inf')
    best_set = set()

    for node in top_nodes:
        subset, commEff = Greedy(G, teams, node)
        if commEff > max_eff:
            max_eff = commEff
            best_set = subset

    return best_set, max_eff

def analyze_network_by_labels(G, labels=['T', 'DB', 'DM', 'AI']):
    """
    Analyze the network by labels, calculate closeness centrality,
    and identify top nodes for each label.
    """
    results = defaultdict(dict)
    
    for label in labels:
        # Get nodes with the current label and create a subgraph
        nodes_with_label = [node for node, data in G.nodes(data=True) if data.get('label') == label]
        subgraph = G.subgraph(nodes_with_label)
        
        # Calculate closeness centrality
        closeness_scores = nx.closeness_centrality(subgraph)
        
        # Set closeness centrality as node attribute
        nx.set_node_attributes(subgraph, closeness_scores, 'closeness_centrality')
        
        # Find node(s) with max closeness centrality
        max_score = max(closeness_scores.values())
        top_nodes = [node for node, score in closeness_scores.items() if score == max_score]
        
        # Store results
        results[label] = {
            'subgraph': subgraph,
            'closeness_scores': closeness_scores,
            'max_node': max(closeness_scores, key=closeness_scores.get),
            'top_nodes': top_nodes
        }
    
    # Print results
    # for label, data in results.items():
    #     print(f"Label {label}:")
    #     print(f"  Node with highest closeness centrality: {data['max_node']}")
    #     print(f"  Top nodes: {data['top_nodes']}")
    
    # Combine all top nodes
    all_top_nodes = [node for data in results.values() for node in data['top_nodes']]
    print(f"\nTotal number of top nodes across all labels: {len(all_top_nodes)}")
    
    return results, all_top_nodes

def get_connected_subgraph(G, size):
    if size > G.number_of_nodes():
        raise ValueError("Requested subgraph size is larger than the graph.")
    
    start_node = random.choice(list(G.nodes))
    visited = set([start_node])
    queue = [start_node]

    while len(visited) < size and queue:
        node = queue.pop(0)
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
            if len(visited) == size:
                break

    return list(visited)

def select_nodes_by_label(G, labels=['T', 'DB', 'DM', 'AI'], size_per_label=10):
    random.seed(12)
    results = defaultdict(dict)
    selected_nodes = []

    for label in labels:
        nodes_with_label = [node for node, data in G.nodes(data=True) if data.get('label') == label]
        subgraph = G.subgraph(nodes_with_label)
        
        try:
            selected = get_connected_subgraph(subgraph, size_per_label)
        except ValueError as e:
            print(f"Warning for label {label}: {str(e)}")
            selected = list(subgraph.nodes)  # Take all nodes if not enough
        
        results[label] = {
            'subgraph': subgraph,
            'selected_nodes': selected
        }
        selected_nodes.extend(selected)

    return results, selected_nodes


def simple_random_selection(network):
    """
    Randomly selects a node from each unique label in the network.

    Parameters:
    - network (networkx.Graph): The network graph with 'label' attributes.

    Returns:
    - list: Selected nodes, one for each unique label.
    """
    label_groups = {}

    # Group nodes by label
    for node, data in network.nodes(data=True):
        label = data.get('label')
        if label:
            label_groups.setdefault(label, []).append(node)

    # Select one random node from each label group
    return [random.choice(nodes) for nodes in label_groups.values()]

def inteam_eff(G, source):
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


def average_distance(G, nodes):
    """
    Compute the average distance between a list of nodes in a network.

    Parameters:
    G (networkx.Graph): The network graph
    nodes (list): List of nodes to compute the average distance between

    Returns:
    float: The average distance between the nodes
    """
    if len(nodes) < 2:
        return 0  # No distance to compute if there's only one node or less

    total_distance = 0
    pair_count = 0

    # Generate all unique pairs of nodes
    for node1, node2 in itertools.combinations(nodes, 2):
        try:
            # Compute the shortest path length between the pair
            distance = nx.shortest_path_length(G, node1, node2)
            sumDistance = nx.dijkstra_path_length(G, node1, node2, weight='weight')
            total_distance += (distance/sumDistance)
            pair_count += 1
        except nx.NetworkXNoPath:
            # If there's no path between the nodes, we can either ignore it or handle it as needed
            # Here, we'll just print a warning
            print(f"Warning: No path between nodes {node1} and {node2}")

    # Compute and return the average distance
    if pair_count > 0:
        return total_distance / pair_count
    else:
        return float('inf')  # or any other value to indicate no valid distances were found


def comm_eff(G, leaders:list, alpha=1):
    inTeam = 0.0
    crossTeam = 0.0
    for node in leaders:
        inTeam += G.nodes[node]['closeness_centrality']
    
    crossTeam += average_distance(G, leaders)

    #     # print(f"{leaders[0]} --> {leader}: length: {len_shortest_path}, distance: {sumDistance}, closeness: {closeness}")
    return round(alpha*(crossTeam) + inTeam, 4)


def Greedy(graph_G, teams:list, seed_node):
    if graph_G is None:
        RuntimeError("One or Both of the graphs is None! ")

    subset = set()
    subset.add(seed_node)
    labels = []
    labels.append(graph_G.nodes[seed_node]['label'])

    while len(subset) < len(teams):
        best_node = None
        max_eff = float('-inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['label'] not in labels:
                temp_subset.add(node)

                total_node_eff = comm_eff(graph_G, list(temp_subset))

                if total_node_eff > max_eff:
                    max_eff = total_node_eff
                    best_node = node
                    # print(f"node: {best_node}, inteam score: {in_team_eff}, cross-team score: {cross_team_eff}")

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
    
    # return subset, sum_edge_weights(graph_G.subgraph(subset))
    return subset, round(comm_eff(graph_G, list(subset)), 4)


def randomAlgo(G, num_iters=1000):
    """
    Calculates the communication efficiency of a graph using the simple random selection method.

    Parameters:
    - G: The graph for which to calculate the communication efficiency.

    Returns:
    - The average communication efficiency over a specified number of iterations.
    """

    simple_random_selection(G)

    total = 0.0
    for _ in range(num_iters):
        total += comm_eff(G, simple_random_selection(G))

    return round(total/num_iters, 4)


def create_unique_label_combinations(G, nodesList):
    """
    Create combinations of nodes where each combination has 4 nodes with unique label attributes.

    Parameters:
    nodes (list): List of node objects. Each node should have a 'label' attribute.

    Returns:
    list: List of combinations, where each combination is a list of 4 nodes.
    """
    nodes = []
    for node in nodesList:
        nodes.append({
            'id': node,
            'label': G.nodes[node]['label']
        })
    # Group nodes by their labels
    label_groups = defaultdict(list)
    for node in nodes:
        label_groups[node['label']].append(node)

    # Ensure we have all 4 required labels
    if len(label_groups) != 4:
        raise ValueError("We need exactly 4 different labels in the node set.")

    # Create all possible combinations
    combinations = list(itertools.product(*label_groups.values()))

    return combinations

# def Greedy(graph: nx.Graph, teams: List[str], seed_node: str) -> Tuple[List[str], float]:
#     """
#     Greedily select nodes from the graph to maximize communication efficiency.

#     Args:
#     graph (nx.Graph): The input graph.
#     teams (List[str]): List of team labels.
#     seed_node (int): The initial node to start the selection.

#     Returns:
#     Tuple[List[int], float]: Selected subset of nodes and the communication efficiency.
#     """
#     if not isinstance(graph, nx.Graph):
#         raise ValueError("Input must be a NetworkX graph.")

#     subset = [seed_node]
#     labels = {graph.nodes[seed_node]['label']}

#     while len(subset) < len(teams):
#         best_node, max_gain = None, float('-inf')

#         for node in set(graph.nodes) - set(subset):
#             if graph.nodes[node]['label'] not in labels:
#                 temp_subset = subset + [node]
                
#                 if len(temp_subset) < 3:
#                     marginal_gain = comm_eff(graph, temp_subset) - inteam_eff(graph, temp_subset[0])
#                 else:
#                     marginal_gain = comm_eff(graph, temp_subset) - comm_eff(graph, temp_subset[:-1])

#                 if marginal_gain > max_gain:
#                     max_gain = marginal_gain
#                     best_node = node

#         if best_node is None:
#             break  # No more valid nodes to add

#         subset.append(best_node)
#         labels.add(graph.nodes[best_node]['label'])

#     return subset, round(comm_eff(graph, subset), 4)

