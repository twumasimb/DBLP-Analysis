import networkx as nx
import itertools
import pickle
from collections import Counter

def select_k_nodes_per_team(G, k):
    """
    Select k nodes that minimize the shortest distance between themselves.
    
    Parameters:
    G (networkx.Graph): The input graph
    k (int): Number of nodes to select
    
    Returns:
    list: The selected nodes
    float: The sum of shortest distances between selected nodes
    """
    if k > G.number_of_nodes():
        raise ValueError("k cannot be larger than the number of nodes in the graph")
    
    # Start with the node that has the highest closeness centrality
    centrality = nx.closeness_centrality(G, distance='weight')
    selected_nodes = [max(centrality, key=centrality.get)]
    
    while len(selected_nodes) < k:
        best_node = None
        best_distance_sum = float('inf')
        
        for node in G.nodes():
            if node not in selected_nodes:
                # Calculate the sum of shortest distances to already selected nodes
                distance_sum = sum(nx.shortest_path_length(G, node, s, weight='weight') for s in selected_nodes)
                
                if distance_sum < best_distance_sum:
                    best_node = node
                    best_distance_sum = distance_sum
        
        selected_nodes.append(best_node)
    
    # Calculate the final sum of shortest distances between all selected nodes
    final_distance_sum = sum(nx.shortest_path_length(G, u, v, weight='weight') 
                             for u, v in itertools.combinations(selected_nodes, 2))
    
    return selected_nodes, final_distance_sum

if __name__ == "__main__":

    # Import the main network
    print("Loading main network")
    network = pickle.load(open('./final_networks/main_network.pkl', 'rb'))
    G = network.copy()

    print("Network Attributes")
    print("------------------")
    print(f"Num of Nodes: {G.number_of_nodes()}")
    print(f"Num of Edges: {G.number_of_edges()}")

    label_counts = nx.get_node_attributes(G, 'label')
    label_counts = dict(Counter(label_counts.values()))
    print(label_counts)
    print("\n")

    total_selected_nodes = []
    labels = ['DB', 'T', 'AI', 'DM']
    sizes = [5, 10, 15, 20, 25] # set the size
    
    for k in sizes:
        print(f"Starting with {k} nodes per team")
        for label in labels:
            nodes_with_label = [node for node, data in G.nodes(data=True) if data.get('label') == label]
            subgraph = G.subgraph(nodes_with_label)
            print(f"Selecting top-{k} node for team {label}")
            selected_nodes, total_distance = select_k_nodes_per_team(subgraph, k)

            total_selected_nodes.extend(selected_nodes)

            print(f"Selected {k} nodes: {selected_nodes}")
            print(f"Total shortest distance between selected nodes: {total_distance}")
            print("\n")

        print(f"Total Selected {4 * k} nodes: {total_selected_nodes}")
        k_network = G.subgraph(total_selected_nodes)
        print("\n")

        with open(f'./final_networks/network_size_{k * 4}.pkl', 'wb') as file:
            pickle.dump(k_network, file)