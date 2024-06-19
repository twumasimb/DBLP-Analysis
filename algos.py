import preprocessing as ps
import networkx as nx

def Greedy(graph_G, graph_P, seed_node, metric_fn, beta=None):
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
        min_total_edge_weight = float('-inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['label'] not in labels:
                
                temp_subset.add(node)
                temp_list_all_connected = [i for i in temp_subset for graph_G.nodes[i]['label'] in labels]
                temp_net = graph_G.subgraph(temp_subset)

                # Calculate the total edge weight in the subgraph
                if beta is None:
                    total_edge_weight = metric_fn(ps.subgraph_by_same_label(graph_G, node), node) + \
                    metric_fn(ps.subgraph_by_label(graph_G, graph_P, node), node)
                else:
                    total_edge_weight = beta*metric_fn(ps.subgraph_by_same_label(graph_G, node), node) + \
                    (1-beta)*metric_fn(ps.subgraph_by_label(graph_G, graph_P, node), node)

                # Update the best node if the current node maximizes the total edge weight
                if total_edge_weight > min_total_edge_weight:
                    min_total_edge_weight = total_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += min_total_edge_weight

    return subset, round(communication_efficiency, 4)