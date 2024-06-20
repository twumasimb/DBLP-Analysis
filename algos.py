import preprocessing as ps
import networkx as nx


def Greedy(graph_G, graph_P, seed_node, metric_fn, beta=None):
    if graph_G is None or graph_P is None:
        raise RuntimeError("One or Both of the graphs is None!")

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    subset = set()
    subset.add(seed_node)
    labels = set()
    labels.add(graph_G.nodes[seed_node]['label'])
    communication_efficiency = leader_eff(
        graph_G, graph_P, metric_fn, seed_node, beta)  # Initial communication efficiency with seed node

    while len(subset) < len(graph_P.nodes):
        best_node = None
        max_inf = float('-inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            node_label = graph_G.nodes[node]['label']
            if node_label not in labels:
                temp_subset = subset.copy()
                temp_subset.add(node)

                total_inf = sum(leader_eff(graph_G, graph_P,
                                metric_fn, node, beta) for node in temp_subset)

                # Update the best node if the current node maximizes the total edge weight
                if total_inf > max_inf:
                    max_inf = total_inf
                    best_node = node

        # Add the best node to the subset
        if best_node is None:
            break
        subset.add(best_node)
        labels.add(graph_G.nodes[best_node]['label'])
        communication_efficiency += max_inf

    return subset, round(communication_efficiency, 4)


def leader_eff(graph_G, graph_P, metric_fn, node, beta=None):
    interteam_network = ps.subgraph_by_same_label(graph_G, node)
    intrateam_network = ps.subgraph_by_label(graph_G, graph_P, node)

    iter_team = metric_fn(interteam_network, node)
    intra_team = metric_fn(intrateam_network, node)

    if beta is None:
        return iter_team + intra_team
    else:
        return beta * iter_team + (1 - beta) * intra_team
