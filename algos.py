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
                old_inf = sum(leader_eff(graph_G, graph_P,
                                metric_fn, node, beta) for node in subset)
                
                marginal_inf = total_inf - old_inf # Calculate the marginal influence of the node

                # Update the best node if the current node maximizes the total edge weight
                if marginal_inf > max_inf:
                    max_inf = marginal_inf
                    best_node = node

        # Add the best node to the subset
        if best_node is None:
            break
        subset.add(best_node)
        labels.add(graph_G.nodes[best_node]['label'])
        communication_efficiency += marginal_inf

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


def intra_team_rank(graph_G, metric_fn, node) -> int:
    # Subgraph of nodes with the same label
    team_graph = ps.subgraph_by_same_label(graph_G, node)
    centrality = metric_fn(team_graph)  # Centrality measure on the team graph
    ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
    node_rank = ranked_nodes.index(node) + 1
    return int(node_rank)


def inter_team_rank(graph_G, graph_P, metric_fn, node) -> int:
    # Subgraph of nodes with the same label
    team_graph = ps.subgraph_by_label(graph_G, graph_P, node)
    centrality = metric_fn(team_graph)  # Centrality measure on the team graph
    ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
    node_rank = ranked_nodes.index(node) + 1
    return int(node_rank)


def randomMonteCarlo(graph_G, graph_P, metric_fn, num_iter):
    """
        We calculate the weight of the adjacent nodes the selected node is connected to in its team.
        We then calculate the weight of the subgraph formed by the selected nodes.(iterteam communication)
    """
    total_weight = 0

    for _ in range(num_iter):

        total_weight += sum(leader_eff(graph_G, graph_P, metric_fn,
                            node, beta=None) for node in ps.randomAlgo(graph_G))

    return round(total_weight / num_iter, 2)


def get_top_node_from_each_group(graph_G, graph_P, metric_fn):
    top_nodes = []
    for label in graph_P.nodes:
        team_members = [n for n, attr in graph_G.nodes(
            data=True) if attr['label'] == label]
        team_graph = graph_G.subgraph(team_members)
        centrality = metric_fn(team_graph)
        ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
        print(centrality)
        top_node = ranked_nodes[0] if ranked_nodes else None
        top_nodes.append(top_node)
    return top_nodes


def inteam_influence_only(graph_G, graph_P, metric_fn):
    """
    Calculate the influence of the seed node on its team members only
    """
    network = graph_G.subgraph(get_top_node_from_each_group(graph_G, graph_P, metric_fn))

    print("Intra-team ranking")
    for node in network.nodes():
        print(f"Team :{graph_G.copy().nodes[node]['label']}, Node: {node}, Rank: {intra_team_rank(graph_G, metric_fn, node)}")

    print("\n")

    print("Inter-team ranking")
    for node in network.nodes():
        print(f"Team :{graph_G.copy().nodes[node]['label']}, Node: {node}, Rank: {inter_team_rank(graph_G, graph_P, metric_fn, node)}")
    
    return sum(leader_eff(graph_G, graph_P, metric_fn, user, beta=None) for user in network)
