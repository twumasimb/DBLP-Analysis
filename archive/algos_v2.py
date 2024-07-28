import preprocessing as ps
import networkx as nx

def subgraph_by_label(G, P, node_g):
    """
    Generate a subgraph of G based on nodes connected to node_g's label in P.
    Includes node_g itself regardless of its label's connection status.

    Parameters:
    - G: NetworkX graph
    - P: NetworkX graph representing connections between labels
    - node_g: Node in G from which to generate the subgraph

    Returns:
    - subgraph: NetworkX subgraph of G
    """
    label_g = G.nodes[node_g]['label']
    
    # Find all connected labels in P
    connected_labels = set()
    for u, v in P.edges():
        if v == label_g:
            connected_labels.add(u)
        elif u == label_g:
            connected_labels.add(v)
    
    # Include node_g in the subgraph even if its label has no connections
    connected_labels.add(label_g)
    
    # Select all nodes in G with labels in connected_labels
    selected_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] in connected_labels]
    
    # Generate the subgraph of the selected nodes
    subgraph = G.subgraph(selected_nodes).copy()
    
    return subgraph


def subgraph_by_same_label(G, node_g):
    """
    Generate a subgraph of G consisting of nodes with the same label as node_g.

    Parameters:
    - G: NetworkX graph
    - node_g: Node in G

    Returns:
    - subgraph: NetworkX subgraph of G
    """
    label_g = G.nodes[node_g]['label']
    
    # Select all nodes in G with the same label as node_g
    selected_nodes = [n for n, attr in G.nodes(data=True) if attr['label'] == label_g]
    
    # Generate the subgraph of the selected nodes
    subgraph = G.subgraph(selected_nodes).copy()
    
    return subgraph


def intra_team_rank(graph_G, metric_fn, node) -> int:
    """
    Rank a node within its team based on a centrality metric.
    """
    team_graph = ps.subgraph_by_same_label(graph_G, node)
    
    if len(team_graph) == 0:
        raise ValueError(f"No nodes found with the same label as node {node}.")
    
    centrality = metric_fn(team_graph)
    ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
    
    try:
        node_rank = ranked_nodes.index(node) + 1
    except ValueError:
        raise ValueError(f"Node {node} not found in the centrality rankings of its team.")
    
    return int(node_rank)

def inter_team_rank(graph_G, graph_P, metric_fn, node) -> int:
    """
    Rank a node across different teams based on a centrality metric.
    """
    team_graph = ps.subgraph_by_label(graph_G, graph_P, node)
    
    if len(team_graph) == 0:
        raise ValueError(f"No nodes found in the subgraph defined by labels in graph_P.")
    
    centrality = metric_fn(team_graph)
    ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
    
    try:
        node_rank = ranked_nodes.index(node) + 1
    except ValueError:
        raise ValueError(f"Node {node} not found in the centrality rankings across teams.")
    
    return int(node_rank)


def leader_eff(graph_G, graph_P, metric_fn, node, beta=None):
    interteam_network = ps.subgraph_by_same_label(graph_G, node)
    intrateam_network = ps.subgraph_by_label(graph_G, graph_P, node)

    iter_team = metric_fn(interteam_network, node)
    intra_team = metric_fn(intrateam_network, node)

    if beta is None:
        return iter_team + intra_team
    else:
        return (beta) * iter_team + ((1 - beta)) * intra_team


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
    network = graph_G.subgraph(
        get_top_node_from_each_group(graph_G, graph_P, metric_fn))

    print("Intra-team ranking")
    for node in network.nodes():
        print(
            f"Team :{graph_G.copy().nodes[node]['label']}, Node: {node}, Rank: {intra_team_rank(graph_G, metric_fn, node)}")

    print("\n")

    print("Inter-team ranking")
    for node in network.nodes():
        print(
            f"Team :{graph_G.copy().nodes[node]['label']}, Node: {node}, Rank: {inter_team_rank(graph_G, graph_P, metric_fn, node)}")

    return round(sum(leader_eff(graph_G, graph_P, metric_fn, user, beta=None) for user in network), 2)



def comm_efficiency(graph_G, graph_P, metric_fn, seed_node, lead_set, beta=None):
    """
    Calculate the communication efficiency of the seed node
    """
    # Assuming these functions are defined elsewhere to get relevant subgraphs
    interteam_network = ps.subgraph_by_same_label(graph_G, seed_node)
    intrateam_network = ps.subgraph_by_label(graph_G, graph_P, seed_node)

    iter_team = metric_fn(interteam_network, seed_node)
    leader_team = 0.0
    for lead in lead_set:
        try:
            shortest_path = nx.shortest_path(intrateam_network, source=seed_node, target=lead, weight='weight')
            leader_team += nx.path_weight(intrateam_network, shortest_path, weight='weight')
        except nx.NodeNotFound:
            continue

    return (beta * iter_team) + ((1 - beta) * leader_team) if beta is not None else iter_team + leader_team

def Greedy(graph_G, graph_P, seed_node, metric_fn, beta=None):
    if graph_G is None or graph_P is None:
        raise RuntimeError("One or Both of the graphs is None!")

    if len(graph_P.nodes) > len(graph_G.nodes):
        raise ValueError("Number of nodes in P is greater than the number of nodes in G.")

    subset = {seed_node}
    labels = {graph_G.nodes[seed_node]['label']}
    communication_efficiency = comm_efficiency(graph_G, graph_P, metric_fn, seed_node, subset, beta)

    while len(subset) < len(graph_P.nodes):
        best_node = None
        max_inf = float('-inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            node_label = graph_G.nodes[node]['label']
            if node_label not in labels:
                temp_subset = subset | {node}
                total_inf = comm_efficiency(graph_G, graph_P, metric_fn, seed_node, temp_subset, beta)
                old_inf = communication_efficiency
                marginal_inf = total_inf - old_inf

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

