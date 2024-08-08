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
        closeness_scores = nx.closeness_centrality(subgraph, distance='weight') # Distance ensures that the weights are used.
        
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
            # total_distance += (distance/sumDistance)
            total_distance += (1/sumDistance) # Using the inverse of the sum of weights
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

def comm_eff(G, leaders:list, alpha=1):
    inTeam = 0.0
    crossTeam = 0.0
    for node in leaders:
        inTeam += G.nodes[node]['closeness_centrality']
    
    # crossTeam += average_distance(G, leaders)

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

                total_node_eff = comm_eff(graph_G, list(temp_subset), alpha=100)

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