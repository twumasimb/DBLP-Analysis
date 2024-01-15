import networkx as nx
import random
import pickle
import plotly.graph_objects as go


list_1 = [('software', 'design'), ('software', 'cyber'),
          ('software', 'sales'), ('software', 'advert')]  # Star
list_2 = [('software', 'design'), ('software', 'cyber'), ('software', 'sales'), ('software',
                                                                                 'advert'), ('sales', 'advert'), ('cyber', 'sales'), ('design', 'advert')]  # 3 triads
list_3 = [('software', 'design'), ('software', 'cyber'),
          ('cyber', 'sales'), ('advert', 'sales')]  # Chain
list_4 = [('software', 'design'), ('software', 'cyber'), ('cyber',
                                                          'sales'), ('design', 'sales'), ('design', 'cyber')]  # 1 triad
list_5 = [('software', 'design'), ('software', 'cyber'), ('software', 'sales'), ('software', 'advert'), ('sales', 'advert'),
          ('cyber', 'sales'), ('design', 'advert'), ('design', 'cyber'), ('advert', 'cyber'), ('design', 'sales')]  # Fully connected


# Team_1 = utils.createRandomTeam(network, 'design', 10)
# Team_2 = utils.createRandomTeam(network, 'cyber', 10)
# Team_3 = utils.createRandomTeam(network, 'software', 10)
# Team_4 = utils.createRandomTeam(network, 'advert', 10)
# Team_5 = utils.createRandomTeam(network, 'sales', 10)


def createNetwork(file):
    return nx.read_edgelist(file, create_using=nx.Graph(), nodetype=int)


def networkPreprocessing(G, labels):
    """
        Define some labels and randomly assign them to the nodes in the network.
    """
    # Assign labels to each node
    for node in G.nodes():
        label = random.choice(labels)
        G.nodes[node]["label"] = label

    # Add random weights to the edges
    for u, v in G.edges():
        weight = random.randint(1, 100)
        G[u][v]['weight'] = weight

    # Add random weights to unconnected edges
    for u in G.nodes():
        for v in G.nodes():
            if u != v and not G.has_edge(u, v):
                weight = random.randint(1, 100)
                G.add_edge(u, v, weight=weight)

    return G


def removeEdges(G):
    # Remove edges between nodes with the same label
    for u in G.nodes():
        for v in G.nodes():
            if u != v and G.nodes[u]['label'] == G.nodes[v]['label'] and G.has_edge(u, v):
                G.remove_edge(u, v)
    return G


def remove_edges_based_on_project_network(expert_network, project_network):
    edges_to_remove = []

    for edge in expert_network.edges():
        node1_label = expert_network.nodes[edge[0]]['label']
        node2_label = expert_network.nodes[edge[1]]['label']

        if not project_network.has_edge(node1_label, node2_label):
            edges_to_remove.append(edge)

    expert_network.remove_edges_from(edges_to_remove)

    return expert_network


def createProjectNetwork(list):
    project = nx.Graph()
    project.add_edges_from(list)
    return project


def saveNetwork(G, file):
    pickle.dump(G, open(f'{file}.pkl', 'wb'))


def createRandomTeam(G, label_to_filter, team_size):
    # Initialize a subgraph
    subgraph = nx.Graph()

    nodes_with_label = [node for node, data in G.nodes(
        data=True) if 'label' in data and data['label'] == label_to_filter]

    # Randomly select team_size nodes from the list
    if len(nodes_with_label) >= team_size:
        team_nodes = random.sample(nodes_with_label, team_size)
    else:
        print("Not enough nodes with the specified label to form a team.")
        team_nodes = []

    # Add the selected team nodes to the team subgraph
    for node in team_nodes:
        subgraph.add_node(node, label=label_to_filter)

    return subgraph


def getSubgraph(G, teams):
    """
        This is a subgraph of all the people across the teams. 
    """
    all_nodes = []
    for team in teams:
        team_nodes = [node for node in team.nodes]
        all_nodes.extend(team_nodes)

    return G.subgraph(all_nodes)


def minimize_total_edge_weight_subset(graph_G, graph_P):
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Start with a random node from G
    subset = {random.choice(list(graph_G.nodes))}

    while len(subset) < len(graph_P.nodes):
        best_node = None
        min_total_edge_weight = float('inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            temp_subset.add(node)

            # Calculate the total edge weight in the subgraph
            total_edge_weight = sum_edge_weights(graph_G.subgraph(temp_subset))

            # Update the best node if the current node minimizes the total edge weight
            if total_edge_weight < min_total_edge_weight:
                min_total_edge_weight = total_edge_weight
                best_node = node

        # Add the best node to the subset
        subset.add(best_node)

    return graph_G.subgraph(subset)


def sum_edge_weights(graph):
    total_weight = 0

    for _, _, data in graph.edges(data=True):
        if 'weight' in data:
            total_weight += data['weight']

    return total_weight


def minimize_max_edge_subset(graph_G, graph_P):
    print("This function is running!")
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Initialize the subset with nodes from P
    subset = set()

    # Add random node
    key = random.choice(list(graph_G.nodes))
    subset.add(key)  # graph_G.nodes[51506]
    print(f"Random node is: {key}")
    # print(f"The length of subset {len(subset)} vs length of project {len(graph_P.nodes)}")

    # iteratively search for the best nodes
    while len(subset) < len(graph_P.nodes):
        print("Entered While loop \n")
        best_node = None
        min_max_edge_weight = float('inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            temp_subset.add(node)

            # print(f"Current subset: {temp_subset}")

            # Create a subgraph with the current subset
            subgraph = graph_G.subgraph(temp_subset)

            # Calculate the maximum edge weight in the subgraph
            try:
                max_edge_weight = max(subgraph.edges(
                    data=True), key=lambda edge: edge[2]['weight'])[2]['weight']
            except:
                print("Error getting max_edge_weight, weight set to inf")
                max_edge_weight = float('inf')

            # Update the best node if the current node minimizes the maximum edge weight
            if max_edge_weight < min_max_edge_weight:
                min_max_edge_weight = max_edge_weight
                best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        print("Current Subset is: ", subset)
    return graph_G.subgraph(subset)


def minimize_max_edge_subset_labels(graph_G, graph_P):
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Initialize the subset with nodes from P
    subset = set()

    # Add random node
    key = random.choice(list(graph_G.nodes))
    subset.add(key)
    labels = []
    labels.append(graph_G.nodes[key]['label'])

    # iteratively search for the best nodes
    while len(subset) < len(graph_P.nodes):
        best_node = None
        min_max_edge_weight = float('inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['label'] in labels:
                pass
            else:
                temp_subset.add(node)

                # Create a subgraph with the current subset
                subgraph = graph_G.subgraph(temp_subset)

                # Calculate the maximum edge weight in the subgraph
                try:
                    max_edge_weight = max(subgraph.edges(
                        data=True), key=lambda edge: edge[2]['weight'])[2]['weight']
                except:
                    # print("Error getting max_edge_weight, weight set to inf")
                    max_edge_weight = float('inf')

                # Update the best node if the current node minimizes the maximum edge weight
                if max_edge_weight < min_max_edge_weight:
                    min_max_edge_weight = max_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])

    return graph_G.subgraph(subset)


def plotNetwork(G):

    pos = nx.spring_layout(G)  # Position the nodes using a layout algorithm

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += tuple([x0, x1, None])
        edge_trace["y"] += tuple([y0, y1, None])

    # Create the graph figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
        ),
    )

    # Display the interactive graph
    fig.show()

# def getNextBest(node, network, project_network):
#     """
#         This subroutine returns the best nodes in the adjacent node
#     """
#     node_label = network.nodes[node]['label']
#     neighbors = list(project_network.neighbors(node_label))
#     list_of_selected_nodes = []
#     try:
#         for label in neighbors:
#             labeled_neighbors = [n for n in network.neighbors(
#                 node) if network.nodes[n]['label'] == label]
#             for u in labeled_neighbors:
#                 max_weight = 0
#                 selected_node = None
#                 if network[node][u]['weight'] > max_weight:
#                     max_weight = network[node][u]['weight']
#                     selected_node = u
#             list_of_selected_nodes.append(selected_node)
#             # print("..")
#         # visited_array.append(node)
#     except:
#         print("Node has no neighbors")

#     return list_of_selected_nodes


# def getCoordinator(network, project_network):

#     key = random.choice(list(network.nodes))
#     visited_array = []
#     first_list = getNextBest(key, network, project_network)
#     coordinators = []
#     coordinators.extend(first_list)
#     labels = set([network.nodes[n]['label'] for n in first_list])
#     visited_array.append(key)
#     while len(coordinators) < len(project_network):
#         for i in coordinators:
#             # and network.nodes[i]['label'] not in list(labels)
#             if i not in visited_array:
#                 key = i
#                 new_list = getNextBest(key, network, project_network)
#                 coordinators.extend(new_list)
#                 visited_array.append(key)
#                 print(f"new key is : {key}")
#                 break
#         new_list = getNextBest(key, network, project_network)
#         print(f"new list : {new_list}")
#         labels = labels.union(set([network.nodes[n]['label']
#                               for n in new_list]))  # Update
#         print(f"Updated labels : {labels}")
#         # coordinators.extend(new_list)
#         print(f"{len(coordinators)} coordinators added")

#     return coordinators


def minimize_max_edge_subset_connected(graph_G, graph_P):
    """
        This algorithm checks the connectivity of the graph before it runs the maximum weight.
    """
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Initialize the subset with a random node from G
    random_node = random.choice(list(graph_G.nodes))
    subset = {random_node}

    while len(subset) < len(graph_P.nodes):
        best_node = None
        min_max_edge_weight = float('inf')

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            temp_subset.add(node)

            # Create a subgraph with the current subset
            subgraph = graph_G.subgraph(temp_subset)

            # Check if the subgraph is connected
            if nx.is_connected(subgraph):
                # Calculate the maximum edge weight in the subgraph
                max_edge_weight = max(subgraph.edges(
                    data=True), key=lambda edge: edge[2]['weight'])[2]['weight']

                # Update the best node if the current node minimizes the maximum edge weight
                if max_edge_weight < min_max_edge_weight:
                    min_max_edge_weight = max_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)

    return graph_G.subgraph(subset)


def compute_influence(graph):
    if graph is None:
        print("Error: Graph is None.")
        return None

    for node in graph.nodes:
        total_weight = sum(edge['weight']
                           for _, _, edge in graph.edges(node, data=True))
        graph.nodes[node]['influence'] = total_weight

    # Scale scores to 100
    max_influence = max(graph.nodes[node]['influence'] for node in graph.nodes)
    scale_factor = 100 / max_influence

    for node in graph.nodes:
        graph.nodes[node]['influence'] *= scale_factor

    return graph


def find_highest_influence(graph):
    if graph is None:
        print("Error: Graph is None.")
        return None

    max_influence_node = max(graph.nodes, key=lambda node: (
        graph.nodes[node]['influence'], node))
    return max_influence_node


def maximize_total_edge_weight_subset_labels(graph_G, graph_P):
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Start with a random node from G
    key = random.choice(list(graph_G.nodes))
    subset = set()
    subset.add(key)
    labels = []
    labels.append(graph_G.nodes[key]['label'])

    while len(subset) < len(graph_P.nodes):
        best_node = None
        min_total_edge_weight = 0.0

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['label'] not in labels:
                temp_subset.add(node)

                # Calculate the total edge weight in the subgraph
                total_edge_weight = sum_edge_weights(
                    graph_G.subgraph(temp_subset))

                # Update the best node if the current node minimizes the total edge weight
                if total_edge_weight > min_total_edge_weight:
                    min_total_edge_weight = total_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])

    return graph_G.subgraph(subset)


def sum_edge_weights(graph):
    total_weight = 0

    for _, _, data in graph.edges(data=True):
        if 'weight' in data:
            total_weight += data['weight']

    return total_weight


def compute_influence_within_groups(graph):
    """
        This takes in the entire network and returns the most influential nodes in each team as a list
    """
    if graph is None:
        print("Error: Graph is None.")
        return None

    max_influence_nodes = {}  # To store the node with the highest influence for each label

    for label in set(nx.get_node_attributes(graph, 'label').values()):
        group_nodes = [node for node in graph.nodes if graph.nodes[node]['label'] == label]

        max_total_weight = max(sum(graph[node1][node2]['weight'] for node1, node2 in graph.edges(group_nodes)) for group_nodes in group_nodes)
        scale_factor = 100 / max_total_weight if max_total_weight != 0 else 0

        max_node = None
        max_influence = 0

        for node in group_nodes:
            total_weight = sum(graph[node1][node2]['weight'] for node1, node2 in graph.edges(node))
            influence_score = round(total_weight * scale_factor, 2)
            graph.nodes[node]['influence'] = influence_score

            # Check if the current node has higher influence
            if influence_score > max_influence:
                max_influence = influence_score
                max_node = node

        max_influence_nodes[label] = {'node': max_node, 'influence': max_influence}

    return max_influence_nodes


# Top nodes only
def top_influential_nodes(network):
    top_nodes = [node for node in network.nodes if network.nodes[node]['influence'] == 100]
    return network.subgraph(top_nodes)


def monte_carlo(graph_G, graph_P, num_iter):
    comm_eff = 0
    selected_nodes = {}
    for i in range(num_iter):
        best = maximize_total_edge_weight_subset_labels(graph_G, graph_P)
        selected_nodes.union(set(best.nodes))
        eff = sum_edge_weights(best)
        comm_eff = comm_eff + eff
    avg_comm_eff = comm_eff/num_iter

    return selected_nodes, avg_comm_eff