import pickle
import random
import networkx as nx
random.seed(42)


#### Move these to a utils file later ####
import inspect, re

def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)

def createProjectNetwork(list):
    project = nx.Graph()
    project.add_edges_from(list)
    return project


def parse_and_save_paper_data(input_file, output_file):
    """
    Parses paper data from a text file and saves it as a pickle file.

    Parameters:
    input_file (str): The path to the text file containing the paper data.
    output_file (str): The path to the output file to save the parsed paper data to.
    """
    def parse_paper_data(data):
        paper = {}
        lines = data.split('\n')

        for line in lines:
            if line.startswith("#*"):
                paper['title'] = line[2:].strip()
            elif line.startswith("#@"):
                paper['authors'] = line[2:].strip().split(', ')
            elif line.startswith("#c"):
                paper['venue'] = line[2:].strip()
            elif line.startswith("#t"):
                paper['year'] = int(line[2:].strip())

        return paper

    with open(input_file, 'r', encoding='utf-8') as file:
        # assuming each paper is separated by two newlines
        papers_data = file.read().split('\n\n')

    papers = [parse_paper_data(paper_data) for paper_data in papers_data]

    # Save as pickle
    with open(output_file, 'wb') as pickle_file:
        pickle.dump(papers, pickle_file)


def sample_date(data, start_year: int, end_year: int):
    """
    Filter the input data based on the 'year' key and save the filtered data to a pickle file.

    Args:
        data (list): A list of dictionaries representing the input data.

    Returns:
        None
    """
    sample_data = [
        item for item in data if 'year' in item and start_year <= item['year'] <= end_year]
    return sample_data


def sample_by_venue(data, venues: list):
    """
    Save a subset of data based on specified venues to a pickle file.

    Args:
        data (list): The input data to filter.
        venues (list): A list of venues to filter the data by.
        file_name (str): The name of the output file (without the extension).

    Returns:
        None
    """
    sample_data = [
        item for item in data if 'venue' in item and item['venue'] in venues]
    return sample_data


class Author:
    """
    Represents an author in the DBLP analysis.

    Attributes:
        id (int): The ID of the author.
        name (str): The name of the author.
        coauthors (list): A list of coauthors of the author.
        venues (list): A list of venues where the author has published.
        papers (list): A list of papers authored by the author.
        venue_papers (dict): A dictionary mapping venues to the papers published by the author in each venue.
    """

    def __init__(self):
        self.id = None
        self.name = None
        self.date = None
        self.coauthors = []
        self.venues = []
        self.papers = []
        self.venue_papers = {}
        self.venue_dates = {}

    def author_info(self):
        """
        Returns a dictionary containing information about the author.

        Returns:
            dict: A dictionary with the following keys:
                - 'id': The ID of the author.
                - 'author': The name of the author.
                - 'coauthors': A list of coauthors of the author.
                - 'venues': A list of venues where the author has published.
                - 'papers': A list of papers authored by the author.
                - 'num_of_papers': The number of papers authored by the author.
                - 'venue_papers': A dictionary mapping venues to the papers published by the author in each venue.
        """
        dictionary = {
            'id': self.id,
            'author': self.name,
            'coauthors': self.coauthors,
            'venues': self.venues,
            'papers': self.papers,
            # 'year': self.date,  
            'num_of_papers': len(self.papers),
            'venue_papers': self.venue_papers,
            'venue_dates': self.venue_dates
        }
        return dictionary

    # def get_num_of_papers(self):
    #     """
    #     Returns the number of papers authored by the author.

    #     Returns:
    #         int: The number of papers authored by the author.
    #     """
    #     return len(self.papers)


def get_author_data(list_of_authors, papers):
    """
    Retrieves author data based on a list of authors and a collection of papers.

    Args:
        list_of_authors (list): A list of authors.
        papers (list): A collection of papers.

    Returns:
        list: A list of author data who have published at least 2 papers.

    """
    # Create a dictionary mapping authors to their data
    author_data = {author: {'coauthors': [], 'papers': set(), 'venues': set(), 'venue_papers': {}, 'venue_dates': {}} for author in list_of_authors}

    # Populate the dictionary
    for paper in papers:
        authors_in_paper = set(paper.get('authors', '')[0].split(','))
        for author in authors_in_paper:
            if author in author_data:
                coauthors = set(authors_in_paper) - set([author])
                if coauthors:
                    author_data[author]['coauthors'].extend(list(coauthors))
                author_data[author]['papers'].add(paper.get('title'))
                author_data[author]['venue_dates'][paper.get('venue')] = paper.get('year')  # Store the year of publication
                author_data[author]['venues'].add(paper.get('venue'))
                author_data[author]['venue_papers'][paper.get('venue')] = author_data[author]['venue_papers'].get(paper.get('venue'), 0) + 1

    # Create the Author objects
    authors_data = []
    for id_value, (author, data) in enumerate(author_data.items()):
        if len(data['papers']) >= 5:
            user = Author()
            user.id = id_value
            user.name = author
            user.coauthors = data['coauthors'] # I used this so that I can count the number of time they have coauthored a paper
            user.papers = data['papers'] 
            user.venues = list(data['venues'])
            user.venue_papers = data['venue_papers']
            user.venue_dates = data['venue_dates']
            authors_data.append(user.author_info())

    return authors_data


def get_num_papers(dataset, author_name):
    """
    Preprocesses the dataset by extracting the number of papers for a specific author.

    Args:
        dataset (list): A list of dictionaries representing the dataset.
        author_name (str): The name of the author to filter the dataset.

    Returns:
        int: The number of papers for the specified author.
    """
    for data in dataset:
        if 'author' in data and data['author'] == author_name:
            return data['num_of_papers']

    return 0


def assign_labels(item):
    # Get the venue with the maximum count
    max_venue = max(item['venue_papers'], key=item['venue_papers'].get)

    db_venues = ['VLDB', 'ICDE', 'ICDT', 'EDBT', 'PODS', 'SIGMOD Conference']
    ai_venues = ['ICML', 'ECML', 'COLT', 'UAI']
    theory_venues = ['SODA', 'STOC', 'FOCS', 'STACS']
    dm_venues = ['KDD', 'ICDM', 'PKDD', 'WWW', 'SDM']

    # Assign the label based on the venue category
    if max_venue in db_venues:
        return 'DB'
    elif max_venue in ai_venues:
        return 'AI'
    elif max_venue in theory_venues:
        return 'T'
    elif max_venue in dm_venues:
        return 'DM'
    else:
        return None


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


def get_connected_subgraph(G, size):
    # Check if the requested size is greater than the graph
    if size > G.number_of_nodes():
        raise ValueError("Requested subgraph size is larger than the graph.")
    
    # Start with a random node
    start_node = random.choice(list(G.nodes))
    visited = set([start_node])
    queue = [start_node]

    # Perform a BFS/DFS until the subgraph has the desired number of nodes
    while len(visited) < size and queue:
        node = queue.pop(0)  # For BFS; use pop() for DFS
        neighbors = list(G.neighbors(node))
        random.shuffle(neighbors)  # Shuffle neighbors to get a random selection
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
            if len(visited) == size:
                break

    return visited

def create_subnet(network, nodes_per_team):
    """
    Creates a smaller graph by selecting a fixed number of nodes from each label in the given dataset.

    Parameters:
    dataset (networkx.Graph): The original dataset graph.

    Returns:
    networkx.Graph: The new graph containing the selected samples.
    """

    # Create a list to store the selected samples
    selected_samples = []
    labels = ['T', 'DM', 'DB', 'AI']

    # Iterate over each label
    for label in labels:
        # Get all the nodes with the current label
        nodes_with_label = [node for node in network.nodes if network.nodes[node]['label'] == label]

        net_T = network.subgraph(nodes_with_label)

        nodes_in_component = get_connected_subgraph(net_T, nodes_per_team)

        # Add the selected nodes to the list of selected samples
        selected_samples.extend(nodes_in_component)

    # Create a new graph with the selected samples
    subnet = network.subgraph(selected_samples)

    return subnet


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


def remove_edges_based_on_project_network(expert_network, project_network):
    """
        We take out all edges that are not in the project network.
    """
    edges_to_remove = []

    for edge in expert_network.edges():
        node1_label = expert_network.nodes[edge[0]]['label']
        node2_label = expert_network.nodes[edge[1]]['label']

        if node1_label != node2_label and not project_network.has_edge(node1_label, node2_label):
            edges_to_remove.append(edge)

    expert_network.remove_edges_from(edges_to_remove)

    return expert_network

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
        min_total_edge_weight = float('-inf')

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
                if total_edge_weight > min_total_edge_weight:
                    min_total_edge_weight = total_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += min_total_edge_weight

    # print(f"Coordinators Communication Efficiency: {sum_edge_weights(graph_G.subgraph(subset))}")
    # print(f"Coordinators : {subset}")

    # return subset, sum_edge_weights(graph_G.subgraph(subset))
    return subset, round(communication_efficiency, 4)


def RandomGreedy(graph_G, graph_P):
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Start with a random node from G
    key = random.choice(list(graph_G.nodes))
    print(f"Seed Node: {key}")
    subset = set()
    subset.add(key)
    labels = []
    labels.append(graph_G.nodes[key]['label'])
    communication_efficiency = 0.0

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
                    graph_G.subgraph(temp_subset)) + (average_weight_of_adjacent_nodes(graph_G, node))

                # Update the best node if the current node minimizes the total edge weight
                if total_edge_weight > min_total_edge_weight:
                    min_total_edge_weight = total_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += min_total_edge_weight
    print(
        f"Coordinators Communication Efficiency: {sum_edge_weights(graph_G.subgraph(subset))}")

    # return subset, round(sum_edge_weights(graph_G.subgraph(subset)), 4)
    return subset, round(communication_efficiency, 4)


def InfluenceGreedy(graph_G, graph_P):
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Start with a random node from G
    top_nodes = get_top_ranked_node_each_group(graph_G)
    key = random.choice(top_nodes)
    print(f"Seed Node: {key}")  
    subset = set()
    subset.add(key)
    labels = []
    labels.append(graph_G.nodes[key]['label'])
    communication_efficiency = 0.0

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
                    graph_G.subgraph(temp_subset)) + (average_weight_of_adjacent_nodes(graph_G, node))

                # Update the best node if the current node minimizes the total edge weight
                if total_edge_weight > min_total_edge_weight:
                    min_total_edge_weight = total_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += min_total_edge_weight

    print(
        f"Coordinators Communication Efficiency: {sum_edge_weights(graph_G.subgraph(subset))}")

    # return graph_G.subgraph(subset)
    return subset, round(communication_efficiency, 4)


def add_weights(network, alpha=0.5, criterion='min'):
    """
    Adds edges to the network with a weight value based on the minimum weight in the network.

    Parameters:
    - network: The network to add edges to.

    Returns:
    - network
    """

    # Find the minimum weight in the network
    if criterion == 'min':
        min_weight = min(nx.get_edge_attributes(network, 'weight').values())
    if criterion == 'mean':
        min_weight = sum(nx.get_edge_attributes(network, 'weight').values())/len(nx.get_edge_attributes(network, 'weight').values())

    # Iterate over all pairs of nodes
    for node1 in network.nodes():
        for node2 in network.nodes():
            # Check if there is no edge between the nodes
            if not network.has_edge(node1, node2):
                # Add the edge with the weight value
                network.add_edge(node1, node2, weight=(alpha * min_weight))

    return network

# Get the subgraph of a given node in G based on the labels in P
def subgraph_by_label(G, P, node_g):
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