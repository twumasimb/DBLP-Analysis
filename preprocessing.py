import pickle
import random
import networkx as nx


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
        if len(data['papers']) >= 2:
            user = Author()
            user.id = id_value
            user.name = author
            user.coauthors = data['coauthors']
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


def get_node_rank(graph, node):
    label = graph.nodes[node]['label']
    node_ranks = {}
    for n in graph.nodes():
        if graph.nodes[n]['label'] == label:
            rank = average_weight_of_adjacent_nodes(graph, n)
            node_ranks[n] = rank

    ranked_nodes = sorted(node_ranks, key=node_ranks.get, reverse=True)
    node_rank = ranked_nodes.index(node) + 1
    # print(ranked_nodes)
    return node_rank


def average_weight_of_adjacent_nodes(graph, node):
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
        return total_weight / count


def sum_edge_weights(graph):
    total_weight = 0

    for _, _, data in graph.edges(data=True):
        if 'weight' in data:
            total_weight += data['weight']

    return total_weight


def Greedy(graph_G, graph_P, seed_node):
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

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

                # Update the best node if the current node maximizes the total edge weight
                if total_edge_weight > min_total_edge_weight:
                    min_total_edge_weight = total_edge_weight
                    best_node = node

        # Add the best node to the subset
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['label'])
        communication_efficiency += min_total_edge_weight

    # print(f"Coordinators Communication Efficiency: {utils.sum_edge_weights(graph_G.subgraph(subset))}")
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
    print(f"Coordinators : {subset}")

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
    top_nodes = [
        node for node in graph_G.nodes if graph_G.nodes[node]['influence'] == 100]
    for node in top_nodes:
        print(graph_G.nodes[node])
    key = random.choice(top_nodes)
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
    print(f"Coordinators : {subset}")

    # return graph_G.subgraph(subset)
    return subset, round(communication_efficiency, 4)