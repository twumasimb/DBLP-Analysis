import pickle
import networkx as nx
import random
import chime


def randomAlgo(network):
    """
    Randomly selects a node from each unique venue in the network.

    Parameters:
    - network: The network graph.

    Returns:
    - A list of selected nodes.
    """
    # Get node venues
    node_venues = nx.get_node_attributes(network, 'venues')

    selected_nodes = {}
    for node, venues in node_venues.items():
        for venue in venues:
            # If the venue is not already in selected_nodes, add the node
            if venue not in selected_nodes:
                selected_nodes[venue] = node

    return selected_nodes.values()


def remove_edges_based_on_project_network(expert_network, project_network):
    """
    Removes edges from the expert_network that do not exist in the project_network.

    Args:
        expert_network (networkx.Graph): The expert network graph.
        project_network (networkx.Graph): The project network graph.

    Returns:
        networkx.Graph: The expert network graph with edges removed.
    """
    edges_to_remove = [(node1, node2) for node1, node2 in expert_network.edges()
                       if not project_network.has_edge(expert_network.nodes[node1]['venues'][0],
                                                       expert_network.nodes[node2]['venues'][0])]

    expert_network.remove_edges_from(edges_to_remove)

    return expert_network


def compute_influence(graph):
    """
    Computes the influence score for each node in the given graph.

    Parameters:
    graph (networkx.Graph): The graph for which to compute the influence score.

    Returns:
    networkx.Graph: The graph with influence scores assigned to each node.
    """

    if graph is None:
        print("Error: Graph is None.")
        return None

    # Compute influence scores
    for node, data in graph.nodes(data=True):
        total_weight = sum(edge['weight']
                           for _, _, edge in graph.edges(node, data=True))
        num_papers = len(data["papers"])
        num_coauthors = len(data["coauthors"])
        influence = num_papers + 1.5 * num_coauthors + total_weight
        data['influence'] = influence

    # Scale scores to 100
    max_influence = max(data['influence']
                        for _, data in graph.nodes(data=True))
    scale_factor = 100 / max_influence

    for _, data in graph.nodes(data=True):
        data['influence'] *= scale_factor

    return graph


def get_top_node_per_venue(graph):
    """
    Returns a dictionary containing the top node per venue based on influence score.

    Args:
        graph (networkx.Graph): The input graph representing the network.

    Returns:
        tuple: A tuple containing two elements:
            - A dictionary where the keys are venues and the values are the top nodes for each venue.
            - A list of the top nodes for all venues.
    """
    top_nodes = {}
    nodes = []
    graph = compute_influence(graph)
    for node in graph.nodes:
        venue = graph.nodes[node]['venues'][0]
        if venue not in top_nodes:
            top_nodes[venue] = node
        elif graph.nodes[node]['influence'] > graph.nodes[top_nodes[venue]]['influence']:
            top_nodes[venue] = node

    return top_nodes, list(top_nodes.values())


def createProjectNetwork(edges):
    """
    Creates a project network graph from a list of edges.

    Parameters:
    edges (list): A list of edges representing the connections between nodes in the project network.

    Returns:
    networkx.Graph: A networkx graph representing the project network.
    """
    project = nx.Graph()
    project.add_edges_from(edges)
    return project


def sum_edge_weights(graph):
    """
    Calculates the total weight of all edges in the given graph.

    Parameters:
    graph (networkx.Graph): The graph to calculate the total weight of.

    Returns:
    float: The total weight of all edges in the graph.
    """
    return sum(data.get('weight', 0) for _, _, data in graph.edges(data=True))


def monte_carlo(f, graph_G, graph_P, num_iter):
    """
    Perform Monte Carlo simulation to estimate the average communication efficiency.

    Parameters:
    f (function): A function that takes in graph_G and graph_P as input and returns a graph.
    graph_G (Graph): The network of experts.
    graph_P (Graph): The project network.
    num_iter (int): The number of iterations for the simulation.

    Returns:
    float: The average communication efficiency rounded to 2 decimal places.
    """

    comm_eff = sum(sum_edge_weights(f(graph_G, graph_P))
                   for _ in range(num_iter))
    avg_comm_eff = comm_eff / num_iter
    print(f"Average Communication efficiency is : {avg_comm_eff}")
    return round(avg_comm_eff, 2)


def randomGreedy(graph_G, graph_P):
    """
    This function implements a random greedy algorithm to find a subgraph of graph_G 
    that has the same number of nodes as graph_P. The algorithm starts with a random node 
    from graph_G and iteratively adds the node that maximizes the total edge weight of the subgraph.

    Parameters:
    graph_G (networkx.Graph): The graph to find the subgraph in.
    graph_P (networkx.Graph): The graph to match the number of nodes with.

    Returns:
    networkx.Graph: The resulting subgraph of graph_G.
    """
    if not graph_G or not graph_P or len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Invalid input graphs.")
        return None

    # Start with a random node from G
    subset = {random.choice(list(graph_G.nodes))}
    labels = [graph_G.nodes[next(iter(subset))]['venues'][0]]

    while len(subset) < len(graph_P.nodes):
        # Find the node that maximizes the total edge weight of the subgraph
        candidates = [(node, sum_edge_weights(graph_G.subgraph(list(subset) + [node])))
                      for node in set(graph_G.nodes) - subset
                      if graph_G.nodes[node]['venues'][0] not in labels]

        if not candidates:
            print("Warning: No suitable node found. Terminating the loop.")
            break

        # Add the best node to the subset
        best_node, _ = max(candidates, key=lambda x: x[1])
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['venues'][0])

    return graph_G.subgraph(subset)


def influenceGreedy(graph_G, graph_P):
    """
    This function implements a greedy algorithm to find a subgraph of graph_G 
    that has the same number of nodes as graph_P. The algorithm starts with a random node 
    from the top nodes per venue in graph_G and iteratively adds the node that maximizes 
    the total edge weight of the subgraph.

    Parameters:
    graph_G (networkx.Graph): The graph to find the subgraph in.
    graph_P (networkx.Graph): The graph to match the number of nodes with.

    Returns:
    networkx.Graph: The resulting subgraph of graph_G.
    """
    if not graph_G or not graph_P or len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Invalid input graphs.")
        return None

    # Start with a random node from the top nodes per venue
    top_nodes = get_top_node_per_venue(graph_G)
    key = random.choice(top_nodes[1])
    subset = {key}
    labels = [graph_G.nodes[key]['venues'][0]]

    while len(subset) < len(graph_P.nodes):
        # Find the node that maximizes the total edge weight of the subgraph
        candidates = [(node, sum_edge_weights(graph_G.subgraph(list(subset) + [node])))
                      for node in set(graph_G.nodes) - subset
                      if graph_G.nodes[node]['venues'][0] not in labels]

        if not candidates:
            print("Warning: No suitable node found. Terminating the loop.")
            break

        # Add the best node to the subset
        best_node, _ = max(candidates, key=lambda x: x[1])
        subset.add(best_node)
        labels.append(graph_G.nodes[best_node]['venues'][0])

    return graph_G.subgraph(subset)


def randomMonteCarlo(graph, num_iter):
    total_weight = 0

    for _ in range(num_iter):
        total_weight += sum_edge_weights(graph.subgraph(randomAlgo(graph)))

    avg_weight = round(total_weight / num_iter, 2)
    return avg_weight

    ####### PREPROCESSING #######


def process_paper_data(input_file, output_file, sample_size=None):
    """
    Parses paper data from a text file, saves it as a pickle file, and creates a graph from the author data.

    Parameters:
    input_file (str): The path to the text file containing the paper data.
    output_file (str): The path to the output file to save the parsed paper data to.
    sample_size (int, optional): The number of random samples to take from the author data. If None, all author data is used.
    """
    # Parse paper data and save it as a pickle file
    print('Parsing paper data...')
    parse_and_save_paper_data(input_file, output_file)
    print('Paper data parsed and saved.')

    # Load parsed paper data
    print('Loading parsed paper data...')
    with open(output_file, 'rb') as pickle_file:
        papers = pickle.load(pickle_file)
    print('Parsed paper data loaded.')

    # Get list of authors
    print('Getting list of authors...')
    list_of_authors = set(
        author for paper in papers for author in paper.get('authors', []))
    print('List of authors obtained.')

    # Get author data
    print('Getting author data...')
    authors_data = get_author_data(list_of_authors, papers)
    print('Author data obtained.')

    # If a sample size is provided, take a random sample of the author data
    if sample_size is not None:
        print(f'Taking a random sample of {sample_size} items...')
        authors_data = random.sample(authors_data, sample_size)
        print('Sample obtained.')

    # Create a graph from the author data
    print('Creating graph from author data...')
    G = create_graph_from_data(authors_data)
    print('Graph created.')

    # Save the graph as a pickle file
    print('Saving graph...')
    with open('graph.pkl', 'wb') as file:
        pickle.dump(G, file)
    print('Graph saved.')


# process_paper_data('dblp_data.txt', 'papers.pkl', 'author_data.pkl')


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

        return paper

    with open(input_file, 'r', encoding='utf-8') as file:
        # assuming each paper is separated by two newlines
        papers_data = file.read().split('\n\n')

    papers = [parse_paper_data(paper_data) for paper_data in papers_data]

    # Save as pickle
    with open(output_file, 'wb') as pickle_file:
        pickle.dump(papers, pickle_file)


def get_author_data(list_of_authors, papers):
    id_value = 0
    authors_data = []
    for author in list_of_authors:
        user = Author()
        user.id = id_value
        user.name = author
        for paper in papers:
            # print(paper)
            authors_in_paper = paper.get('authors', '')
            if authors_in_paper:
                authors_in_paper = set(authors_in_paper[0].split(','))
            else:
                authors_in_paper = set()
            if len(set([author]).intersection(authors_in_paper)) == 1:
                user.coauthors.extend(set(authors_in_paper)-set([user.name]))
                user.papers.extend(
                    set([paper.get('title')]) - set(user.papers))
                user.venues.extend(
                    set([paper.get('venue')]) - set(user.venues))
                # Update the number of papers published at each venue
                user.venue_papers[paper.get('venue')] = user.venue_papers.get(
                    paper.get('venue'), 0) + 1

        id_value = id_value + 1
        authors_data.append(user.author_info())

    return authors_data


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
        self.coauthors = []
        self.venues = []
        self.papers = []
        self.venue_papers = {}

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
            'num_of_papers': self.get_num_of_papers(),
            'venue_papers': self.venue_papers
        }
        return dictionary

    def get_num_of_papers(self):
        """
        Returns the number of papers authored by the author.

        Returns:
            int: The number of papers authored by the author.
        """
        return len(self.papers)


def create_graph_from_data(data):
    """
    Creates a graph from author data.

    Parameters:
    data (list): A list of dictionaries representing the author data.

    Returns:
    networkx.Graph: A networkx graph representing the author data.
    """
    # Create a graph
    G = nx.Graph()

    # Iterate through the data list and add nodes and edges to the graph
    for item in data:
        author_id = item['id']
        author_name = item['author']
        coauthors = item['coauthors']
        venue_papers = item['venue_papers']
        papers = item['papers']
        num_of_papers = item['num_of_papers']

        # If there is more than one venue, set the venue to the one with the highest value in venue_papers
        if len(item['venues']) > 1:
            venue = max(venue_papers, key=venue_papers.get)
        else:
            venue = item['venues'][0]

        # Add nodes with data
        G.add_node(author_name, id=author_id, author=author_name, coauthors=coauthors,
                   venue=venue, papers=papers, num_of_papers=num_of_papers)

        # Add edges with weights
        for coauthor in coauthors:
            if G.has_edge(author_name, coauthor):
                G[author_name][coauthor]['weight'] += 1
            else:
                G.add_edge(author_name, coauthor, weight=1)

    # Update node labels from author names to author IDs
    node_mapping = {item['author']: item['id'] for item in data}
    G = nx.relabel_nodes(G, node_mapping, copy=False)

    return G
