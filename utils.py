import json
import re
import networkx as nx
import pickle
import random

# Find connected components
connected_components = list(nx.connected_components(G))

# Find the largest connected component
largest_connected_component = max(connected_components, key=len)
largest_subgraph = G.subgraph(largest_connected_component)

# Print properties of the largest connected component
print("Properties of the Largest Connected Component:")
print(f"Number of Nodes: {len(largest_connected_component)}")
print(f"Number of Edges: {largest_subgraph.number_of_edges()}")
print(f"Diameter: {nx.diameter(largest_subgraph)}" if len(
    largest_connected_component) > 1 else "Diameter: Not applicable (single node)")

# Save the largest connected component to a pickle file
with open('largest_cc.pkl', 'wb') as connected_component_file:
    pickle.dump(largest_subgraph, connected_component_file)


# Generate random numbers between 1 and 100
random_numbers = [random.randint(1, 100) for _ in range(len(network.edges))]

# Multiply the weights with the generated random numbers
for edge, random_number in zip(network.edges, random_numbers):
    network.edges[edge]['weight'] *= random_number

# Create a graph

new_data = pickle.load(open('author_data.pkl', 'rb'))
G = nx.Graph()

# Iterate through the data list and add nodes and edges to the graph
for data in new_data:
    author_id = data['id']
    author_name = data['author']
    coauthors = data['coauthors']
    venues = data['venues']
    papers = data['papers']
    num_of_papers = data['num_of_papers']

    # Add nodes with data
    G.add_node(author_name, id=author_id, author=author_name, coauthors=coauthors,
               venues=venues, papers=papers, num_of_papers=num_of_papers)

    # Add edges with weights
    for coauthor in coauthors:
        if G.has_edge(author_name, coauthor):
            G[author_name][coauthor]['weight'] += 1
        else:
            G.add_edge(author_name, coauthor, weight=1)

# General information about the graph
print("-----Graph Information-----")
print(f"Number of Nodes: {G.number_of_nodes()}")
print(f"Number of Edges: {G.number_of_edges()}")
print()

# Update node labels from author names to author IDs
node_mapping = {author_data['author']: author_data['id']
                for author_data in new_data}
G = nx.relabel_nodes(G, node_mapping, copy=False)


# Get list of all authors
list_of_authors = set()

for paper in data:
    authors = paper.get('authors')[0].split(',')
    for author in authors:
        # print(author)
        list_of_authors.add(author)

list_of_authors = list(list_of_authors)
list_of_authors = [item for item in list_of_authors if item != '']
print(len(list_of_authors))

# Get Author Details


def get_author_data(list_of_authors, papers):

    id_value = 0
    authors_data = []
    for author in list_of_authors:
        user = Author()
        user.id = id_value
        user.name = author
        for paper in papers:
            authors_in_paper = set(paper.get('authors', '')[0].split(','))
            if len(set([author]).intersection(authors_in_paper)) == 1:
                user.coauthors.extend(set(authors_in_paper)-set([user.name]))
                # user.coauthors.extend(list((authors_in_paper) - set([author])))
                user.papers.extend(
                    set([paper.get('title')]) - set(user.papers))
                user.venues.extend(
                    set([paper.get('venue')]) - set(user.venues))

        id_value = id_value + 1
        authors_data.append(user.author_info())

    # return authors_data

        # Save the graph as a pickle file
    with open('author_data.pkl', 'wb') as all_authors_data:
        pickle.dump(authors_data, all_authors_data)


# Create Author attributes
class Author:
    def __init__(self):
        # Define class attributes to hold values from the dictionary
        self.id = None
        self.name = None
        self.coauthors = []
        self.venues = []
        self.papers = []

    def author_info(self):
        dictionary = {
            'id': self.id,
            'author': self.name,
            'coauthors': self.coauthors,
            'venues': self.venues,
            'papers': self.papers,
            'num_of_papers': self.get_num_of_papers()
        }
        return dictionary

    def get_num_of_papers(self):
        return len(self.papers)


# Count Authors per venue


def count_papers_and_authors_per_venue(filtered_papers):
    venue_counts = {}

    for paper in filtered_papers:
        venue = paper.get('venue', '').lower()
        authors = paper.get('authors', [])

        # Count papers per venue
        venue_counts.setdefault(venue, {'papers': 0, 'authors': set()})
        venue_counts[venue]['papers'] += 1

        # Count authors per venue
        venue_counts[venue]['authors'].update(authors)

    return venue_counts


def main():
    with open('filtered_dataset.pkl', 'rb') as filtered_pickle_file:
        filtered_papers = pickle.load(filtered_pickle_file)

    venue_counts = count_papers_and_authors_per_venue(filtered_papers)

    print("Venue\t\t\tPapers\tAuthors")
    print("-" * 40)

    for venue, counts in venue_counts.items():
        num_papers = counts['papers']
        num_authors = len(counts['authors'])
        print(f"{venue.ljust(15)}\t{num_papers}\t{num_authors}")


if __name__ == "__main__":
    main()


# Filter by venue


def filter_by_venues(papers, target_venues):
    filtered_papers = [paper for paper in papers if paper.get(
        'venue', '').lower() in target_venues]
    return filtered_papers


def main():
    with open('papers.pkl', 'rb') as pickle_file:
        papers = pickle.load(pickle_file)

    target_venues = {'aaai', 'ijcai', 'kdd', 'nips', 'aamas'}
    filtered_papers = filter_by_venues(papers, target_venues)

    # Save the filtered dataset as a new pickle file
    with open('filtered_papers.pkl', 'wb') as filtered_pickle_file:
        pickle.dump(filtered_papers, filtered_pickle_file)


if __name__ == "__main__":
    main()


# Count papers in a venue


def count_aaai_venues(papers):
    aaai_count = 0

    for paper in papers:
        venue = paper.get('venue', '')
        if 'aaai' in venue.lower():
            aaai_count += 1

    return aaai_count


def main():
    with open('papers.pkl', 'rb') as pickle_file:
        papers = pickle.load(pickle_file)

    aaai_count = count_aaai_venues(papers)

    print("Number of venues containing 'AAAI':", aaai_count)


if __name__ == "__main__":
    main()


# Get all venues


def get_all_venues(papers):
    all_venues = set()

    for paper in papers:
        venue = paper.get('venue', '')
        if venue:
            all_venues.add(venue)

    return list(all_venues)


def main():
    with open('papers.pkl', 'rb') as pickle_file:
        papers = pickle.load(pickle_file)

    venues_list = get_all_venues(papers)

    print("List of All Venues:")
    for venue in venues_list:
        print(venue)


if __name__ == "__main__":
    main()


# Get data from text file.


def parse_paper_data(data):
    paper = {}
    lines = data.split('\n')

    for line in lines:
        if line.startswith("#*"):
            paper['title'] = line[2:].strip()
        elif line.startswith("#@"):
            paper['authors'] = line[2:].strip().split(', ')
        elif line.startswith("#t"):
            paper['year'] = int(line[2:].strip())
        elif line.startswith("#c"):
            paper['venue'] = line[2:].strip()
        elif line.startswith("#index"):
            paper['index_id'] = int(line.split()[1])
        elif line.startswith("#%"):
            paper.setdefault('references', []).append(int(line.split()[1]))
        elif line.startswith("#!"):
            paper['abstract'] = line[2:].strip()

    return paper


def main():
    with open('dblp_data.txt', 'r') as file:
        # assuming each paper is separated by two newlines
        papers_data = file.read().split('\n\n')

    papers = [parse_paper_data(paper_data) for paper_data in papers_data]

    # Save as pickle
    with open('dataset.pkl', 'wb') as pickle_file:
        pickle.dump(papers, pickle_file)


if __name__ == "__main__":
    main()


network = pickle.load(open('largest_cc.pkl', 'rb'))

# Add edges between authors who share a common neighbor
for node1, data1 in network.nodes(data=True):
    for node2, data2 in network.nodes(data=True):
        common_neighbors = set(network.neighbors(
            node1)) & set(network.neighbors(node2))
        if node1 != node2 and not network.has_edge(node1, node2) and common_neighbors:
            network.add_edge(node1, node2, weight=0.5)

# Add edges between nodes with the same venue and no existing edge
for node1, data1 in network.nodes(data=True):
    for node2, data2 in network.nodes(data=True):
        if node1 != node2 and data1['venues'] == data2['venues'] and not network.has_edge(node1, node2):
            network.add_edge(node1, node2, weight=0.5)

# Add edges between authors who share a common neighbor
for node1, data1 in network.nodes(data=True):
    for node2, data2 in network.nodes(data=True):
        if node1 != node2 and not network.has_edge(node1, node2):
            network.add_edge(node1, node2, weight=0)


# # Add edges between authors who share a common neighbor
# for node1, data1 in network.nodes(data=True):
#     for node2, data2 in network.nodes(data=True):
#         if node1 != node2 and not network.has_edge(node1, node2):
#             network.add_edge(node1, node2, weight=0.1)


# Save the new network to a pickle file
with open('network.pkl', 'wb') as new_net:
    pickle.dump(network, new_net)


top_10_authors = []
# Create subgraphs for each conference with the top 10 authors by the number of papers
conferences = []
for venue in nx.get_node_attributes(network, 'venues').values():
    conferences.extend(venue)

for conference in set(conferences):
    conference_authors = [author for author, data in network.nodes(
        data=True) if data['venues'][0] == conference]
    top_authors = sorted(
        conference_authors, key=lambda author: network.nodes[author]['num_of_papers'], reverse=True)[:10]
    top_10_authors.extend(top_authors)

subgraph = network.subgraph(top_10_authors)


network = pickle.load(open('one_venue_network.pkl', 'rb'))

# Scaling the weights of the edges to be as a percentage of the maximum edge weight
import networkx as nx

def scale_edge_weights(graph):
    # Step 1: Find the maximum edge weight
    max_edge_weight = max([data['weight'] for _, _, data in graph.edges(data=True)], default=1)

    # Step 2: Scale each edge weight as a percentage of the maximum edge weight
    for u, v, data in graph.edges(data=True):
        if 'weight' in data:
            data['weight'] = (data['weight'] / max_edge_weight) * 100

    return graph
# Example usage:

net = scale_edge_weights(network)


# Save the adjusted weighted network to a pickle file
with open('weighted_network.pkl', 'wb') as connected_component_file:
    pickle.dump(net, connected_component_file)


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


def createProjectNetwork(list):
    project = nx.Graph()
    project.add_edges_from(list)
    return project


def sum_edge_weights(graph):
    total_weight = 0

    for _, _, data in graph.edges(data=True):
        if 'weight' in data:
            total_weight += data['weight']

    return total_weight


def randomGreedy(graph_G, graph_P):
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
    labels.append(graph_G.nodes[key]['venues'])

    while len(subset) < len(graph_P.nodes):
        best_node = None
        min_total_edge_weight = 0.0

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['venues'] not in labels:
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
        labels.append(graph_G.nodes[best_node]['venues'])

    return graph_G.subgraph(subset)


def influenceGreedy(graph_G, graph_P):
    if graph_G is None or graph_P is None:
        print("Error: One or both of the graphs is None.")
        return None

    if len(graph_P.nodes) > len(graph_G.nodes):
        print("Error: Number of nodes in P is greater than the number of nodes in G.")
        return None

    # Start with a random node from influence set
    graph_G = compute_influence(graph_G)
    top_nodes = [
        node for node in graph_G.nodes if graph_G.nodes[node]['influence'] == 100]
    for node in top_nodes:
        print(graph_G.nodes[node])
    key = random.choice(top_nodes)
    subset = set()
    subset.add(key)
    labels = []
    labels.append(graph_G.nodes[key]['venues'])

    while len(subset) < len(graph_P.nodes):
        best_node = None
        min_total_edge_weight = 0.0

        # Iterate over nodes in G not in the subset
        for node in set(graph_G.nodes) - subset:
            # Create a temporary subset with the new node
            temp_subset = subset.copy()
            if graph_G.nodes[node]['venues'] not in labels:
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
        labels.append(graph_G.nodes[best_node]['venues'])

    return graph_G.subgraph(subset)


def createProjectNetwork(list):
    project = nx.Graph()
    project.add_edges_from(list)
    return project


project_1 = [('NIPS', 'IJCAI'), ('NIPS', 'AAAI'), ('NIPS', 'AAMAS'), ('NIPS', 'KDD'), ('IJCAI', 'AAAI'),
             ('IJCAI', 'AAMAS'), ('IJCAI', 'KDD'), ('AAAI', 'AAMAS'), ('AAAI', 'KDD'), ('AAMAS', 'KDD')]

project_1 = createProjectNetwork(project_1)

def monte_carlo(f, graph_G, graph_P, num_iter):
    comm_eff = 0
    selected_nodes = set()
    for i in range(num_iter):
        best = f(graph_G, graph_P)
        selected_nodes = selected_nodes.union(set(best.nodes))
        eff = sum_edge_weights(best)
        comm_eff = comm_eff + eff
    avg_comm_eff = comm_eff/num_iter
    print(f"Selected nodes: {selected_nodes}")
    print(f"Average Communication efficiency is : {avg_comm_eff}")

    return selected_nodes, avg_comm_eff

coordinators, avg_comm_eff = monte_carlo(randomGreedy,subgraph, project_1, 1000)
coordinators, avg_comm_eff = monte_carlo(influenceGreedy,subgraph, project_1, 1000)