import networkx as nx
from collections import defaultdict
import heapq

class Individual:
    def __init__(self, id, skills):
        self.id = id
        self.skills = set(skills)

def create_graph(individuals, connections):
    G = nx.Graph()
    for ind in individuals:
        G.add_node(ind.id, individual=ind)
    for i, j, weight in connections:
        G.add_edge(i, j, weight=weight)
    return G

def shortest_path(G, start, end):
    return nx.shortest_path(G, start, end, weight='weight')

def calculate_distance(G, path):
    return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))

# RarestFirst algorithm
def rarest_first(G, task):
    skills_support = defaultdict(set)
    for node in G.nodes():
        ind = G.nodes[node]['individual']
        for skill in ind.skills:
            skills_support[skill].add(node)
    
    rarest_skill = min(skills_support.items(), key=lambda x: len(x[1]))
    
    best_individual = None
    min_diameter = float('inf')
    
    for candidate in skills_support[rarest_skill[0]]:
        max_distance = 0
        for skill in task:
            if skill not in G.nodes[candidate]['individual'].skills:
                closest = min(skills_support[skill], key=lambda x: nx.shortest_path_length(G, candidate, x, weight='weight'))
                distance = nx.shortest_path_length(G, candidate, closest, weight='weight')
                max_distance = max(max_distance, distance)
        
        if max_distance < min_diameter:
            min_diameter = max_distance
            best_individual = candidate
    
    team = set([best_individual])
    for skill in task:
        if skill not in G.nodes[best_individual]['individual'].skills:
            closest = min(skills_support[skill], key=lambda x: nx.shortest_path_length(G, best_individual, x, weight='weight'))
            path = shortest_path(G, best_individual, closest)
            team.update(path)
    
    return team

# EnhancedSteiner algorithm
def enhanced_steiner(G, task):
    H = G.copy()
    skill_nodes = {}
    for skill in task:
        skill_node = f"skill_{skill}"
        H.add_node(skill_node)
        skill_nodes[skill] = skill_node
        for node in G.nodes():
            if skill in G.nodes[node]['individual'].skills:
                H.add_edge(node, skill_node, weight=float('inf'))
    
    steiner_nodes = list(skill_nodes.values())
    mst = nx.algorithms.approximation.steiner_tree(H, steiner_nodes)
    
    team = set(mst.nodes()) - set(skill_nodes.values())
    return team

# CoverSteiner algorithm
def cover_steiner(G, task):
    def greedy_cover(individuals, task):
        covered = set()
        team = set()
        while covered != task:
            best = max(individuals, key=lambda x: len(set(x.skills) & (task - covered)))
            team.add(best.id)
            covered |= set(best.skills) & task
        return team
    
    individuals = [G.nodes[node]['individual'] for node in G.nodes()]
    cover = greedy_cover(individuals, set(task))
    
    steiner_tree = nx.algorithms.approximation.steiner_tree(G, cover)
    return set(steiner_tree.nodes())

# Example usage
individuals = [
    Individual(1, ['a', 'b']),
    Individual(2, ['b', 'c']),
    Individual(3, ['c', 'd']),
    Individual(4, ['d', 'e']),
    Individual(5, ['e', 'a'])
]

connections = [
    (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 1, 1)
]

G = create_graph(individuals, connections)
task = ['a', 'c', 'e']

print("RarestFirst team:", rarest_first(G, task))
print("EnhancedSteiner team:", enhanced_steiner(G, task))
print("CoverSteiner team:", cover_steiner(G, task))