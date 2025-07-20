import networkx as nx

def are_isomorphic(n1, n2, edges1, edges2):
    if n1 != n2:
        return False

    graph1 = nx.Graph()
    graph1.add_nodes_from(range(1, n1 + 1))
    graph1.add_edges_from(edges1)

    graph2 = nx.Graph()
    graph2.add_nodes_from(range(1, n2 + 1))
    graph2.add_edges_from(edges2)

    GM = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2)
    return GM.is_isomorphic()

if __name__ == '__main__':
    n1, m1 = list(map(int, input().split()))
    edges1 = []
    for _ in range(m1):
        u, v = list(map(int, input().split()))
        edges1.append((u, v))

    n2, m2 = list(map(int, input().split()))
    edges2 = []
    for _ in range(m2):
        u, v = list(map(int, input().split()))
        edges2.append((u, v))
    
    ans = 1 if are_isomorphic(n1, n2, edges1, edges2) else 0
    print(ans)
