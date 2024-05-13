import networkx as nx
import matplotlib.pyplot as plt


# Create a graph with 3 nodes and 2 edges
def create_graph_init():
    graphic = nx.Graph()

    graphic.add_node(1)
    graphic.add_nodes_from([2, 3])

    graphic.add_edge(1, 2)
    graphic.add_edges_from([(1, 3), (2, 3)])

    nx.draw(graphic, with_labels=True, node_color='red')
    plt.show()


# Las medidas de centralidad son métricas que te permiten entender la importancia de un nodo dentro de una red.
# Puedes calcular la centralidad de grado, cercanía e intermediación.
def calculate_measure_of_centrality():
    points = 15
    graphic = nx.gnp_random_graph(points, 0.5, directed=False)

    degree_centrality = nx.degree_centrality(
        graphic)  # La centralidad de grado es el número de enlaces que inciden en un nodo.
    print("Degree Centrality: ", degree_centrality)
    closeness_centrality = nx.closeness_centrality(
        graphic)  # La centralidad de cercanía es la longitud promedio del camino más corto entre el nodo y todos los
    # demás nodos en el gráfico.
    print("Closeness Centrality: ", closeness_centrality)
    betweenness_centrality = nx.betweenness_centrality(
        graphic)  # La centralidad de intermediación es el número de caminos más cortos que pasan por el nodo.
    print("Betweenness Centrality: ", betweenness_centrality)
    nx.draw(graphic, with_labels=True, node_color=range(points))
    plt.show()


# La detección de comunidades es una técnica que te permite identificar grupos de nodos que están más conectados
# entre sí que con el resto de la red.
def detect_communities():
    points = 10
    graphic = nx.gnp_random_graph(points, 0.5, directed=True)
    communities = nx.community.greedy_modularity_communities(
        graphic)  # La maximización codiciosa de la modularidad es un método rápido para detectar comunidades en redes.
    print("Communities: ", communities)
    nx.draw(graphic, with_labels=True, node_color=range(points))
    plt.show()


# Mostrar el grafo de la red social de un club de karate que se divide en dos comunidades (clubes) diferentes.
def show_graph_karate_to_show_club_members():
    graphic = nx.karate_club_graph()  # Zachary's karate club graph is a social network of a university karate club
    nx.draw(graphic, with_labels=True, node_color=range(len(graphic.nodes())), cmap=plt.colormaps.get_cmap('cool'))
    plt.show()
    print("Club: ", graphic.nodes[33]['club'])
    print("Club: ", graphic.nodes[1]['club'])


# Los grafos dirigidos son aquellos en los que las aristas tienen una dirección, es decir, van de un nodo a otro.
# Tiene un solo sentido
def show_graph_managed():
    graphic = nx.DiGraph()

    graphic.add_edge(1, 2)
    graphic.add_edge(2, 3)
    graphic.add_edge(3, 4)
    graphic.add_edge(3, 5)

    nx.draw(graphic, with_labels=True, node_color='green')
    plt.show()


# Encuentra la ruta más corta entre dos nodos en un gráfico.
def show_shortest_path():
    graphic = nx.Graph()

    graphic.add_edge('A', 'B', weight=1)
    graphic.add_edge('B', 'C', weight=2)
    graphic.add_edge('A', 'C', weight=3)
    graphic.add_edge('C', 'D', weight=4)

    # Encontrar la ruta más corta de A a D
    shortest_path = nx.shortest_path(graphic, 'A', 'D', weight='weight')
    print("Shortest Path: ", shortest_path)

    # Encontrar la longitud de la ruta más corta de A a D
    shortest_path_length = nx.shortest_path_length(graphic, 'A', 'D', weight='weight')
    print("Shortest Path Length: ", shortest_path_length)

    # Mostrar en un gráfico
    nx.draw(graphic, with_labels=True, node_color='yellow')
    plt.show()


# Encuentra los componentes conectados en un gráfico.
def show_connected_components():
    graphic = nx.Graph()

    # Añade tus nodos y aristas aquí
    graphic.add_edge('A', 'B')
    graphic.add_edge('B', 'C')
    graphic.add_edge('D', 'E')
    graphic.add_edge('F', 'G')
    graphic.add_edge('G', 'H')

    # Encuentra todos los componentes conectados
    connected_components = nx.connected_components(graphic)

    # Imprime cada componente conectado
    for component in connected_components:
        print("Connected component: ", component)

    # Muestra el gráfico
    nx.draw(graphic, with_labels=True, node_color=range(len(graphic.nodes())), cmap=plt.colormaps.get_cmap('cool'))
    plt.show()


# Analizar la red de Les Misérables para encontrar la importancia de cada personaje en la historia.
def analyze_les_miserables_network():
    graphic = nx.les_miserables_graph()

    # Calcula el grado de cada nodo
    degrees = [graphic.degree(n) for n in graphic.nodes()]

    # Imprime el grado de cada nodo
    for node, degree in zip(graphic.nodes(), degrees):
        print(f"Node {node} has degree: {degree}")

    # Dibuja el grafo
    nx.draw(graphic, with_labels=True, node_color=range(len(graphic.nodes())), cmap=plt.colormaps.get_cmap('cool'))
    plt.show()


if __name__ == '__main__':
    create_graph_init()
    calculate_measure_of_centrality()
    detect_communities()
    show_graph_karate_to_show_club_members()
    show_graph_managed()
    show_shortest_path()
    show_connected_components()
    analyze_les_miserables_network()
