
#ROC+S
#Libraries
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# Size of the ROC+
num_cliques = 5  # Number of cliques
clique_size = 5  # Nodes per clique

def create_roc_with_outliers(num_cliques, clique_size):
    G = nx.Graph()
    labels = {}
    
    for i in range(num_cliques):
        clique_nodes = range(i * clique_size, (i + 1) * clique_size)
        G.add_nodes_from(clique_nodes)
        G.add_edges_from([(u, v) for u in clique_nodes for v in clique_nodes if u != v])
        
        # Assign community labels
        for node in clique_nodes:
            labels[node] = i
        
        # Add an outlier node with a unique label
        outlier_node = num_cliques * clique_size + i
        G.add_node(outlier_node)
        labels[outlier_node] = -(i + 1)  # Unique label for each outlier
        
        # Connect outlier to cliques
        G.add_edge(clique_nodes[-1], outlier_node)
        if i < num_cliques - 1:
            G.add_edge(outlier_node, (i + 1) * clique_size)
        else:
            G.add_edge(outlier_node, 0)
    print("adjacency matrix", nx.adjacency_matrix(G))   
    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    print(adjacency_matrix)     
    print(G)
    print(labels)
    return G, labels

# Create the ROC with uniquely labeled outliers
G_roc, labels_roc = create_roc_with_outliers(num_cliques, clique_size)
labels_array = np.array([labels_roc[node] for node in G_roc.nodes()])

# Visualization of the ROC dataset with outliers
pos = nx.spring_layout(G_roc)  # Layout for visualization
plt.figure(figsize=(10, 10))

# Map community labels to colors
community_colors = [labels_roc[node] for node in G_roc.nodes()]

# Draw nodes with community colors
nx.draw_networkx_nodes(G_roc, pos, node_color=community_colors, cmap=plt.cm.tab10, node_size=300)
nx.draw_networkx_edges(G_roc, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G_roc, pos, font_size=10, font_color='black')

# Add a colorbar for the ground truth communities
sm = plt.cm.ScalarMappable(cmap=plt.cm.tab10, norm=plt.Normalize(vmin=-num_cliques, vmax=num_cliques - 1))
sm.set_array([])
plt.colorbar(sm, ticks=range(-num_cliques, num_cliques), label="Ground Truth Community", ax=plt.gca())

plt.title("Ring of Cliques (ROC) with Uniquely Labeled Outliers")
plt.axis('off')  # Turn off axis
plt.show()