
#ROC+L
#libraries
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# Size of the ROC+
num_cliques = 5  # Number of cliques
clique_size = 5  # Nodes per clique

def create_roc_with_grouped_outliers(num_cliques, clique_size):
    G = nx.Graph()
    labels = {}
    
    for i in range(num_cliques):
        clique_nodes = list(range(i * clique_size, (i + 1) * clique_size))
        G.add_nodes_from(clique_nodes)
        G.add_edges_from([(u, v) for u in clique_nodes for v in clique_nodes if u != v])
        
        # Assign community labels
        for node in clique_nodes:
            labels[node] = i
        
        # Add an outlier node
        outlier_node = num_cliques * clique_size + i
        G.add_node(outlier_node)
        labels[outlier_node] = i  # Belongs to the same community

        # Connect outlier to the last node of the clique (bridge)
        G.add_edge(clique_nodes[-1], outlier_node)

        # Add a new edge from a different node in the clique to outlier
        candidates = [n for n in clique_nodes if n != clique_nodes[-1]]
        random_node = random.choice(candidates)
        G.add_edge(random_node, outlier_node)

        # Optionally connect outliers in a ring
        if i < num_cliques - 1:
            G.add_edge(outlier_node, (i + 1) * clique_size)
        else:
            G.add_edge(outlier_node, 0)

    return G, labels

# Create the updated ROC graph
G_roc, labels_roc = create_roc_with_grouped_outliers(num_cliques, clique_size)
labels_array = np.array([labels_roc[node] for node in G_roc.nodes()])

# Visualization of the ROC dataset with outliers assigned to cliques
pos = nx.spring_layout(G_roc, seed=42)
plt.figure(figsize=(8, 8))

community_colors = [labels_roc[node] for node in G_roc.nodes()]
nx.draw_networkx_nodes(G_roc, pos, node_color=community_colors, cmap=plt.cm.tab10, node_size=300)
nx.draw_networkx_edges(G_roc, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G_roc, pos, font_size=10, font_color='black')

sm = plt.cm.ScalarMappable(cmap=plt.cm.tab10, norm=plt.Normalize(vmin=0, vmax=num_cliques - 1))
sm.set_array([])
plt.colorbar(sm, ticks=range(num_cliques), label="Community (including outliers)", ax=plt.gca())

plt.title("ROC with Outliers Assigned to Cliques + Distinct New Edge")
plt.axis('off')
plt.show()
#print adjacency matrix
print(nx.adjacency_matrix(G_roc).A) 

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