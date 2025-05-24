#libraries
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
from node2vec import Node2Vec
from sklearn.cluster import AffinityPropagation, KMeans, DBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ROC+L
num_cliques = 15  # Number of cliques
clique_size = 30 # Nodes per clique

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

# Create the ROC+U
G_roc, labels_roc = create_roc_with_grouped_outliers(num_cliques, clique_size)
labels_array = np.array([labels_roc[node] for node in G_roc.nodes()])
unique_label_count = len(set(labels_array))

# >>>>>>>Function to run clustering multiple times
def run_clustering(G, labels, num_runs):
    results = pd.DataFrame({'Node': list(G.nodes()), 'True_Label': labels})
    nmi_aff, nmi_kmeans, nmi_dbscan, nmi_som = [], [], [], []

    for run in range(num_runs):
        #creat the Node2Vec
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
        
        #Affinity Propagation>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        # Compute similarity matrix (negative Euclidean distances)
        similarities = -euclidean_distances(embeddings)
        # Compute----1>>>>> preference value as average similarity ############>>>>>>>>>>>>>>>PArameter
        preference_value = np.median(similarities)
        print(preference_value)
        affinity_propagation = AffinityPropagation(preference=preference_value, random_state=run)
        pred_labels_aff = affinity_propagation.fit_predict(embeddings)
        results[f'Run_{run+1}_Affinity'] = pred_labels_aff
        nmi_aff.append(normalized_mutual_info_score(labels, pred_labels_aff))
        print(f'Run {run+1} NMI Score (Affinity): {nmi_aff[-1]:.4f}')

        # KMeans clustering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        kmeans = KMeans(n_clusters=unique_label_count, random_state=run, n_init=10)
        pred_labels_kmeans = kmeans.fit_predict(embeddings)
        results[f'Run_{run+1}_KMeans'] = pred_labels_kmeans
        nmi_kmeans.append(normalized_mutual_info_score(labels, pred_labels_kmeans))
        print(f'Run {run+1} NMI Score (KMeans): {nmi_kmeans[-1]:.4f}')

        # DBSCAN clustering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        dbscan = DBSCAN(eps=1.4, min_samples=5, metric='euclidean')
        pred_labels_dbscan = dbscan.fit_predict(embeddings)
        results[f'Run_{run+1}_DBSCAN'] = pred_labels_dbscan
        nmi_dbscan.append(normalized_mutual_info_score(labels, pred_labels_dbscan))
        print(f'Run {run+1} NMI Score (DBSCAN): {nmi_dbscan[-1]:.4f}')

        # SOM clustering>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        scaler = MinMaxScaler()
        embeddings_normalized = scaler.fit_transform(embeddings)
        som_x, som_y = 6, 6
        som = MiniSom(som_x, som_y, embeddings.shape[1], sigma=0.5, learning_rate=0.5)
        som.random_weights_init(embeddings_normalized)
        som.train_random(embeddings_normalized, 1000)
        winner_coordinates = np.array([som.winner(vec) for vec in embeddings_normalized])
        unique_clusters = {tuple(coord): i for i, coord in enumerate(np.unique(winner_coordinates, axis=0))}
        pred_labels_som = np.array([unique_clusters[tuple(coord)] for coord in winner_coordinates])
        results[f'Run_{run+1}_SOM'] = pred_labels_som
        nmi_som.append(normalized_mutual_info_score(labels, pred_labels_som))
        print(f'Run {run+1} NMI Score (SOM): {nmi_som[-1]:.4f}')

    # Save 
    nmi_row = pd.DataFrame([['NMI', ''] + nmi_aff + nmi_kmeans + nmi_dbscan + nmi_som],
                           columns=list(results.columns[:2]) + list(results.columns[2:]))
    results = pd.concat([results, nmi_row], ignore_index=True)
    print('mean nmi score (Affinity):', np.mean(nmi_aff))
    print('std nmi score (Affinity):', np.std(nmi_aff))
    print('mean nmi score (KMeans):', np.mean(nmi_kmeans))
    print('std nmi score (KMeans):', np.std(nmi_kmeans))
    print('mean nmi score (DBSCAN):', np.mean(nmi_dbscan))
    print('std nmi score (DBSCAN):', np.std(nmi_dbscan))
    print('mean nmi score (SOM):', np.mean(nmi_som))
    print('std nmi score (SOM):', np.std(nmi_som))
    return results

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df_results = run_clustering(G_roc, labels_array, num_runs=30)
df_results.to_csv('30-30_Roc+s-node2vec_affinity_kmeans_dbscan_som_results.csv', index=False)
print("Results saved to 15-50_Roc+s-node2vec_affinity_kmeans_dbscan_som_results.csv")

# >>>>>>>> Plotting
pos = nx.spring_layout(G_roc, seed=42)
plt.figure(figsize=(7, 7))
community_colors = [labels_roc[node] for node in G_roc.nodes()]
nx.draw_networkx_nodes(G_roc, pos, node_color=community_colors, cmap=plt.cm.tab20, node_size=300)
nx.draw_networkx_edges(G_roc, pos, width=1.0, alpha=0.6)
nx.draw_networkx_labels(G_roc, pos, font_size=9, font_color='black')
sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20, norm=plt.Normalize(vmin=0, vmax=num_cliques - 1))
sm.set_array([])
plt.colorbar(sm, ticks=range(num_cliques), label="Community (including outliers)", ax=plt.gca())
plt.title("ROC with Outliers Assigned to Cliques + Extra Clique-Outlier Edge")
plt.axis('off')
plt.show()
