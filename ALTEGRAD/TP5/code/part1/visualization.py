"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
from deepwalk import deepwalk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse.linalg import eigs


# Loads the web graph
G = nx.read_weighted_edgelist('../data/web_sample.edgelist', delimiter=' ', create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


############## Task 3
# Extracts a set of random walks from the web graph and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20

model = deepwalk(G, n_walks, walk_length, n_dim)

############## Task 4
# Visualizes the representations of the 100 nodes that appear most frequently in the generated walks
def visualize(model, n, dim):

    nodes = model.wv.index_to_key[:n]
    DeepWalk_embeddings = np.empty(shape=(n, dim))
    for i in range(n):
        DeepWalk_embeddings[i] = model.wv[nodes[i]]
    
    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2)

    vecs_pca = my_pca.fit_transform(DeepWalk_embeddings)
    vecs_tsne = my_tsne.fit_transform(vecs_pca)

    fig, ax = plt.subplots()
    ax.scatter(vecs_tsne[:,0], vecs_tsne[:,1],s=3)
    for x, y, node in zip(vecs_tsne[:,0] , vecs_tsne[:,1], nodes):     
        ax.annotate(node, xy=(x, y), size=8)
    fig.suptitle('t-SNE visualization of node embeddings',fontsize=30)
    fig.set_size_inches(20,15)
    plt.savefig('embeddings.pdf')  
    plt.show()


############## Task 5
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', create_using=nx.Graph())
labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
idx_to_label = {}
for i in range(labels.shape[0]):
    idx_to_label[labels[i, 0]] = labels[i, 1]
labels = [idx_to_label[int(node)] for node in G.nodes()]
labels = np.array(labels)

# visualize the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw_networkx(G, with_labels=True, node_color=labels, cmap=plt.cm.Set1, pos=pos)

############### Task 6
# Extracts a set of random walks from the web graph and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20

model = deepwalk(G, n_walks, walk_length, n_dim)

n = G.number_of_nodes()

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

############### Task 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

############### Task 8
laplacian = nx.normalized_laplacian_matrix(G).astype(np.float32)
from scipy.sparse.linalg import eigs

_, U = eigs(laplacian, k=2, which='SR')
U = U.real

X_train, X_test, y_train, y_test = train_test_split(U, labels, test_size=test_size, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

