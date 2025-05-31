#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A graph-based geodesic nearest centroid classifier

Created 10/05/2025

Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import networkx as nx
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
import scipy.sparse._csr
from numpy import dot
from numpy import trace
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import Isomap
from xgboost import XGBClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Train the GNCC classifier
def train_geo_NCC(treino, target_treino):
    m = treino.shape[1]
    c = len(np.unique(target_treino))    
    # Matrix to store the c centroids
    centros = np.zeros((c, m))
    # Compute the centroids
    for i in range(c):
        indices = np.where(target_treino==i)[0]
        amostras = treino[indices, :]
        centros[i, :] = amostras.mean(axis=0)
    return centros
    

# Test the GNCC classifier
def test_geo_NCC(teste, target_teste, nn, centros):
    n = teste.shape[0]
    k = centros.shape[0]
    m = teste.shape[1]
    # Labels for the test set samples
    rotulos = np.zeros(teste.shape[0]) 
    # Append the c centroids in the test data
    data = np.vstack((teste, centros))
    # Build a graph
    CompleteGraph = sknn.kneighbors_graph(data, n_neighbors=n-1, mode='distance')
    # Adjacency matrix
    W_K = CompleteGraph.toarray()
    # NetworkX format
    K_n = nx.from_numpy_array(W_K)
    # MST
    W_mst = nx.minimum_spanning_tree(K_n)
    mst = [(u, v, d) for (u, v, d) in W_mst.edges(data=True)]
    mst_edges = []
    for edge in mst:
        edge_tuple = (edge[0], edge[1], edge[2]['weight'])
        mst_edges.append(edge_tuple)
    # Create the k-NNG
    knnGraph = sknn.kneighbors_graph(data, n_neighbors=nn, mode='distance')
    # Adjacency matrix
    W = knnGraph.toarray()
    # NetworkX format
    G = nx.from_numpy_array(W)
    # To assure the k-NNG is connected we add te MST edges
    G.add_weighted_edges_from(mst_edges)
    # Array for geodesic distances
    distancias = np.zeros((k, n+k))
    coord_centros = list(range(n, n+k))
    for j in range(k):
        # Dijkstra's algorithm for geodesic distances
        length, path = nx.single_source_dijkstra(G, coord_centros[j])
        # Sort vertices
        dists = list(dict(sorted(length.items())).values()) 
        distancias[j, :] = dists
    # Labels
    rotulos = np.zeros(n)    
    # Assign labels to data points
    for j in range(n):
        rotulos[j] = distancias[:, j].argmin()
    # Return labels
    return rotulos


#%%%%%%%%%%%%%%%%%%%%  Data loading
# GNCC x NCC
X = skdata.fetch_openml(name='AP_Lung_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Kidney', version=1)       
#X = skdata.fetch_openml(name='AP_Ovary_Lung', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Ovary_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Ovary', version=1)   
#X = skdata.fetch_openml(name='AP_Breast_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Endometrium_Breast', version=1)
#X = skdata.fetch_openml(name='AP_Endometrium_Lung', version=1)
#X = skdata.fetch_openml(name='AP_Endometrium_Colon', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Prostate_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Uterus', version=1)
#X = skdata.fetch_openml(name='micro-mass', version=1)
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='mfeat-zernike', version=1)
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='optdigits', version=1)
#X = skdata.fetch_openml(name='satimage', version=1)
#X = skdata.fetch_openml(name='stock', version=2)
#X = skdata.fetch_openml(name='analcatdata_authorship', version=2)
#X = skdata.fetch_openml(name='texture', version=1)
#X = skdata.fetch_openml(name='segment', version=1)

# GNCC x KNN
#X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)    
#X = skdata.fetch_openml(name='Olivetti_Faces', version=1)    
#X = skdata.fetch_openml(name='oh5.wc', version=1)       
#X = skdata.fetch_openml(name='leukemia', version=1)    
#X = skdata.fetch_openml(name='AP_Colon_Ovary', version=1)     
#X = skdata.fetch_openml(name='BurkittLymphoma', version=1)    
#X = skdata.fetch_openml(name='dbworld-bodies', version=1)     
#X = skdata.fetch_openml(name='variousCancers_final', version=1)     
#X = skdata.fetch_openml(name='CIFAR_10_small', version=1)     
#X = skdata.fetch_openml(name='rabe_131', version=2)          
#X = skdata.fetch_openml(name='kidney', version=2)            
#X = skdata.fetch_openml(name='breast-tissue', version=2)     
#X = skdata.fetch_openml(name='baskball', version=2)          
#X = skdata.fetch_openml(name='vineyard', version=2)          
#X = skdata.fetch_openml(name='ar4', version=1)               
#X = skdata.fetch_openml(name='aids', version=1)              

dados = X['data']
target = X['target']  

# To deal with sparse matrix data
if type(dados) == scipy.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dados = dados.to_numpy()
        target = target.to_numpy()

# Remove nan's
dados = np.nan_to_num(dados)

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))
#nn = round(np.sqrt(n))      # number of nearest neighbors
nn = round(np.log(n))      # number of nearest neighbors

# Transforma rótulos para inteiros
rotulos = list(np.unique(target))
numbers = np.zeros(n)
for i in range(n):
    numbers[i] = rotulos.index(target[i])

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)
print()

inicio = 0.1
fim = 0.5
tamanho = 30
sample_list = np.linspace(inicio, fim, tamanho)
acc_svm = []
acc_ncc = []
acc_knn = []
acc_topncc = []
acc_xgb = []
prec_svm = []
prec_ncc = []
prec_knn = []
prec_topncc = []
prec_xgb = []
rec_svm = []
rec_ncc = []
rec_knn = []
rec_topncc = []
rec_xgb = []
f1_svm = []
f1_ncc = []
f1_knn = []
f1_topncc = []
f1_xgb = []


inicio = time.time()

for size in sample_list:
    # Divide conjunto em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(dados, numbers, train_size=size, random_state=42)

    # Apply regular NCC
    ncc = NearestCentroid()
    ncc.fit(X_train, y_train) 
    y_pred = ncc.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    acc_ncc.append(acc)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    prec_ncc.append(prec)
    rec_ncc.append(rec)
    f1_ncc.append(f1)
    print('Accuracy NCC: ', acc)
    print('Precision NCC: ', prec)
    print('Recall NCC: ', rec)
    print('F1 NCC: ', rec)
    print()

    # Apply SVM
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    y_pred = svm.predict(X_test)    
    acc = balanced_accuracy_score(y_test, y_pred)
    acc_svm.append(acc)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    prec_svm.append(prec)
    rec_svm.append(rec)
    f1_svm.append(f1)
    print('Accuracy SVM: ', acc)
    print('Precision SVM: ', prec)
    print('Recall SVM: ', rec)
    print('F1 SVM: ', rec)
    print()

    # Apply k-NN
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train) 
    y_pred = knn.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    acc_knn.append(acc)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    prec_knn.append(prec)
    rec_knn.append(rec)
    f1_knn.append(f1)
    print('Accuracy KNN: ', acc)
    print('Precision KNN: ', prec)
    print('Recall KNN: ', rec)
    print('F1 KNN: ', rec)
    print()

    # Apply XGBoost
    bst = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.5, objective='binary:logistic')
    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    acc_xgb.append(acc)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    prec_xgb.append(prec)
    rec_xgb.append(rec)
    f1_xgb.append(f1)
    print('Accuracy XGB: ', acc)
    print('Precision XGB: ', prec)
    print('Recall XGB: ', rec)
    print('F1 XGB: ', rec)
    print()
    
    # Apply GNCC
    centros = train_geo_NCC(X_train, y_train)
    rotulos = test_geo_NCC(X_test, y_test, nn, centros)
    acc = balanced_accuracy_score(y_test, rotulos)
    prec = precision_score(y_test, rotulos, average='macro')
    rec = recall_score(y_test, rotulos, average='macro')
    f1 = f1_score(y_test, rotulos, average='macro')
    acc_topncc.append(acc)
    prec_topncc.append(prec)
    rec_topncc.append(rec)
    f1_topncc.append(f1)
    print('Accuracy GNCC: ', acc)
    print('Precision GNCC: ', prec)
    print('Recall GNCC: ', rec)
    print('F1 GNCC: ', rec)
    print()
    print('-------------------------------------')
    

fim = time.time()
print()
print('Elapsed time: %f s' %(fim-inicio))
print()

print('Média ACC (NCC): ', sum(acc_ncc)/len(acc_ncc))
#print('Média ACC (SVM): ', sum(acc_svm)/len(acc_svm))
#print('Média ACC (KNN): ', sum(acc_knn)/len(acc_knn))
#print('Média ACC (XGB): ', sum(acc_xgb)/len(acc_xgb))
print('Média ACC (GEO-NCC): ', sum(acc_topncc)/len(acc_topncc))
print()
print('Média Precision (NCC): ', sum(prec_ncc)/len(prec_ncc))
#print('Média F1 (SVM): ', sum(f1_svm)/len(f1_svm))
#print('Média F1 (KNN): ', sum(f1_knn)/len(f1_knn))
#print('Média F1 (XGB): ', sum(f1_xgb)/len(f1_xgb))
print('Média Precision (GEO-NCC): ', sum(prec_topncc)/len(prec_topncc))
print()
print('Média Recall (NCC): ', sum(rec_ncc)/len(rec_ncc))
#print('Média F1 (SVM): ', sum(f1_svm)/len(f1_svm))
#print('Média F1 (KNN): ', sum(f1_knn)/len(f1_knn))
#print('Média F1 (XGB): ', sum(f1_xgb)/len(f1_xgb))
print('Média Recall (GEO-NCC): ', sum(rec_topncc)/len(rec_topncc))
print()
print('Média F1 (NCC): ', sum(f1_ncc)/len(f1_ncc))
#print('Média F1 (SVM): ', sum(f1_svm)/len(f1_svm))
#print('Média F1 (KNN): ', sum(f1_knn)/len(f1_knn))
#print('Média F1 (XGB): ', sum(f1_xgb)/len(f1_xgb))
print('Média F1 (GEO-NCC): ', sum(f1_topncc)/len(f1_topncc))
print()


# Plot graphics
plt.figure(1)
plt.plot(sample_list, acc_ncc, color='red', label='NCC')
plt.plot(sample_list, acc_topncc, color='blue', label='GEO-NCC')
#plt.plot(sample_list, acc_svm, color='green', label='SVM')
#plt.plot(sample_list, acc_knn, color='black', label='KNN')
plt.legend()
plt.xlabel('Training set size (%)')
plt.ylabel('Accuracies')
plt.savefig('Accuracies.png')
plt.show()
plt.close()

