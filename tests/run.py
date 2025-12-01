import numpy as np
from itertools import product
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari


from graphs.lmgec import (
    LMGEC,
    clustering_accuracy,
    clustering_f1_score,
    preprocess_dataset
)


from graphs.lmgec.datasets import datagen

# Load any dataset 
dataset = "acm"
beta = 2.0
runs = 3
tempertaure = 10.0
max_iter = 10
As, Xs, labels = datagen(dataset)
k = len(np.unique(labels))
views = list(product(As, Xs))

for v in range(len(views)):
    A, X = views[v]
    tf_idf = dataset in ["acm", "dblp", "imdb", "photos"]
    norm_adj, features = preprocess_dataset(A,X, tf_idf=tf_idf, beta=beta)

    if type(features) != np.ndarray:
        features = features.toarray()
    if type(norm_adj) == np.matrix:
        norm_adj = np.asarray(norm_adj)
    
    views[v] = (norm_adj, features)

metrics = {}
metrics['acc'] = []
metrics['nmi'] = []
metrics['ari'] = []
metrics['f1'] = []
metrics['loss'] = []
metrics['time'] = []

for run in range(runs):
    t0 = time()
    Hs = []

    for v, (S,X) in enumerate(views):
        features = S @ X
        x = features
        x = StandardScaler(with_std=False).fit_transform(x)
        Hs.append(x)
    
    Z, F, XW_consensus, losses = LMGEC(k, k+1, tempertaure, max_iter).fit(Hs)


    metrics['time'].append(time()-t0)
    metrics['acc'].append(clustering_accuracy(labels, Z))
    metrics['nmi'].append(nmi(labels, Z))
    metrics['ari'].append(ari(labels, Z))
    metrics['f1'].append(clustering_f1_score(labels, Z, average='macro'))
    metrics['loss'].append(losses[-1])

results = {
    'mean' : {k:np.mean(v).round(4) for k,v in metrics.items()},
    'std' : {k:np.std(v).round(4) for k,v in metrics.items()}
}

means = results['mean']
acc_mean = means['acc']
f1_mean = means['f1']
nmi_mean = means['nmi']
ari_mean = means['ari']

print(f'dataset : {dataset} \nAccuracy : {acc_mean}\nf1 : {f1_mean}\nnmi : {nmi_mean}\nari : {ari_mean}')

