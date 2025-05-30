import numpy as np
import torch
from sklearn.cluster import KMeans


@torch.inference_mode()
def clarity_score(V):
    # V.shape = (n_neurons) x n_samples x n_features
    V_nrmed = torch.nn.functional.normalize(V, dim=-1)
    clarity = ((V_nrmed.mean(-2).pow(2).sum((-1))) - 1 / V.shape[-2]) / (V.shape[-2] - 1) * V.shape[-2]
    return clarity


@torch.inference_mode()
def clarity_score_with_act(V, act):
    # V.shape = (n_neurons) x n_samples x n_features
    V_nrmed = torch.nn.functional.normalize(V, dim=-1) * act
    clarity = ((V_nrmed.mean(-2).pow(2).sum((-1))) - 1 / V.shape[-2]) / (V.shape[-2] - 1) * V.shape[-2]
    return clarity


@torch.inference_mode()
def redundancy_score(V):
    # V.shape = (n_neurons) x n_samples x n_features
    # compute pairwise similarities across samples and ignore the ones with itself and for each element take the maximal one
    device = V.device
    V_nrmed = torch.nn.functional.normalize(V, dim=-1)
    sims = torch.matmul(V_nrmed, V_nrmed.swapaxes(-1, -2))
    sims = sims - 2 * torch.eye(sims.shape[-1]).to(device)  # remove diagonal
    redundancy = sims.max(-1).values.mean(-1)
    return redundancy


@torch.inference_mode()
def similarity_score(x, y):
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return cos_sim


@torch.inference_mode()
def polysemanticity_score(V, replace_empty_clusters=True, random_state=123, n_clusters=2):
    # V.shape = (n_neurons) x n_samples x n_features
    device = V.device

    n_clusters = n_clusters
    clusters = [KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(e.detach().cpu()) for e in V]
    c_centers = torch.stack([torch.from_numpy(c.cluster_centers_) for c in clusters], dim=0).to(device)

    clarity = clarity_score(c_centers)
    poly = 1 - clarity

    if replace_empty_clusters:
        # retrieve the ones where a cluster has no samples
        counts = [torch.from_numpy(np.unique(c.labels_, return_counts=True)[1]) for c in clusters]
        counts = torch.stack([x if len(x) == n_clusters else torch.zeros(n_clusters) for x in counts], dim=0)
        v_not = V[counts.amin(-1) < 2]
        if v_not.shape[0] > 0:
            clarity_not = 0
            num_samples = min(10, v_not.shape[1])
            for i in range(num_samples):
                clarity_not += clarity_score(torch.stack([v_not.mean(1), v_not[:, i]], dim=1))
            poly[counts.amin(-1) < 2] = 1 - clarity_not.double()/num_samples
    return poly
