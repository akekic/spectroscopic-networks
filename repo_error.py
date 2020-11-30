import graph_tool.all as gt
import random
import numpy as np

G = gt.collection.data["celegansneural"]
# Advice from issue #375
G = gt.Graph(G, directed=False, prune=True)
# Find the approximate Block separation of the Network
ground_state_estimation = gt.minimize_nested_blockmodel_dl(G, mcmc_args=dict(niter=1000))
# Take 1% of the non existent edges as potential edges
potential_edges = [(v1, v2) for v1 in G.vertices() for v2 in G.vertices() if random.random() < 0.01 if not ((v1, v2) in G.edges() or v1 <= v2)]
probs = [[] for _ in range(len(potential_edges))]
bs = ground_state_estimation.get_bs()
bs += [np.zeros(1)] * (10 - len(bs))
ground_state = ground_state_estimation.copy(bs=bs, sampling=True)

def collect_edge_probs(s):
    for i in range(len(potential_edges)):
        p = s.get_edges_prob([potential_edges[i]], entropy_args=dict(partition_dl=False))  # Here the error seems to happen
        probs[i].append(p)

    print 'Hello World'

gt.mcmc_equilibrate(ground_state, force_niter=1000, mcmc_args=dict(niter=10), callback=collect_edge_probs)

def get_avg(p):
    p = np.array(p)
    p_avg = np.exp(p).mean()
    return p_avg

probabilities = [(potential_edges[i][0], potential_edges[i][1], get_avg(probs[i])) for i in range(len(potential_edges))]
probabilities = sorted(probabilities, key=lambda probabilities: probabilities[2], reverse=True)
print probabilities
