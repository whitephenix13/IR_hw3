import numpy as np

def dcg_k(rel,k):
    _rel=list(rel)
    _rel = np.asfarray(_rel)[:k]
    if _rel.size:
            return np.sum((np.power(2,_rel)-1) / np.log2(np.arange(2, _rel.size + 2)))
    return 0.
def ndcg_k(rel,k):
    _rel=list(rel)
    dcg_max = dcg_k(sorted(_rel, reverse=True), k)
    print(dcg_k(_rel, k),dcg_max)
    if not dcg_max:
        return 0.
    return dcg_k(_rel, k) / dcg_max