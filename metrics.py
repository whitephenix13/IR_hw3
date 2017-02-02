import numpy as np

def dcg_k(rel,k):
    _rel=list(rel)
    _rel = np.asfarray(_rel)[:k]
    if _rel.size:
            return np.sum(_rel / np.log2(np.arange(2, _rel.size + 2)))
    return 0.
def ndcg_k(rel,k):
    _rel=list(rel)
    dcg_max = dcg_k(sorted(_rel, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_k(_rel, k) / dcg_max

def delta_switch_ndcg(rel1,rel2,pos1,pos2):
    #calculate the difference induced in ndcg by making the change:
    #(pos1,rel1) => (pos1,rel2)
    #(pos2,rel2) => (pos2,rel1)
    #NOTE THAT rank = pos+1
    return(float(rel2-rel1)/ np.log2(pos1+2) + float(rel1-rel2)/ np.log2(pos2+2))