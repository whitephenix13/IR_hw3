import query as q

query = q.load_queries('./HP2003/Fold1/test.txt', 64)
qids = query.get_qids()
feature_vectors = query.get_feature_vectors()
labels = query.get_labels()
qval = query.values()
print(qval[10])