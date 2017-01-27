import query as q

query = q.load_queries('./HP2003/Fold1/train.txt', 64)
qids = query.get_qids()
feature_vectors = query.get_feature_vectors()
labels = query.get_labels()
qval = query.values()

query_train = []
epochs_model = {"POINTWISE":[3,5,7,9],"PAIRWISE":[3,5,7,9],"LISTWISE":[3,5,7,9]}
best_epochs_model = {"POINTWISE":0,"PAIRWISE":0,"LISTWISE":0}
FOLD_NUMBER = 5
for i in range(1,FOLD_NUMBER+1):
    query_train = q.load_queries('./HP2003/Fold' + str(i) + '/train.txt')
    query_valid = q.load_queries('./HP2003/Fold' + str(i) + '/vali.txt')
    query_test =  q.load_queries('./HP2003/Fold' + str(i) + '/test.txt')
