__author__ = 'agrotov'
import os
os.environ["THEANO_FLAGS"] = "floatX=float32"

import itertools
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
from itertools import count
import query as q
# taken from here https://gist.githubusercontent.com/bwhite/3726239/raw/2c92e90259b01b4a657d20c0ad8390caadd59c8b/rank_metrics.py
import metrics as met
import pickle

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

def lambda_loss(output, lambdas):
    # assume lambda is a row vector
    return np.dot(lambdas, output)

#TEST OF THIS FUNCTION AT THE BOTTOM OF THIS FILE
def create_S(_labels):
    # compute Suv matrix using labels
    # current ranking       perfect ranking     S matrix :     a     b     c
    # a (label: 0)          b (label: 1)               a       0     1    0
    # b (label: 1)          a (label: 0)               b       -1     0     0
    # c (label: 0)          c (label: 0)               c       0     0     0
    # Since the matrix is anti-symmetric, we only have to loop over half of it.
    #
    # Optimization : let assume we have the following labels: [0,1,0,1], the S matrix is then
    # [ 0  1 0 1]
    # [ 0  0 0 0] meaning that if the ligne label is zero, the whole line is zero, otherwise,
    # [ 0 -1 0 1] the line is the labels values with 1 or -1
    # [ 0  0 0 0]
    size = len(_labels)
    S = np.zeros((size, size), dtype=np.float32)  # (line index, column index)
    for u in range(size):
        if (_labels[u] == 0):
            S[u] = np.zeros(size)
        else:
            S[u] = (list(map(lambda x: x * -1, _labels[:u + 1])) + list(_labels[u + 1:]))
            S[u][u] = 0
    return S

#TEST OF THIS FUNCTION AT THE BOTTOM OF THIS FILE
def compute_lambda_matrix (_S, _scores,_labels,_useListwise):
    size=len(_scores)
    lamb_matrix = np.zeros((size, size), dtype=np.float32)  # (line index, column index)
    dcg_max = met.dcg_k(sorted(_labels, reverse=True), len(_labels))

    for u in range(size):
        for v in range(u, size):
            lamb_matrix[u][v]=0.5*(1-_S[u][v]) + -1.0/(1+np.exp(_scores[u]-_scores[v]))
            lamb_matrix[v][u]=0.5*(1-_S[v][u]) + -1.0/(1+np.exp(_scores[v]-_scores[u]))

            if (_useListwise):
                if(_S[u][v]!=0):
                    delta_dcg= met.delta_switch_dcg(_labels[u],_labels[v],u,v)
                    lamb_matrix[u][v] *= delta_dcg/dcg_max
                    lamb_matrix[v][u] *= delta_dcg/dcg_max
    return lamb_matrix

class LambdaRankHW:

    NUM_INSTANCES = count()

    #different modes are "POINTWISE" "PAIRWISE", "LISTWISE"
    def __init__(self, feature_count, mode = "POINTWISE"):
        self.feature_count = feature_count
        self.usePointwise = (mode == "POINTWISE")
        self.usePairwise = (mode == "PAIRWISE")
        self.useListwise = (mode == "LISTWISE")
        self.output_layer = self.build_model(feature_count, 1, BATCH_SIZE)
        self.iter_funcs = self.create_functions(self.output_layer)

    # train_queries are what load_queries returns - implemented in query.py
    def train_with_queries(self, train_queries, num_epochs):
        try:
            for epoch in self.train(train_queries):
                now = time.time()
                if epoch['number'] % 10 == 0:
                    print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                    print("training loss:\t\t{:.6f}\n".format(epoch['train_loss']))
                if epoch['number'] >= num_epochs:
                    break
        except KeyboardInterrupt:
            pass

    def score(self, query):
        feature_vectors = query.get_feature_vectors()
        scores = self.iter_funcs['out'](feature_vectors)
        return scores


    def build_model(self,input_dim, output_dim,
                    batch_size=BATCH_SIZE):
        """Create a symbolic representation of a neural network with `input_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """
        print ("input_dim",input_dim, "output_dim",output_dim)
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_in,
            num_units=200,
            nonlinearity=lasagne.nonlinearities.tanh,
        )


        l_out = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.linear,
        )

        return l_out

    # Create functions to be used by Theano for scoring and training
    def create_functions(self, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM, L1_reg=0.0000005, L2_reg=0.000003):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """
        X_batch = X_tensor_type('x')
        y_batch = T.fvector('y')

        output_row = lasagne.layers.get_output(output_layer, X_batch, dtype="float32")
        output = output_row.T

        output_row_det = lasagne.layers.get_output(output_layer, X_batch,deterministic=True, dtype="float32")
        # Point-wise loss function (squared error)
        if self.usePointwise:
            loss_train = lasagne.objectives.squared_error(output,y_batch)
        # Pairwise loss function
        if self.usePairwise or self.useListwise:
            loss_train = lambda_loss(output,y_batch)
        loss_train = loss_train.mean()

        # TODO: (Optionally) You can add regularization if you want - for those interested
        # L1_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l1)
        # L2_loss = lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
        # loss_train = loss_train.mean() + L1_loss * L1_reg + L2_loss * L2_reg

        # Parameters you want to update
        all_params = lasagne.layers.get_all_params(output_layer)

        # Update parameters, adam is a particular "flavor" of Gradient Descent
        updates = lasagne.updates.adam(loss_train, all_params)


        # Create two functions:

        # (1) Scoring function, deterministic, does not update parameters, outputs scores
        score_func = theano.function(
            [X_batch],output_row_det,
        )

        # (2) Training function, updates the parameters, output loss
        train_func = theano.function(
            [X_batch,y_batch], loss_train,
            updates=updates,
            # givens={
            #     X_batch: dataset['X_train'][batch_slice],
            #     # y_batch: dataset['y_valid'][batch_slice],
            # },
        )

        print ("finished create_iter_functions")
        return dict(
            train=train_func,
            out=score_func,
        )


    def lambda_function(self,labels, scores, useListwise=False):
        assert len(labels)==len(scores)
        size= len(labels)
        lambdas = np.zeros(len(labels), dtype=np.float32)
        S_matrix=create_S(labels)

        # compute lamb u v thanks to the scores
        delta_ndcg=1
        max_len = len(labels)
        #TODO: REZKA CHECK
        lamb_matrix=compute_lambda_matrix(S_matrix,scores,labels,useListwise)
        # aggregate: calculate lambda u with the sum of lambda u v
        for v in range(size):
            for u in range(v+1, size):
                lambdas[u] += lamb_matrix[u][v] - lamb_matrix[v][u]
        # return lambas (aggregated)
        return lambdas


    def compute_lambdas_theano(self,query, labels,useListwise=False):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)],useListwise)
        return result

    def train_once(self, X_train, query, labels):
        if self.usePairwise:
            lambdas = self.compute_lambdas_theano(query,labels)
            lambdas.resize((BATCH_SIZE, ))
        if self.useListwise:
            lambdas = self.compute_lambdas_theano(query, labels,self.useListwise)
            lambdas.resize((BATCH_SIZE,))
        # had to take the minimum size because there's a label with size of 197
        resize_value = BATCH_SIZE
        if self.usePointwise:
            resize_value = min(resize_value, len(labels))

        # if self.usePairwise:
        #     resize_value = min(resize_value, len(lambdas))
        #     lambdas.resize((resize_value,))


        X_train.resize((resize_value, self.feature_count),refcheck=False)

        if self.usePairwise or self.useListwise:
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        if self.usePointwise :
            batch_train_loss = self.iter_funcs['train'](X_train, labels)
        return batch_train_loss


    def train(self, train_queries):
        X_trains = train_queries.get_feature_vectors()

        #change for python 3
        queries = list(train_queries.values())

        for epoch in itertools.count(1):
            batch_train_losses = []
            random_batch = np.arange(len(queries))
            np.random.shuffle(random_batch)
            for index in range(len(queries)):
                s_time = time.time()
                random_index = random_batch[index]
                labels = queries[random_index].get_labels()
                batch_train_loss = self.train_once(X_trains[random_index],queries[random_index],labels)
                batch_train_losses.append(batch_train_loss)
                print(index, time.time()-s_time)
            avg_train_loss = np.mean(batch_train_losses)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
            }


# train and dump model
def dump_file(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def load_file(filename):
    return pickle.load(open(filename, 'rb'))
#,encoding='latin1'

FOLD_NUMBER = 5
NUM_FEATURE_VECTOR=64

def train_model(epoch, mode="POINTWISE",foldNumber=FOLD_NUMBER):
    for i in range(1,foldNumber+1):
        if(i != 2):
           query_train = q.load_queries('./HP2003/Fold' + str(i)  +'/train.txt', NUM_FEATURE_VECTOR)
           lambda_rank = LambdaRankHW(NUM_FEATURE_VECTOR, mode=mode)
           lambda_rank.train_with_queries(query_train, epoch)
           dump_file(lambda_rank, "model/"+mode + str(i) + ".model")

# validating hyperparameter of model
epochs = [300,400,500,600]

def valid_model(epochs):
    #TODO: REZKA CHECK
    tuned_result = {}
    for i in range(1, FOLD_NUMBER + 1):
        tuned_result[i] = []
        query_valid = q.load_queries('./HP2003/Fold' + str(i) + '/vali.txt', NUM_FEATURE_VECTOR)
        val = query_valid.values()
        for epoch in epochs:
            ndcg_valid = []
            lambda_rank = load_file("model/pointwise" + str(i) + "_" + str(epoch) + ".model")
            labels = query_valid.get_labels()
            j = 0
            for elem in val:
                query_score = lambda_rank.score(elem)
                query_labels = np.array(labels[j])
                sort_index = sorted(range(len(query_score)), key=lambda k: query_score[k])
                query_labels = list(query_labels[sort_index])
                ndcg_valid.append(met.ndcg_k(query_labels, 10))
                j += 1
            #sort labels with respect to score

            tuned_result[i].append(np.array(ndcg_valid).mean())
    return tuned_result

def who_wins(tuned_result):
    tuned_model = []
    for i, key in enumerate(tuned_result):
        epoch_win = str(epochs[np.argmax(np.array(tuned_result[key]))])
        print(epoch_win + " is the best epoch for Fold " + str(i+1))
        tuned_model.append(str(i+1) + "_" + epoch_win)
    return tuned_model

def test_model_tuned(tuned_model):
    #TODO: REZKA CHECK
    for i in range(1, FOLD_NUMBER + 1):
        query_test = q.load_queries('./HP2003/Fold' + str(i) + '/test.txt', NUM_FEATURE_VECTOR)
        val = query_test.values()
        lambda_rank = load_file("model/pointwise" + tuned_model[i-1] + ".model")
        labels=query_test.get_labels()
        mean_ndcg_test_set = []
        j = 0
        for elem in val:
            query_score = lambda_rank.score(elem)
            query_labels = np.array(labels[j])
            sort_index = sorted(range(len(query_score)), key=lambda k: query_score[k])
            query_labels = list(query_labels[sort_index])
            mean_ndcg_test_set.append(met.ndcg_k(query_labels, 10))
            j += 1
        print(np.array(mean_ndcg_test_set).mean())

#train_model(600, "POINTWISE")

#tuned_result = valid_model(epochs)
#tuned_model = who_wins(tuned_result)
#print(tuned_model)

# best model according to hyperparameter tuning
#tuned_model = ["1_300", "2_600", "3_500", "4_400", "5_600"]
#test_model(tuned_model)


def test_model(mode,tuned_value=None):
    #TODO:REKA CHECK
    model_name  = mode
    tuned_name = "" if tuned_value==None else "_"+str(tuned_value)
    if(mode == "POINTWISE"):
        model_name="pointwise"
    for i in range(1, FOLD_NUMBER + 1):
        query_test = q.load_queries('./HP2003/Fold' + str(i) + '/test.txt', 64)
        val = query_test.values()
        lambda_rank = load_file("model/"+model_name+ str(i)+tuned_name + ".model")
        labels=query_test.get_labels()
        mean_ndcg_test_set = []
        j=0
        for elem in val:
            #Strange step: if take abs value, it works: see report
            query_score = np.abs(lambda_rank.score(elem))
            query_labels= np.array(labels[j])
            sort_index = sorted(range(len(query_score)), reverse=True,key=lambda k: query_score[k])
            query_labels = list(query_labels[sort_index])
            mean_ndcg_test_set.append(met.ndcg_k(query_labels, 10))
            j+=1
        print(np.array(mean_ndcg_test_set).mean())

#train_model(5, "LISTWISE",foldNumber=5)

#query_train = q.load_queries('./HP2003/Fold2/train.txt', NUM_FEATURE_VECTOR)
#lambda_rank = LambdaRankHW(NUM_FEATURE_VECTOR, mode='LISTWISE')
#lambda_rank.train_with_queries(query_train, 5)
#dump_file(lambda_rank, "model/listwise_test.model")

#tuned_result = valid_model(epochs)
#tuned_model = who_wins(tuned_result)
# print(tuned_model)
#
#test_model_tuned(tuned_model)
test_model("LISTWISE")
#test_model("POINTWISE")

#TEST THE IMPLEMENTED FUNCTION
# test_S=create_S([0,1,1])
# print(test_S)
# test_scores=[1,2,3]
# labels=[1,0,0]
# print(test_scores)
# print(compute_lambda_matrix(test_S,test_scores))
# print(compute_lambda_matrix(test_S,test_scores,labels,_useListwise=False))
