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
import pickle

NUM_EPOCHS = 500

BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100
LEARNING_RATE = 0.00005
MOMENTUM = 0.95

def lambda_loss(output, lambdas):
    # assume lambda is a row vector
    return np.dot(lambdas, output)


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
        if self.usePairwise:
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

    def lambda_function(self,labels, scores):
        assert len(labels)==len(scores)
        size= len(labels)
        S_matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)  # (line index, column index)
        lamb_matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)  # (line index, column index)
        lambdas = np.zeros(len(labels), dtype=np.float32)
        # compute Suv matrix using labels
        # current ranking       perfect ranking     S matrix :     a     b     c
        # a (label: 0)          b (label: 1)               a       0     -1    0
        # b (label: 1)          a (label: 0)               b       1     0     0
        # c (label: 0)          c (label: 0)               c       0     0     0
        # Since the matrix is anti-symmetric, we only have to loop over half of it.
        for u in range(size):
            for v in range(u, size):
                if labels[v]>labels[u]:
                    S_matrix[u][v] = -1  # since u<v
                    S_matrix[v][u] = 1  # by anti-symmetry
        # compute lamb u v thanks to the scores
        for u in range(size):
            for v in range(size):
                lamb_matrix[u][v] = 1.0/2.0*(1-S_matrix[u][v]) - 1.0 / (1 + np.exp(scores[u] - scores[v]))
        # aggregate: calculate lambda u with the sum of lambda u v
        for v in range(size):
            for u in range(v+1, size):
                lambdas[u] += lamb_matrix[u][v] - lamb_matrix[v][u]
        # return lambas (aggregated)
        return lambdas


    def compute_lambdas_theano(self,query, labels):
        scores = self.score(query).flatten()
        result = self.lambda_function(labels, scores[:len(labels)])
        return result

    def train_once(self, X_train, query, labels):
        if self.usePairwise:
            lambdas = self.compute_lambdas_theano(query,labels)
            lambdas.resize((BATCH_SIZE, ))

        # had to take the minimum size because there's a label with size of 197
        resize_value = BATCH_SIZE
        if self.usePointwise:
            resize_value = min(resize_value, len(labels))

        # if self.usePairwise:
        #     resize_value = min(resize_value, len(lambdas))
        #     lambdas.resize((resize_value,))


        X_train.resize((resize_value, self.feature_count),refcheck=False)

        if self.usePairwise:
            batch_train_loss = self.iter_funcs['train'](X_train, lambdas)
        if self.usePointwise:
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


# test
def DCG(test_set, r):
    assert r <= len(test_set)
    assert r!= 0
    res=0
    for k in range(1,r+1):#r is the rank so it starts at 1
        rel_r = test_set[k-1]#index of element = rank -1 since rank starts at 1 and index at 0
        res+=float(np.power(2,rel_r)-1)/(np.log2(1+k))
    return res


def nDCG(test_set, r):
    ordered_set= list(test_set)
    ordered_set.sort(reverse=True)
    return DCG(test_set, r) / DCG(ordered_set, r)


def dump_file(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def load_file(filename):
    return pickle.load(open(filename, 'rb'))

epochs_model = {"POINTWISE":[3,5,7,9],"PAIRWISE":[3,5,7,9],"LISTWISE":[3,5,7,9]}
best_epochs_model = {"POINTWISE":0,"PAIRWISE":0,"LISTWISE":0}
FOLD_NUMBER = 5
NUM_FEATURE_VECTOR=64
num_epoch=1
for i in range(1,FOLD_NUMBER+1):
    query_train = q.load_queries('./HP2003/Fold' + str(i) + '/train.txt', NUM_FEATURE_VECTOR)
    lambda_rank = LambdaRankHW(NUM_FEATURE_VECTOR, mode="PAIRWISE")
    lambda_rank.train_with_queries(query_train, num_epoch)
    dump_file(lambda_rank, "model/pairwise" + str(i) + ".model")

# query_train = q.load_queries('./HP2003/Fold' + str(1) + '/train.txt', 64)
# query_valid = q.load_queries('./HP2003/Fold' + str(1) + '/vali.txt', 64)
# lambda_rank = LambdaRankHW(64,mode="POINTWISE")
# val = query_valid.values()
# labels = query_valid.get_labels()
# result_prev = []
# result = []
# lambda_rank.train_with_queries(query_train, 100)
# for elem, label in zip(val, labels):
#     print(nDCG(lambda_rank.score(elem), 10), label[:10])