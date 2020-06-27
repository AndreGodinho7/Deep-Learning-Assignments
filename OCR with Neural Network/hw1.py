#!/usr/bin/env python

# Deep Structured Learning Homework 1
# Fall 2019

import argparse
from itertools import count
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations 

def load_data(path, bias=False, dev_fold=8, test_fold=9):
    """
    path: location of OCR data
    feature_rep: a function or None. Use it to transform the binary pixel
        representation into something more interesting in Q 2.2a
    bias: whether to add a bias term as an extra feature dimension
    """
    label_counter = count()
    labels = defaultdict(lambda: next(label_counter))
    X = []
    y = []
    fold = []
    with open(path) as f:
        for line in f:
            tokens = line.split()
            pixels = [int(t) for t in tokens[6:]]
            letter = labels[tokens[1]]
            fold.append(int(tokens[5]))
            X.append(pixels)
            y.append(letter)
    X = np.array(X, dtype='int8')
    y = np.array(y, dtype='int8')

    if bias:
        bias_vector = np.ones((X.shape[0], 1), dtype=int)
        X = np.hstack((X, bias_vector))

    fold = np.array(fold, dtype='int8')
    # boolean masks, not indices
    train_ix = (fold != dev_fold) & (fold != test_fold)
    dev_ix = fold == dev_fold
    test_ix = fold == test_fold

    train_X, train_y = X[train_ix], y[train_ix]
    dev_X, dev_y = X[dev_ix], y[dev_ix]
    test_X, test_y = X[test_ix], y[test_ix]

    return {"train": (train_X, train_y),
            "dev": (dev_X, dev_y),
            "test": (test_X, test_y)}

def custom_features(x_i):
    """
    x_i (n_features, )
    returns (???, ): It's up to you to define an interesting feature
        representation. One idea: pairwise pixel features (see the handout).
    """
    # Q2.2 a
    x_i = x_i.reshape((1,128)) # re-shape as an array of [1x128]
    matrix = x_i.T*x_i # [128x128]
    
    # calculate upper triangular matrix (w/o diagonal)
    x_i = matrix[np.triu_indices(len(matrix),k=0)]
    return x_i # (8256,)

class LinearModel(object):
    def __init__(self, n_classes, n_features, feature_function=None, **kwargs):
        self.W = np.zeros((n_classes, n_features))
        self.feature_function = feature_function

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            if self.feature_function:
                x_i = self.feature_function(x_i)
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        predicted_labels = []
        for x_i in X:
            if self.feature_function:
                x_i = self.feature_function(x_i)
            scores = np.dot(self.W, x_i)  # (n_classes)
            predicted_label = scores.argmax(axis=0)
            predicted_labels.append(predicted_label)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Question 2.1 b
        score = np.dot(self.W, x_i.T)
        predicted_label = score.argmax(axis = 0)
        
        if predicted_label != y_i:
            self.W[predicted_label] = self.W[predicted_label] - x_i
            self.W[y_i] = self.W[y_i] + x_i  
        return

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        l2_penalty (float): BONUS
        """
        # Question 2.2 b
        score = np.dot(self.W, x_i.T)
        predicted_label = score.argmax(axis = 0)

        if predicted_label != y_i:    
            exp = np.exp([score])
            zx = np.sum(exp)

            expectation = (exp/zx).T*(x_i.reshape((1,len(x_i))))
            if l2_penalty != 0:
                self.W = self.W - learning_rate*(expectation + l2_penalty*self.W)
            else:
                self.W = self.W - learning_rate*expectation

            self.W[y_i] = self.W[y_i] + learning_rate*x_i
        return 
        
class MLP(object):
    # Q3. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        # softmax layer must have the same number of nodes of the output layer
        # # hidden units = # classes
        self.W1 = np.random.rand(n_features, n_classes)
        self.b1 = np.zeros((1, n_classes))
        self.W2 = np.random.rand(n_classes,n_classes)
        self.b2 = np.zeros((1,n_classes))
        
    def linear_activation(self,x):
        return x

    def softmax(self,x):    
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def forward_propagation(self, X):
        z1 = np.dot(X,self.W1) + self.b1 # [n_samples x 26]
        self.a1 = self.linear_activation(z1) # [n_samples x 26]
        z2 = np.dot(self.a1, self.W2) + self.b2 # [n_samples x 26]
        output = self.softmax(z2) # [n_samples x 26]
        return output
    
    def predict(self, input):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        output = self.forward_propagation(input)
        output = np.argmax(output,axis = 1)
        return output
        
    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def linearact_derivative(self,x):
        return np.ones(x.shape)

    def softmax_derivative(self, output, y):
        n_samples = y.shape[0]
        y_onehot = np.zeros((len(y),output.shape[1]))
        y_onehot[np.arange(n_samples),y] = 1
        res = output - y_onehot
        return res/n_samples

    def back_propagation(self, X,y,output, learning_rate):
        dL_dz2 = self.softmax_derivative(output,y) # [n_samples x 26]
        dz2_dw2 = self.a1 # [n_samples x 26]
        dL_dw2 = np.dot(dz2_dw2.T,dL_dz2) # [26x26]

        dL_da1 = np.dot(dL_dz2,self.W2.T) # [n_samples x 26]
        da1_dz1 = self.linearact_derivative(self.a1) # [n_samples x 26]
        dL_dz1 = dL_da1 * da1_dz1 # [n_samples x 26]
        dL_dw1 = np.dot(X.T, dL_dz1) # [128 x 26]

        self.W2 -= learning_rate * dL_dw2
        self.b2 -= learning_rate * np.sum(dL_dz2, axis=0)
        self.W1 -= learning_rate * dL_dw1
        self.b1 -= learning_rate * np.sum(dL_dz1, axis=0)

    def train_epoch(self, X, y, learning_rate = 0.5):
        pred = self.forward_propagation(X)
        self.back_propagation(X, y, pred, learning_rate)
        
def plot(epochs, valid_accs, test_accs,train_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.plot(epochs, train_accs, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument("-custom_features", action="store_true",
                        help="""Whether to use the custom_features() function
                        to postprocess the binary pixel features, as in Q2.2,
                        part (a).""")
    parser.add_argument('-bias', action='store_true',
                        help="""Whether to add an extra bias feature to all
                        samples in the dataset. In an MLP, where there can be
                        biases for each neuron, adding a bias feature to the
                        input is not sufficient.""")
    opt = parser.parse_args()

    feature_function = custom_features if opt.custom_features else None
    data = load_data(opt.data, bias=opt.bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 26
    if opt.custom_features:
        n_feats = feature_function(train_X[0]).shape[0]
    else:
        n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats, feature_function)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats, feature_function)
    else:
        # Q3. Be sure to experiment with different values for hidden_size.
        hidden_size = 2  # tune me!
        model = MLP(n_classes, n_feats, hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    train_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(train_X, train_y)
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        train_accs.append(model.evaluate(train_X, train_y))

    print("Train set accuracy after "+ str(epochs[-1]) +" epochs: "+str(round(train_accs[-1],3)))
    print("Validation set accuracy after "+ str(epochs[-1]) +" epochs: "+str(round(valid_accs[-1],3)))
    print("Test set accuracy after "+ str(epochs[-1]) +" epochs: "+str(round(test_accs[-1],3)))

    # plot
    plot(epochs, valid_accs, test_accs,train_accs)


if __name__ == '__main__':
    main()