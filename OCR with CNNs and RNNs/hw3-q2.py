import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


def read_data(filepath, partitions=None):
    """Read the OCR dataset."""
    labels = {}
    f = open(filepath)
    x_seq = []
    y_seq = []
    X = []
    y = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip('\t\n')
            fields = line.split('\t')
            letter = fields[1]
            last = int(fields[2]) == -1
            if letter in labels:
                k = labels[letter]
            else:
                k = len(labels)
                labels[letter] = k
            partition = int(fields[5])
            if partitions is not None and partition not in partitions:
                continue
            x = np.array([float(v) for v in fields[6:]])
            x_seq.append(x)
            y_seq.append(k)
            if last:
                x_seq = torch.tensor(x_seq, dtype=torch.float32)
                y_seq = torch.tensor(y_seq, dtype=torch.long)

                X.append(x_seq)
                y.append(y_seq)
                x_seq = []
                y_seq = []

    ll = ['' for k in labels]
    for letter in labels:
        ll[labels[letter]] = letter
    return X, y, ll


def pairwise_features(x_i):
    """
    x_i (n_features)
    """
    feat_size = x_i.shape[0]
    ix = np.triu_indices(feat_size)
    return np.array(np.outer(x_i, x_i)[ix])


class BILSTM(nn.Module):
    
    def __init__(self, input_size, output_size, batch_size = 1, n_layers = 1, **kwargs):
        super(BILSTM, self).__init__()

        self.input_size = self.hidden_size = input_size # 128
        self.output_size = output_size # 26
        self.n_layers = n_layers # 1
        self.batch_size = batch_size # 1

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.Tanh()
        )

        self.layer2 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=True)

        self.output_layer = nn.Linear(self.hidden_size * 2, self.output_size)

        self.hidden = self.init_hidden()

    def forward(self, x, **kwargs):
        out = self.layer1(x) 
        
        out = out.view(out.shape[0], self.batch_size, out.shape[1])
        out, lstm_hidden = self.layer2(out, self.hidden)  
        
        out = self.output_layer(out) 
        return out
    
    def init_hidden(self):
        h0 = torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size)

        return (Variable(h0), Variable(c0))

def train_epoch(model, X_train, y_train, optimizer, criterion):
    loss_value = 0

    for xseq, yseq in zip(X_train, y_train):
        optimizer.zero_grad()
        
        outputs = model(xseq)
        outputs = outputs.view(outputs.shape[0], outputs.shape[2]) # [L x 26]
        loss = criterion(outputs, yseq)
        loss_value += loss.item()

        # backprop
        loss.backward()
        optimizer.step()

    return loss_value

def predict(model, X):
    """X (n_examples x n_features)"""
    
    scores = model(X)  
    predicted_labels = scores.argmax(dim=-1)  
    return predicted_labels

def evaluate(model, X, y, decision=0):
    """Evaluate model on data."""
    n_correct = 0
    n_total = 0
    
    for xseq, yseq in zip(X, y):
        model.eval()
    
        yseq_hat = predict(model, xseq)
        if decision == 0: # mistake = # different characters between yseq and yseq_hat
            n_correct += (sum([yseq[t] == yseq_hat[t] for t in range(len(yseq))])).item()
            n_total += len(yseq)
        
        else: # at least 1 character different between yseq and yseq_hat
            n_correct += not(False in torch.eq(yseq.T,yseq_hat.T))
            n_total += 1 
        model.train()

    return n_correct / n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=["perceptron", "crf", "BILSTM"])
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-learning_rate', type=float, default=.001)
    parser.add_argument('-l2_decay', type=float, default=0.)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    parser.add_argument("-no_pairwise", action="store_true",
                        help="""If you pass this flag, the model will use
                        binary pixel features instead of pairwise ones. For
                        submission, you only need to report results with
                        pairwise features, but using binary pixels can be
                        helpful for debugging because models can be trained
                        much faster and with less memory.""")
    opt = parser.parse_args()

    model = opt.model
    l2_decay = opt.l2_decay
    learning_rate = opt.learning_rate

    np.random.seed(42)

    print('Loading data...')
    feature_function = pairwise_features if  opt.no_pairwise else None
    X_train, y_train, labels = read_data(opt.data, partitions=set(range(8)))
    X_dev, y_dev, _ = read_data(opt.data, partitions={8})
    X_test, y_test, _ = read_data(opt.data, partitions={9})

    n_classes = len(labels)
    if feature_function is not None:
        n_features = len(feature_function(X_train[0][0]))
    else:
        n_features = len(X_train[0][0])

    print('Training %s model...' % opt.model)

    clf = BILSTM(n_features, n_classes)
    total = sum(len(xseq) for xseq in X_train)
    
    # get an optimizer
    optims = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        clf.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    accuracies = []
    
    for epoch in range(1, opt.epochs + 1):
        train_order = np.random.permutation(len(X_train))
        X_train = [X_train[i] for i in train_order]
        y_train = [y_train[i] for i in train_order]

        epoch_lossvalue = train_epoch(
            clf, X_train, y_train, optimizer, criterion)
        acc = evaluate(clf, X_dev, y_dev)
        accuracies.append(acc)
        
        if model == 'BILSTM':
            avg_loss = epoch_lossvalue / total
            print('Epoch: %d, Loss: %f, Validation accuracy: %f' % (
                  epoch, avg_loss, acc))
            train_losses.append(avg_loss)

    plt.plot(range(1, opt.epochs + 1), accuracies, 'bo-')
    plt.title('Validation accuracy')
    plt.savefig('Validation accuracy %s.pdf' % model)

    plt.figure()
    plt.plot(range(1, opt.epochs + 1), train_losses, 'bo-')
    plt.title('Train loss')
    plt.savefig('Train loss %s.pdf' % model)
    
    print('Evaluating...')
    test_acc = evaluate(clf, X_test, y_test)
    print('Test accuracy: %f' % test_acc)
    
    with open("Test accuracy BILSTM.txt","w") as f:
        print('Test accuracy: %f' % test_acc, file = f)


if __name__ == "__main__":
    main()
