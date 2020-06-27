#!/usr/bin/env python

import argparse
from itertools import count
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns 

class OCRDataset(Dataset):
    """Binary OCR dataset."""

    def __init__(self, path, dev_fold=8, test_fold=9):
        """
        path: location of OCR data
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

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        fold = torch.tensor(fold, dtype=torch.long)
        # boolean masks, not indices
        train_idx = (fold != dev_fold) & (fold != test_fold)
        dev_idx = fold == dev_fold
        test_idx = fold == test_fold

        self.X = X[train_idx]
        self.y = y[train_idx]

        self.dev_X = X[dev_idx]
        self.dev_y = y[dev_idx]

        self.test_X = X[test_idx]
        self.test_y = y[test_idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN(nn.Module):
    
    def __init__(self, n_classes, **kwargs):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            # in_ch = 1 (greyscale) ; out_ch = 16 ; kernel size = 5 = F ; padding = (F-1)/2 = 2 ; stride = 1
            nn.Conv2d(1 , 16, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            # in_ch = 16 (previous layer) ; out_ch = 32 ; kernel size = 7 = F ; padding = (F-1)/2 = 3 ; stride = 1
            nn.Conv2d(16, 32, kernel_size = 7, stride = 1, padding = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.output_layer = nn.Linear(4 * 2 * 32, n_classes)

    def forward(self, x, **kwargs):
        x = x.view(x.shape[0],1,16,8) # [64, 1, 16, 8]
        out = self.layer1(x) # [64, 16, 8, 4]
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out) # [64, 32 , 4, 2]
        out = out.reshape(out.size(0), -1) # [64, 256]
        out = self.output_layer(out) # [64, 26]
        return out

class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)
        """
        super(LogisticRegression, self).__init__()
        # Implement me!

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        raise NotImplementedError


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_size (int)
        layers (int)
        activation_type (str)
        dropout (float): dropout probability
        """
        super(FeedforwardNetwork, self).__init__()
        # Implement me!

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        raise NotImplementedError


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """

    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    loss_value = loss.item()

    # backprop
    loss.backward()
    optimizer.step()

    return loss_value

def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')
    

def plot_kernels(tensor, name, heatmap=0, greyscale=0, num_cols=8):
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    for i in range(num_kernels):
        if heatmap:
            sns.set()
            if i == 0:
                cbar_ax = fig.add_axes([.91, .3, .03, .4])
                
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)

            sns.heatmap(tensor[i][0,:,:], ax=ax1,
                        cbar=i == 0,
                        xticklabels=0, yticklabels=0,
                        cbar_ax=None if i else cbar_ax)
        else:
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            if greyscale == 1:
                ax1.imshow(tensor[i][0,:,:], cmap='gray')
            else:
                ax1.imshow(tensor[i][0,:,:])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    if heatmap:    
        fig.tight_layout(rect=[0, 0, .9, 1])
        ax1.figure.savefig('%s heatmap.pdf' %(name), bbox_inches ='tight')
    else:
        plt.savefig('%s plot.pdf' %(name), bbox_inches ='tight')

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('model',
                            choices=['logistic_regression', 'mlp', 'cnn'],
                            help="Which model should the script run?")
        parser.add_argument('-data', default='letter.data',
                            help="Path to letter.data OCR corpus.")
        parser.add_argument('-epochs', default=20, type=int,
                            help="""Number of epochs to train for. You should not
                            need to change this value for your plots.""")
        parser.add_argument('-batch_size', default=64, type=int,
                            help="Size of training batch.")
        parser.add_argument('-learning_rate', type=float, default=0.001)
        parser.add_argument('-l2_decay', type=float, default=0)
        parser.add_argument('-hidden_sizes', type=int, default=200)
        parser.add_argument('-layers', type=int, default=1)
        parser.add_argument('-dropout', type=float, default=0.3)
        parser.add_argument('-activation',
                            choices=['tanh', 'relu'], default='relu')
        parser.add_argument('-optimizer',
                            choices=['sgd', 'adam'], default='adam')
        opt = parser.parse_args()

        dataset = OCRDataset(opt.data)
        train_dataloader = DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True)

        dev_X, dev_y = dataset.dev_X, dataset.dev_y
        test_X, test_y = dataset.test_X, dataset.test_y

        n_classes = torch.unique(dataset.y).shape[0]  # 26
        n_feats = dataset.X.shape[1]  # 128

        # initialize the model
        if opt.model == 'logistic_regression':
            model = LogisticRegression(n_classes, n_feats)
        elif opt.model == 'mlp':
            model = FeedforwardNetwork(
                n_classes, n_feats,
                opt.hidden_size, opt.layers,
                opt.activation, opt.dropout)
        else:
            model = CNN(n_classes)

        # get an optimizer
        optims = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD}

        optim_cls = optims[opt.optimizer]
        optimizer = optim_cls(
            model.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.l2_decay)

        # get a loss criterion
        criterion = nn.CrossEntropyLoss()

        # training loop
        epochs = torch.arange(1, opt.epochs + 1)
        train_mean_losses = []
        valid_accs = []
        train_losses = []
        for ii in epochs:
            print('Training epoch {}'.format(ii))
            for i, (X_batch, y_batch) in enumerate(train_dataloader):
                loss = train_batch(
                    X_batch, y_batch, model, optimizer, criterion)
                train_losses.append(loss)

            mean_loss = torch.tensor(train_losses).mean().item()
            print('Training loss: %.4f' % (mean_loss))

            train_mean_losses.append(mean_loss)
            valid_accs.append(evaluate(model, dev_X, dev_y))
            print('Valid acc: %.4f' % (valid_accs[-1]))

        print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
        # plot
        plot(epochs, train_mean_losses, ylabel='Loss', name='training-loss')
        plot(epochs, valid_accs, ylabel='Accuracy', name='validation-accuracy')

        #plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        #filters = model.modules()
        model_layers = [i for i in model.children()] 
        first_layer = model_layers[0] # sequential first layer
        second_layer = model_layers[1] # sequential second layer
        
        first_kernels = first_layer[0].weight.data.numpy() # cnn first layer weights (tensor values)
        plot_kernels(first_kernels, 'first layer grey', greyscale = 1)
        
        second_kernels = second_layer[0].weight.data.numpy() # cnn second layer weights (tensor values)
        plot_kernels(second_kernels, 'second layer grey', greyscale = 1)
        
        plot_kernels(first_kernels,'first layer', heatmap = 1)
        plot_kernels(second_kernels,'second layer', heatmap = 1)

if __name__ == '__main__':
    main()
