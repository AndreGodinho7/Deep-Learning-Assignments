import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from vanClasses import Encoder, Decoder, Seq2Seq
from readInput import prepareData

SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 30

def indexesFromSentence(lang, sentence, **kwargs):
    vocabulary = kwargs.get('vocabulary')
    indexes = []
    for word in sentence.split(' '):
        for char in word:
            if vocabulary:
                if char in vocabulary:
                    indexes.append(lang.char2index[char])
                else: 
                    indexes.append(UNK_token)
            else:
                indexes.append(lang.char2index[char])
    return indexes

def tensorFromSentence(lang, sentence, **kwargs):
    vocabulary = kwargs.get('vocabulary')
    if vocabulary:
        indexes = indexesFromSentence(lang, sentence, vocabulary=vocabulary)
    else:
        indexes = indexesFromSentence(lang, sentence)

    indexes.append(EOS_token)
    if lang.name == 'eng':
        indexes.insert(0, SOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorsFromPair(lang, word, **kwargs):
    vocabulary = kwargs.get('vocabulary')
    if vocabulary:
        tensor = tensorFromSentence(lang, word, vocabulary=vocabulary)
    else: 
        tensor = tensorFromSentence(lang, word)
    return tensor

def train_epoch(model, X_train, y_train, optimizer, criterion, clip):
    loss_value = 0

    model.train()

    for src, trg in zip(X_train, y_train):
        optimizer.zero_grad()
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss_value += loss.item()

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss_value

def predict(model, src, trg, teacher_forcing_ratio=0):
    """X (n_examples x n_features)"""
    
    scores = model(src, trg, teacher_forcing_ratio=0)
    predicted_labels = scores.argmax(dim=-1)  
    predicted_labels = predicted_labels[1:].view(-1) # remove tensor with 0s
    return predicted_labels

def evaluate(model, X_dev, y_dev, decision=1):
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for src, trg in zip(X_dev, y_dev):
            pred = predict(model, src, trg, teacher_forcing_ratio=0)
            trg = trg[1:].view(-1) # remove <sos>

            if decision == 0: # # mistakes = # different characters between trg and pred
                #n_correct += (sum([trg[t] == pred[t] for t in range(len(pred))])).item()
                soma = 0
                for t in range(len(pred)):
                    try:
                        soma += trg[t] == pred[t]
                    except:
                        pass
                n_correct += soma.item()
                n_total += len(pred)
            
            else: # at least 1 character different between trg and pred
                try:
                    n_correct += not(False in torch.eq(trg.T,pred.T))
                except RuntimeError: # trg and pred have different lenghts
                    pass
                n_total += 1

    return n_correct / n_total  

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=["perceptron", "crf", "seq2seq"])
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

    np.random.seed(42)

    train_input_lang, train_output_lang, train_pairs = prepareData('train', 'ar', 'eng', reverse=False, reverse_source=True)
    valid_input_lang, valid_output_lang, valid_pairs = prepareData('valid', 'ar', 'eng', reverse=False, reverse_source=True)
    test_input_lang, test_output_lang, test_pairs = prepareData('test', 'ar', 'eng', reverse=False, reverse_source=True)

    X_train = [item[0] for item in train_pairs]
    y_train = [item[1] for item in train_pairs]
    
    X_dev = [item[0] for item in valid_pairs]
    y_dev = [item[1] for item in valid_pairs]
    
    X_test = [item[0] for item in test_pairs]
    y_test = [item[1] for item in test_pairs]

    hidden_size = 50
    encoder = Encoder(train_input_lang.n_chars, hidden_size)
    decoder = Decoder(hidden_size, train_output_lang.n_chars, train_input_lang.biggest_word)

    clf = Seq2Seq(encoder, decoder)
    
    #clf.apply(init_weights)

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

    total = sum(len(src) for src in X_train)
    CLIP = 1

    for epoch in range(1, opt.epochs + 1):
        train_order = np.random.permutation(len(X_train))

        if epoch == 1:
            valid_order = np.random.permutation(len(X_dev))
            test_order = np.random.permutation(len(X_test))

            X_train = [tensorsFromPair(train_input_lang, X_train[i]) for i in train_order]
            y_train = [tensorsFromPair(train_output_lang, y_train[i]) for i in train_order]

            X_dev = [tensorsFromPair(train_input_lang, X_dev[i], vocabulary=train_input_lang.char2index.keys()) for i in valid_order]
            y_dev = [tensorsFromPair(train_output_lang, y_dev[i], vocabulary=train_output_lang.char2index.keys()) for i in valid_order]
        
            X_test = [tensorsFromPair(train_input_lang, X_test[i], vocabulary=train_input_lang.char2index.keys()) for i in test_order]
            y_test = [tensorsFromPair(train_output_lang, y_test[i], vocabulary=train_output_lang.char2index.keys()) for i in test_order]
        
        else:
            X_train = [X_train[i] for i in train_order]
            y_train = [y_train[i] for i in train_order]
            
        epoch_lossvalue = train_epoch(
            clf, X_train, y_train, optimizer, criterion, CLIP)

        acc = evaluate(clf, X_dev, y_dev)
        accuracies.append(acc)
        
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
    
    with open("Test accuracy Seq2Seq.txt","w") as f:
        print('Test accuracy: %f' % test_acc, file = f)


if __name__ == "__main__":
    main()

