import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random

SOS_token = 0
EOS_token = 1

class BidirectEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size) # embbeding size = hidden size
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, bidirectional=True)

        self.hidden = self.init_hidden()

    def forward(self, src):

        # src = [src len, batch size]
        embedded = self.embedding(src) # [src len, batch size, # embbeding size = hidden size]
        
        output, hidden = self.lstm(embedded)
        
        #outputs = [src len, batch size, hid dim = emb dim * n directions]
        #hidden[0] = [n layers * n directions, batch size, hid dim]
        #hidden[1] = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        return output, hidden

    def init_hidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)

        return (Variable(h0), Variable(c0))

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, biggest_word):
        super().__init__()
        self.hidden_size = hidden_size # embbeding size = hidden size
        self.output_size = output_size
        self.max_length = biggest_word

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        #input = [batch size]
        #hidden[0] = [n layers * n directions, batch size, hid dim]
        #hidden[1] = [n layers * n directions, batch size, hid dim]
        #encoder_outputs = [src len, batch size, hid dim = emb dim * n_directions]
        
        encoder_outputs = encoder_outputs.squeeze() 
        input = input.unsqueeze(0)  #input = [1, batch size]
        embedded = self.embedding(input)

        # Calculating Alignment Scores
        x = torch.tanh( self.fc_hidden(hidden[0]) + self.fc_encoder(encoder_outputs) )
        w_alignment = self.weight.unsqueeze(2) 
        alignment_scores = x.bmm(w_alignment)

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        # Concatenating context vector with embedded input word
        output = torch.cat((embedded[0], context_vector[0]), 1).unsqueeze(0) 

        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)
        
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden[0] = [n layers * n directions, batch size, hid dim]
        #hidden[1] = [n layers * n directions, batch size, hid dim]
        
        prediction = self.out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, attn_weights

    def init_hidden(self):
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)

        return (Variable(h0), Variable(c0))

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        #tensor to store decoder outputs
        if self.training is True:
            trg_len = trg.shape[0]
            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        else:
            outputs = torch.zeros(2*self.decoder.max_length, batch_size, trg_vocab_size) 

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, h = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        if self.training is True:
            loop_condition = trg_len
        else: 
            loop_condition = 2*self.decoder.max_length

        for t in range(1, loop_condition):
            if t == 1:
                hidden = self.decoder.init_hidden()
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            if self.training is False and top1.item() == EOS_token:
                break

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        outputs = outputs[:t+1]
        return outputs