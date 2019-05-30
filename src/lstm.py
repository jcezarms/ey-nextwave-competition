'''
Functions and classes to help train a LSTM selfwork
'''

import torch.nn as nn
import torch
import numpy as np


class LSTM(nn.Module):
    def __init__(self, output_size, seq_features, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(LSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.lstm = nn.LSTM(seq_features, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out

        lstm_out, hidden = self.lstm(x, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

    def train_nn(self, train_loader, valid_loader, scorer, criterion, optimizer,
                batch_size=64, n_epochs=5, clip=5, print_every=25,
                train_on_gpu=False):

        self.train_on_gpu = train_on_gpu

        if train_on_gpu:
            self.cuda()

        predictions = []
        true_labels = []

        train_losses = []
        valid_losses = []

        counter = 0

        self.train()

        for e in range(n_epochs):
            # initialize hidden state
            h = self.init_hidden(batch_size)

            # batch loop
            self.train()
            for inputs, labels in train_loader:
                if len(inputs)!=batch_size:
                    break

                counter += 1

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                inputs = inputs.type(torch.cuda.LongTensor)
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                self.zero_grad()

                # get the output from the model
                output, h = self(inputs.float(), h)
                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), labels.float())
                train_losses.append(loss.item())
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()

                # loss stats
                # Get validation loss

            val_h = self.init_hidden(batch_size)
            self.eval()

            for inputs, labels in valid_loader:
                if len(inputs)!=batch_size:
                    break

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                inputs = inputs.type(torch.cuda.LongTensor)
                val_h = tuple([each.data for each in val_h])

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = self(inputs.float(), val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                valid_losses.append(val_loss.item())

                # convert output probabilities to predicted class (0 or 1)
                pred = torch.round(output.squeeze())  # rounds to the nearest integer

                true_labels.append(labels)
                predictions.append(pred)

        predicted = np.array([label for tensor in predictions for label in tensor.cpu().detach().numpy()])
        true      = np.array([label for tensor in true_labels for label in tensor.cpu().detach().numpy()])

        print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Train Loss: {:.6f}...".format(np.mean(train_losses),
                      "Val Loss: {:.6f}".format(np.mean(valid_losses))))
        score = f1_score(true, predicted)

        print("Test score: {:.3f}".format(score))
