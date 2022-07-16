import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


train_iter, vocab= d2l.load_data_time_machine(batch_size, num_steps)
vocab_size=len(vocab)
num_layers=2
num_hiddens=256
num_inputs=vocab_size


class RNNModel(nn.Moudle):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn=rnn_layer
        self.vocab_size=vocab_size
        self.num_hiddens=self.rnn.hidden_size

    def forward(self, inputs, state):
        X=F.one_hot(inputs.T.long(), self.vocab_size)
        X=X.to(torch.float320)
        Y, state=self.rnn(X, state)
        output=self.linear(Y.reshape((-1, Y.shape[-1])))

        return output, state

rnn_layer=nn.RNN(num_inputs, num_hiddens, num_layers)
model = RNNModel(rnn_layer,vocab_size)
