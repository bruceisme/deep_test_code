import clloections
import math
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).init(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
    
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)

        return self.decoder(dec_X, dec_state)





class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)

        #嵌入层
        self.embedding=nn.Embedding(vocab_size, embed_size)
        #编码层
        self.rnn=nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        #X形状(batch_size, num_steps. embed_size)
        X=self.embedding(X)
        #X形状(num_steps.batch_size, embed_size)
        X=X.permute(1,0,2)
        output, state=self.rnn(X)
        #outputs形状(num_steps, batch_size, num_hiddens)
        #state[0]的形状（num_layers, batch_size, num_hiddens）
        return output, state

class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def forward(self, X, state):
        #X形状（batch_size, num_steps, embed_size）
        X=self.embedding(X).permute(1,0,2)
        context=state[-1].repeat(X.shape[0], 1, 1)
        X_and_context=torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1,0,2)
        #output的形状(batch_size, num_steps. vocab_size)

        return output, state

