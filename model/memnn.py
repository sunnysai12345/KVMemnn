import os
import argparse
import numpy as np
import os
import keras
from keras.layers import Lambda
#from keras import backend as k
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras import regularizers, constraints, initializers, activations
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
from reader import Data,Vocabulary

def reshape4(tensor,batch_size,pad_length,seq_length):
    v = keras.backend.zeros((batch_size, pad_length, 431))
    tensor = keras.backend.reshape(tensor, (batch_size, pad_length, seq_length))
    tensor = keras.layers.concatenate([tensor, v], axis=2)
    return tensor

def reshape(tensor,batch_size,seq_length,embed_size,pad_length):
    tensor = keras.backend.sum(tensor, axis=2)
    tensor=keras.backend.reshape(tensor,(batch_size,embed_size,seq_length))
    return tensor

def reshape2(tensor,batch_size,pad_length,seq_length):
    tensor=tensor[:batch_size, :pad_length,]
    v = keras.backend.zeros((batch_size, pad_length, 1523))
    tensor = keras.backend.reshape(tensor, (batch_size, pad_length,431))
    tensor = keras.layers.concatenate([v, tensor], axis=2)
    return tensor

def reshaped(tensor,batch_size,pad_length,seq_length):
    tensor = keras.backend.reshape(tensor, (batch_size,pad_length,seq_length))
    return tensor

def memnn(pad_length=20,batch_size=100,embedding_size=200,n_chars=20,vocab_size=1000,
              n_labels=20,
              embedding_learnable=False,
              encoder_units=256,
              decoder_units=256,
              trainable=True):
    input1 = Input(shape=(pad_length,), dtype='float32')
    input2 = Input(shape=(431, 4), dtype='float32')

    input_embed1 = Embedding(vocab_size, embedding_size,
                             input_length=pad_length,
                             trainable=True,
                             name='OneHot1')(input1)
    input_embed2 = Embedding(vocab_size, embedding_size,
                             input_length=431,
                             trainable=True,
                             name='OneHot2')(input2)
    input_embed2 = Lambda(reshape, arguments={'batch_size': batch_size, 'seq_length': 431, 'embed_size': embedding_size,
                                              'pad_length': pad_length}, name='input_key_embed')(input_embed2)
    dropout = Dropout(0.2)(input_embed1)
    encoder = LSTM(encoder_units, return_sequences=True)(dropout)
    decoder = LSTM(decoder_units, return_sequences=True)(encoder)
    dense1 = Dense(200, activation='tanh')(encoder)
    dense2 = Dense(200, activation='tanh')(decoder)
    dense3 = Dense(200, activation='tanh')(keras.layers.add([dense1, dense2]))
    attention = Activation('softmax')(dense3)
    n_hidden = keras.layers.multiply([attention, encoder])
    output = Dense(1954)(keras.layers.concatenate([encoder, n_hidden]))
    #output = Lambda(reshape4, arguments={'batch_size': batch_size, 'pad_length': pad_length, 'seq_length': 1523},name='lambda2')(output)
    #decoder = Lambda(reshaped)(decoder)
    decoder = Lambda(reshaped,arguments={'batch_size': batch_size, 'pad_length': pad_length, 'seq_length': decoder_units})(decoder)
    n_dense1 = Dense(20, activation='tanh')(input_embed2)
    n_dense1 = Lambda(reshaped,arguments={'batch_size': batch_size, 'pad_length': pad_length, 'seq_length': decoder_units})(n_dense1)
    #n_dense1 = Lambda(reshaped)(n_dense1)
    n_dense2 = Dense(200, activation='tanh')(decoder)
    n_dense3 = Dense(431, activation='tanh')(keras.layers.concatenate([n_dense1, n_dense2], axis=1))
    print(n_dense3.shape)
    n_dense3 = Lambda(reshape2, arguments={'batch_size': batch_size, 'pad_length': pad_length, 'seq_length': 431},
                      name='lambda3')(n_dense3)
    # n_dense3 = Activation('softmax')(n_dense3)
    n_out = keras.layers.add([output, n_dense3])
    n_output = Activation('softmax')(n_out)
    print(input_embed1.shape, input_embed2.shape, encoder.shape, decoder.shape, dense3.shape)
    model = Model([input1, input2], n_output)
    return model