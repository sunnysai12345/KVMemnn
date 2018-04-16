import os
import argparse
import numpy as np
import os
import keras
from keras import backend as K
from keras.layers import Lambda
from keras.layers.normalization import BatchNormalization
#from keras import backend as k
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras import regularizers, constraints, initializers, activations
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
from reader import Data,Vocabulary
import tensorflow as tf

def my_init(shape,input3, dtype=None):
    return K.variable(input3)

def dualenc(pad_length=20,batch_size=100,embedding_size=200,n_chars=20,vocab_size=1000,
              n_labels=20,
              embedding_learnable=False,
              encoder_units=256,
              decoder_units=256,
              trainable=True):
    input1 = Input(shape=(pad_length,))
    input2=Input(shape=(pad_length,))
    input3=Input(shape=(13,))
    input_embed1 = Embedding(vocab_size, embedding_size,
                             input_length=pad_length,
                             trainable=True,
                             name='OneHot1')(input1)
    input_embed2=Embedding(vocab_size, embedding_size,
                             input_length=pad_length,
                             trainable=True,
                             name='OneHot2')(input2)
    dropout = Dropout(0.08)(input_embed1)
    encoder1 = LSTM(encoder_units)(input_embed1)
    encoder2 = LSTM(decoder_units)(input_embed2)
    output = Dense(1)(keras.layers.concatenate([encoder1,encoder2]))
    output=BatchNormalization()(output)
    output=Activation(activation='sigmoid')(output)
    model = Model([input1,input2,input3], output)
    model.layers[4].states[1]=input3
    print(model.summary())
    return model
dualenc()