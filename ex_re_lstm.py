import tensorflow as tf
from keras import backend as K
import keras,os,argparse
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.layers import Lambda
#from keras import backend as k
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Embedding, Activation, Permute
from keras import regularizers, constraints, initializers, activations
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint
from reader_lstm1 import Data,Vocabulary
import pandas as pd
import numpy as np


def run_examples(model, vocabulary, examples):
    ndf = {"output":[],"vector":[]}
    for i in range(len(examples)):
        print(i)
        ndf["output"].append(df["context"][i])
        ndf["vector"].append(run_example(model, vocabulary,df["context"][i],df["response"][i])[:, 50:100])
    ndf = pd.DataFrame(ndf)
    ndf.to_csv("vectors_input.csv")
    print("Saved to file")

def run_example(model,vocabulary, text1,text2):
    #print(text1,text2)
    encoded = vocabulary.string_to_int(text1)
    encoded1=vocabulary.string_to_int(text2)

    prediction = model.predict([np.array([encoded]),np.array([encoded1]),np.array([encoded])])
    print(prediction[:,50:100],type(prediction[:,50:100]), prediction.shape)
    return prediction

def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    anchor = y_pred[:, 0:50]
    positive = y_pred[:, 50:100]
    negative = y_pred[:, 100:150]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def create_base_network(in_dims, out_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Embedding(3003, 100,
                             input_length=20,
                             trainable=True,
                             name='OneHot1'))
    model.add(BatchNormalization())
    model.add(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=2))
    model.add(LSTM(512, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, implementation=2))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(out_dims, activation='linear'))
    model.add(BatchNormalization())

    return model


in_dims = (20,)
out_dims = 50



def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Dataset functions

    vocab = Vocabulary('./fkdata/vocabl_fk.json', padding=args.padding)
    vocab = Vocabulary('./fkdata/vocabl_fk.json',
                              padding=args.padding)
    kb_vocab=Vocabulary('./fkdata/vocabl_fk.json',
                              padding=4)
    print('Loading datasets.')
    training = Data(args.training_data, vocab,kb_vocab)
    validation = Data(args.validation_data, vocab, kb_vocab)
    training.load()
    validation.load()
    training.transform()
    training.kb_out()
    validation.transform()
    validation.kb_out()
    print('Datasets Loaded.')
    print('Compiling Model.')
    # Create the 3 inputs
    anchor_in = Input(shape=in_dims)
    pos_in = Input(shape=in_dims)
    neg_in = Input(shape=in_dims)

    # Share base network with the 3 inputs
    base_network = create_base_network(in_dims, out_dims)
    anchor_out = base_network(anchor_in)
    pos_out = base_network(pos_in)
    neg_out = base_network(neg_in)
    merged_vector = concatenate([anchor_out, pos_out, neg_out], axis=-1)

    # Define the trainable model
    model = Model([anchor_in,pos_in,neg_in],merged_vector)
    model.compile(optimizer=Adam(),
                  loss=triplet_loss)
    print("Model Compiled")
    model.load_weights("re_modellstm_weightsfk_kb.hdf5")
    run_examples(model,vocab,df)


if __name__ == '__main__':
    df = pd.read_csv("./fkdata/testl_fk.csv",encoding="latin1")
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=30, type=int)
    named_args.add_argument('-es', '--embedding', metavar='|',
                            help="""Size of the embedding""",
                            required=False, default=100, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='1', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=20, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/re_train_lstmfk_kb.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/re_val_lstmfk_kb.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=50, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
# Training the model
#model.fit(train_data, y_dummie, batch_size=256, epochs=10)