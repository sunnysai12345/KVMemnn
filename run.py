"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
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
from model.memnn import memnn


# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Dataset functions
    vocab = Vocabulary('./data/vocabulary.json', padding=args.padding)
    vocab = Vocabulary('./data/vocabulary.json',
                              padding=args.padding)
    kb_vocab=Vocabulary('./data/vocabulary.json',
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

    model = memnn(pad_length=args.padding,
                  embedding_size=args.embedding,
                  vocab_size=vocab.size(),
                  batch_size=args.batch_size,
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  embedding_learnable=True,
                  encoder_units=200,
                  decoder_units=200,trainable=True)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', ])
    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')

    try:
        model.fit_generator(generator=training.generator(args.batch_size),
                            steps_per_epoch=300,
                            validation_data=validation.generator(args.batch_size),
                            validation_steps=10,
                            workers=1,
                            verbose=1,
                            epochs=args.epochs)

    except KeyboardInterrupt as e:
        print('Model training stopped early.')
    model.save_weights("model_weights_nkbb.hdf5")

    print('Model training complete.')

    #run_examples(model, input_vocab, output_vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=130, type=int)
    named_args.add_argument('-es', '--embedding', metavar='|',
                            help="""Size of the embedding""",
                            required=False, default=200, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='1', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=20, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/train_data.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/val_data.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=100, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
