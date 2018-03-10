import numpy as np
import keras
from keras.models import Model, load_model
from reader import Data, Vocabulary
import pandas as pd
import os
import argparse
import numpy as np
import os
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from reader import Data,Vocabulary
from model.memnn import memnn

outdf = {'input': [], 'output': []}
EXAMPLES = ["find starbucks <eos>", "What will the weather in Fresno be in the next 48 hours <eos>",
            "give me directions to the closest grocery store <eos>", "What is the address? <eos>",
            "Remind me to take pills", "tomorrow in inglewood will it be windy?"]

def run_example(model, kbs,vocabulary, text):
    print(text)
    encoded = vocabulary.string_to_int(text)
    #print("encoded is", encoded)
    prediction = model.predict([np.array([encoded]), kbs])
    prediction = np.argmax(prediction[0], axis=-1)
    #print(prediction, type(prediction), prediction.shape)
    print(prediction.shape, vocabulary.int_to_string(prediction))
    return vocabulary.int_to_string(prediction)


def run_examples(model, kbs, vocabulary, examples=EXAMPLES):
    predicted = []
    input = []
    for example in examples:
        print('~~~~~')
        input.append(example)
        predicted.append(' '.join(run_example(model, kbs, vocabulary, example)))
        outdf['input'].append(example)
        outdf['output'].append(predicted[-1])
    return predicted


if __name__ == "__main__":
    pad_length = 20
    df = pd.read_csv("../data/test_data.csv")
    inputs = list(df["inputs"])
    outputs = list(df["outputs"])
    vocab = Vocabulary('../data/vocabulary.json', padding=pad_length)

    kb_vocabulary = Vocabulary('../data/vocabulary.json',padding = 4)

    model = memnn(pad_length=20,
                  embedding_size=200,
                  batch_size=1,
                  vocab_size=vocab.size(),
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  embedding_learnable=True,
                  encoder_units=200,
                  decoder_units=200)
    weights_file = "../weights/model_weights_nkbb.hdf5"
    model.load_weights(weights_file, by_name=True)

    kbfile = "../data/normalised_kbtuples.csv"
    df = pd.read_csv(kbfile)
    kbs = list(df["subject"] + " " + df["relation"])
    # print(kbs[:3])
    kbs = np.array(list(map(kb_vocabulary.string_to_int, kbs)))
    kbs = np.repeat(kbs[np.newaxis, :, :], 1, axis=0)
    data = run_examples(model, kbs,vocab, inputs)
    df=pd.DataFrame(columns=["inputs","outputs","prediction"])
    d = {'outputs':[],'inputs':[],'prediction': []}
    for i, o, p in zip(inputs, outputs, data):
        d["outputs"].append(str(o))
        d["inputs"].append(str(i))
        d["prediction"].append(str(p))
    df = pd.DataFrame(d)
    df.to_csv("output_nkbb.csv")
    # print(outputs)

