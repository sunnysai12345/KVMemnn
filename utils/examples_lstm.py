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
from reader_lstm import Data,Vocabulary
from model.dualencoder import dualenc

outdf = {'input': [], 'output': []}
EXAMPLES = ["find starbucks <eos>", "What will the weather in Fresno be in the next 48 hours <eos>",
            "give me directions to the closest grocery store <eos>", "What is the address? <eos>",
            "Remind me to take pills", "tomorrow in inglewood will it be windy?"]

def run_example(model,vocabulary, text1,text2):
    #print(text1,text2)
    encoded = vocabulary.string_to_int(text1)
    encoded1=vocabulary.string_to_int(text2)
    #print(encoded,encoded1,type(encoded),type(encoded1))
    prediction = model.predict([np.array([encoded]),np.array([encoded1]),np.zeros((1,13))])
    prediction = prediction[0][0]
    #print(prediction, type(prediction), prediction.shape)
    return prediction


def run_examples(model, vocabulary, examples):
    ndf = {"input": [], "response": [], "u1": [], "u2": [], "u3": [], "u4": [], "u5": [],"u6":[]}
    predicted = []
    reranked=[]
    for i in range(len(examples)):
        print(i)
        ndf["input"].append(df["input"][i])
        ndf["response"].append(df["response"][i])
        predicted.append([])
        utter=[]
        utter_dict={}
        for j in range(1,6):
            utter_dict[df["u"+str(j)][i]]=run_example(model, vocabulary,df["input"][i],df["u"+str(j)][i])
        sort=sorted(utter_dict.items(), key=lambda x: x[1], reverse=True)
        print(len(sort))
        #tmp=[]
        for k in range(0,6):
            if len(sort)>k:
                ndf["u"+str(k+1)].append(sort[k][0])
            else:
                ndf["u" + str(k + 1)].append("None")

    ndf = pd.DataFrame(ndf)
    ndf.to_csv("cl_reranked_with_lstmfk_kb_t1.csv")
    print("Saved to file")
            #tmp.append(res[0])
        #reranked.append(tmp)


if __name__ == "__main__":
    pad_length = 20
    df = pd.read_csv("../data/ranked_responsesfk_kb.csv",encoding="latin1")
    vocab = Vocabulary('../fkdata/vocabl_fk.json', padding=pad_length)
    model = dualenc(pad_length=20,
                  embedding_size=100,
                  batch_size=1,
                  vocab_size=vocab.size(),
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  encoder_units=256,
                  decoder_units=256)
    weights_file = "../cl_modellstm_weightsfk_kb.hdf5"
    model.load_weights(weights_file, by_name=True)
    run_examples(model,vocab,df)
    #print(data[:3])
    '''ndf = {"input": [], "response": [], "u1": [], "u2": [], "u3": [], "u4": [], "u5": [], "u6": [], "u7": [], "u8": [],"u9": [], "u10": []}
    ndf = {"input": [], "response": [], "ranked": []}
    for i,(inp,out) in enumerate(zip(df["input"],df["response"])):
        ndf["input"].append(inp)
        ndf["response"].append(out)
        ndf["ranked"].append(data[i][0])
    ndf=pd.DataFrame(ndf)
    ndf.to_csv("reranked_with_lstm.csv")'''



