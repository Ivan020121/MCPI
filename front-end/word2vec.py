import re
import numpy as np
import argparse as arg

import warnings

import pandas as pd

warnings.filterwarnings('ignore')

from gensim.models import Word2Vec
from gensim import models
from gensim.models import KeyedVectors
from datetime import datetime
from gensim.models import FastText

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_protein_word2vec(model, protein):
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

def get_protein_embedding(trainFoldPath):
    model = Word2Vec.load("wordmodel")
    trainDataSet = pd.read_csv(trainFoldPath)
    trainPro_seq = trainDataSet['sequence']
    proteinSet = []
    for protein in trainPro_seq:
        text = str(protein)
        N = len(text)
        word = [text[i:i+3] for i in range(N - 3 + 1)]
        w2vp = get_protein_word2vec(model, word)
        proteinSet.append(w2vp)
        word = []
    return proteinSet



def get_proteinSeq(trainFoldPath):
    with open(trainFoldPath, 'r') as f:
        trainDataSet = f.read().strip().split('\n')
    trainPro_seq = [seq.strip().split()[1] for seq in trainDataSet]
    word = []
    seqDataSet = []
    for i in range(len(trainPro_seq)):
        text = trainPro_seq[i]
        for j in range(3):
            word.append(re.findall(r'.{3}', text))
            text = text[1: ]
        # print(np.array(word).shape)
        seqDataSet.append(word)
        word = []
    return seqDataSet


def word2vec(seqDataSet):
    model = Word2Vec.load("wordmodel")
    w2c = []
    proteinSet = []
    textSet = []
    for protein in seqDataSet:
        max = len(protein[0])
        for text in protein:
            length = len(text)
            for word in text:
                vec = model.wv[word]
                textSet.append(vec)
            if max != length:
                a = np.zeros((100,))
                textSet.append(a)
            proteinSet.append(textSet)
            textSet = []
        w2c.append(proteinSet)
        proteinSet = []
    w2c = torch.FloatTensor(w2c)
    return w2c

# a = get_protein_embedding('sequence.csv')
# a = np.array(a)
# np.save('proteinWord2Vec.npy', a)