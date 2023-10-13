import re
import pymongo
import numpy as np
import argparse as arg
import time

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

from flask import Flask,render_template,request

app = Flask(__name__)

import pymongo

# torch.cuda.set_device(1)
# torch.cuda.set_device(1)

import torch.utils.data

from sppDatapre import *

if torch.cuda.is_available():
    device = torch.device("cuda")  # "cuda:0"
else:
    device = torch.device("cpu")

def get_protein_word2vec(model, protein):
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec

def get_protein_embedding(trainPro_seq):
    model = Word2Vec.load("wordmodel")
    # trainDataSet = pd.read_csv(trainFoldPath)
    # trainPro_seq = trainDataSet['sequence']
    proteinSet = []
    for protein in trainPro_seq:
        text = str(protein)
        N = len(text)
        word = [text[i:i+3] for i in range(N - 3 + 1)]
        w2vp = get_protein_word2vec(model, word)
        proteinSet.append(w2vp)
        word = []
    return proteinSet

# connect database
def connect_db(db_name, coll_name):
    mongo_conn = pymongo.MongoClient(host='localhost', port=27017)
    # db = mongo_conn.get_database("mcpi_data")  # specify database
    db = mongo_conn.get_database(db_name)  # specify database
    # coll = db.get_collection("compound_node2vec")  # get collection
    coll = db.get_collection(coll_name)  # get collection
    return coll

# select data
def load_data(db_name, protein_coll_name, seq):
    protein_coll = connect_db(db_name, protein_coll_name)
    protein_result = protein_coll.find_one({"sequence": seq})
    p_id = protein_result["_id"]
    pw2v =pickle.loads(protein_result["word2vec"])
    pw2v = load_tensor(pw2v, torch.FloatTensor)
    # pw2v = create_variable(pw2v)
    pn2v = pickle.loads(protein_result["node2vec"])
    pn2v = load_tensor(pn2v, torch.FloatTensor)
    # pn2v = create_variable(pn2v)
    return p_id, pw2v, pn2v

def load_tensor(data, dtype):
    return dtype([data])

if __name__ == "__main__":

    start = time.clock()
    trainPro_seq = ['MSPLNQSAEGLPQEASNRSLNATETSEAWDPRTLQALKISLAVVLSVITLATVLSNAFVLTTILLTRKLHTPANYLIGSLATTDLLVSILVMPISIAYTITHTWNFGQILCDIWLSSDITCCTASILHLCVIALDRYWAITDALEYSKRRTAGHAATMIAIVWAISICISIPPLFWRQAKAQEEMSDCLVNTSQISYTIYSTCGAFYIPSVLLIILYGRIYRAARNRILNPPSLYGKRFTTAHLITGSAGSSLCSLNSSLHEGHSHSAGSPLFFNHVKIKLADSALERKRISAARERKATKILGIILGAFIICWLPFFVVSLVLPICRDSCWIHPALFDFFTWLGYLNSLINPIIYTVFNEEFRQAFQKIVPFRKAS']
    proteinSet = get_protein_embedding(trainPro_seq)
    end = time.clock()
    print(end-start)

    start = time.clock()
    p_id, pw2v, pn2v = load_data("mcpi_database", "protein_data", trainPro_seq[0])
    end = time.clock()
    print(end-start)