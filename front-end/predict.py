import pymongo
import numpy as np
import pandas as pd
import json
import pickle
from bson.binary import Binary
import argparse
import gc

from sklearn import metrics

import datetime

# torch.cuda.set_device(1)
# torch.cuda.set_device(1)

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from spp_layer import spatial_pyramid_pool
from residual_block import *
from word2vec import *
from proteinCNN import *
from sppDatapre import *

if torch.cuda.is_available():
    device = torch.device("cuda")  # "cuda:0"
else:
    device = torch.device("cpu")

# connect database
def connect_db(db_name, coll_name):
    mongo_conn = pymongo.MongoClient(host='localhost', port=27017)
    # db = mongo_conn.get_database("mcpi_data")  # specify database
    db = mongo_conn.get_database(db_name)  # specify database
    # coll = db.get_collection("compound_node2vec")  # get collection
    coll = db.get_collection(coll_name)  # get collection
    return coll

# select data
def load_data(args):
    compound_coll = connect_db(args.db_name, args.compound_coll_name)
    protein_coll = connect_db(args.db_name, args.protein_coll_name)
    compound_result = compound_coll.find_one({"smiles": args.smiles})
    protein_result = protein_coll.find_one({"sequence": args.seq})
    dm = pickle.loads(compound_result["distance_matrix"])
    dm = load_tensor(dm, torch.FloatTensor)
    dm = create_variable(dm)
    fp = pickle.loads(compound_result["finger_print"])
    fp = load_tensor(fp, torch.FloatTensor)
    fp = create_variable(fp)
    cn2v = pickle.loads(compound_result["node2vec"])
    cn2v = load_tensor(cn2v, torch.FloatTensor)
    cn2v = create_variable(cn2v)
    pw2v =pickle.loads(protein_result["word2vec"])
    pw2v = load_tensor(pw2v, torch.FloatTensor)
    pw2v = create_variable(pw2v)
    pn2v = pickle.loads(protein_result["node2vec"])
    pn2v = load_tensor(pn2v, torch.FloatTensor)
    pn2v = create_variable(pn2v)
    return dm, fp, cn2v, pw2v, pn2v

def load_tensor(data, dtype):
    return dtype([data])

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

# predict
def predict(args):
    model = torch.load(args.model)
    dm, fp, cn2v, pw2v, pn2v = load_data(args)
    result = model.forward(dm, fp, pw2v, cn2v, pn2v)
    if int(result) >= 0.5:
        return 1
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="prediction", description="compound_protein_prediction")
    parser.add_argument("-m", "--model", dest="model", type=str, default="./data/model.pkl")
    parser.add_argument("-d", "--db_name", dest="db_name", type=str)
    parser.add_argument("-c", "--compound_coll_name", dest="compound_coll_name", type=str)
    parser.add_argument("-p", "--protein_coll_name", dest="protein_coll_name", type=str)
    parser.add_argument("-s", "--seq", dest="seq", type=str)
    parser.add_argument("-i", "--smiles", dest="smiles", type=str)
    args = parser.parse_args()
    dm, fp, cn2v, pw2v, pn2v = load_data(args)
    print(predict(args))
# -d mcpi_database -c compound_data -p protein_data -s MSPLNQSAEGLPQEASNRSLNATETSEAWDPRTLQALKISLAVVLSVITLATVLSNAFVLTTILLTRKLHTPANYLIGSLATTDLLVSILVMPISIAYTITHTWNFGQILCDIWLSSDITCCTASILHLCVIALDRYWAITDALEYSKRRTAGHAATMIAIVWAISICISIPPLFWRQAKAQEEMSDCLVNTSQISYTIYSTCGAFYIPSVLLIILYGRIYRAARNRILNPPSLYGKRFTTAHLITGSAGSSLCSLNSSLHEGHSHSAGSPLFFNHVKIKLADSALERKRISAARERKATKILGIILGAFIICWLPFFVVSLVLPICRDSCWIHPALFDFFTWLGYLNSLINPIIYTVFNEEFRQAFQKIVPFRKAS -i CC[C@@]1(C[C@@H]2C3=CC(=C(C=C3CCN2C[C@H]1CC(C)C)OC)OC)O