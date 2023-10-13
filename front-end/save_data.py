import pymongo
import numpy as np
import pandas as pd
import json
import pickle
from bson.binary import Binary

# connect database
def connect_db(db_name, coll_name):
    mongo_conn = pymongo.MongoClient(host='localhost', port=27017)
    # db = mongo_conn.get_database("mcpi_data")  # specify database
    db = mongo_conn.get_database(db_name)  # specify database
    # coll = db.get_collection("compound_node2vec")  # get collection
    coll = db.get_collection(coll_name)  # get collection
    return coll

# construction compound data
def construct_compound_data(csv_path, dm_path, fp_path, n2v_path):
    csv_data = pd.read_csv(csv_path)
    id = csv_data[csv_data.columns[0]]
    smiles = csv_data[csv_data.columns[1]]
    load_dm = np.load(dm_path, allow_pickle=True)
    load_fp = np.load(fp_path, allow_pickle=True)
    load_node2vec = np.load(n2v_path, allow_pickle=True)
    return id, smiles, load_dm, load_fp, load_node2vec

# construction protein data
def construct_protein_data(csv_path, w2v_path, n2v_path):
    csv_data = pd.read_csv(csv_path)
    id = csv_data[csv_data.columns[0]]
    seq = csv_data[csv_data.columns[1]]
    load_w2v = np.load(w2v_path, allow_pickle=True)
    load_node2vec = np.load(n2v_path, allow_pickle=True)
    return id, seq, load_w2v, load_node2vec

# insert compound data
def insert_compound_data(db_name, coll_name, csv_path, dm_path, fp_path, n2v_path):
    coll = connect_db(db_name, coll_name)
    id, smiles, load_dm, load_fp, load_node2vec = construct_compound_data(csv_path, dm_path, fp_path, n2v_path)
    for i in range(len(id)):
        coll.save({
            "_id": str(id[i]),
            "smiles": str(smiles[i]),
            "distance_matrix": Binary(pickle.dumps(load_dm[i], protocol=2)),
            "finger_print": Binary(pickle.dumps(load_fp[i], protocol=2)),
            "node2vec": Binary(pickle.dumps(load_node2vec[i], protocol=2)),
        })

# insert compound data
def insert_protein_data(db_name, coll_name, csv_path, w2v_path, n2v_path):
    coll = connect_db(db_name, coll_name)
    id, seq, load_w2v, load_node2vec = construct_protein_data(csv_path, w2v_path, n2v_path)
    for i in range(len(id)):
        coll.save({
            "_id": str(id[i]),
            "sequence": str(seq[i]),
            "word2vec": Binary(pickle.dumps(load_w2v[i], protocol=2)),
            "node2vec": Binary(pickle.dumps(load_node2vec[i], protocol=2)),
        })

if __name__ == "__main__":
    insert_compound_data("mcpi_database", "compound_data", "./data/ismilesref.csv", "./data/dataset_filter_2_smiles.npy",
                         "./data/dataset_filter_2_fingerprint.npy", "./data/dataset_filter_2_compounds_node2vec.npy")
    insert_protein_data("mcpi_database", "protein_data", "./data/sequence.csv", "./data/dataset_filter_2_protein_word2vec.npy",
                        "./data/dataset_filter_2_proteins_node2vec.npy")