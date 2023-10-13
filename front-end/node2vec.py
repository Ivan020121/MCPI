import pickle
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from pyensembl import EnsemblRelease

from utils import *

#
def node2vec_protein(trainFoldPath, n2vpPath):
    p = []
    with open('proteinN2V.pickle', mode='rb') as f:
        modelp = pickle.load(f)
    dataSet = pd.read_csv(trainFoldPath)
    for protein in dataSet['protein']:
        p.append(modelp.wv[protein])
    print(np.array(p).shape)
    np.save(n2vpPath, p)

def node2vec_compound(trainFoldPath, n2vcPath):
    c = []
    with open('compoundN2V.pickle', mode='rb') as f:
        modelc = pickle.load(f)
    dataSet = pd.read_csv(trainFoldPath)
    for compound in dataSet['chemical']:
        c.append(modelc.wv[str(compound)])
    print(np.array(c).shape)
    np.save(n2vcPath, c)

if __name__ == "__main__":
    trainFoldPath = 'train.csv'
    n2vpPath = 'dataset_filter_2_proteins_node2vec.npy'
    n2vcPath = 'dataset_filter_2_compounds_node2vec.npy'
    node2vec_protein(trainFoldPath, n2vpPath)
    node2vec_compound(trainFoldPath, n2vcPath)

