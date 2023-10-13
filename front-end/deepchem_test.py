#get drug features using Deepchem library
import warnings
warnings.filterwarnings("ignore")
import os
import deepchem as dc
from rdkit import Chem
import numpy as np
import numpy.random as random
import pandas as pd
import scipy.sparse as sp
import hickle as hkl
import pandas as pd


def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix

def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2
    return [feat,adj_mat]

def calculate_graph(id_set,drug_feature):
    compound_drug_data ={}
    for pubchem_id in id_set:
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        compound_drug_data[pubchem_id] = CalculateGraphFeat(feat_mat,adj_list)
    return compound_drug_data

def gcn_extract(id_set,drug_data):#处理成gcn可以运行的数据
    #print(X_drug_data_train)
    drug_graph_data = {}
    for pubchem_id in id_set:
        item_drug_data = drug_data[pubchem_id]
        X_drug_feat_data_train = item_drug_data[0]
        X_drug_adj_data_train =  item_drug_data[1]
        X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
        X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
        calculated_drug_data = {'feat':X_drug_feat_data_train,'adj':X_drug_adj_data_train}
        drug_graph_data[pubchem_id] =calculated_drug_data

if __name__ == '__main__':
    Max_atoms = 150
    israndom = False

    id_set = ['D00002']
    molecules = "CCC1=NC2=CC=CC=C2C(=C1C)C(=O)OCC(=O)C3=CC=CC=C3Cl"

    molecules = Chem.MolFromSmiles(molecules)
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mol_object = featurizer.featurize(datapoints=molecules)
    features = mol_object[0].atom_features
    degree_list = mol_object[0].deg_list
    adj_list = mol_object[0].canon_adj_list

    drug_feature = {'D00002': [features, adj_list, degree_list]}

    drug_data = calculate_graph(id_set, drug_feature)
    drug_data = gcn_extract(id_set, drug_data)
    pass