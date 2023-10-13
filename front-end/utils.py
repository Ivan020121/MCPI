import csv

import numpy as np
import re

import pandas as pd
import torch
from torch.autograd import Variable

import pubchempy as pcp

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from torch.utils.data import Dataset, DataLoader

from pyensembl import EnsemblRelease

def get_3DDistanceMatrix(trainFoldPath):
    with open(trainFoldPath, 'r') as f:# 打开 smiles存储文件
        trainCpi_list = f.read().strip().split('\n')# 按照行分割数据
    trainDataSet = [cpi.strip().split()[0] for cpi in trainCpi_list]#
    smilesDataset = []
    for smile in trainDataSet:
        mol = Chem.MolFromSmiles(smile)
        bm = molDG.GetMoleculeBoundsMatrix(mol)
        print(len(bm))
        # mol2 = Chem.AddHs(mol) # 加氢
        AllChem.EmbedMolecule(mol, randomSeed=1)      #通过距离几何算法计算3D坐标
        dm = AllChem.Get3DDistanceMatrix(mol)
        dm_tensor = torch.FloatTensor([sl for sl in dm])
        # print(len(dm))

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

# def getTrainDataSet(trainFoldPath):
#     with open(trainFoldPath, 'r') as f:
#         trainCpi_list = f.read().strip().split('\n')
#     trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
#     return trainDataSet#[[smiles, sequence, interaction],.....]
def getTrainDataSet(trainFoldPath):
    trainDataSet = pd.read_csv(trainFoldPath)
    chemical = trainDataSet['chemical']
    protein = trainDataSet['protein']
    label = trainDataSet['label']
    trainDataSet = []
    for i in range(len(label)):
        trainDataSet.append([chemical[i], protein[i], label[i]])
    return trainDataSet#[[smiles, sequence, interaction],.....]

def loadTrainSete(trainFoldPath):
    trainSet = pd.read_csv(trainFoldPath)
    c_id = trainSet.chemical.tolist()
    p_id = trainSet.protein.tolist()
    # label = trainSet.label.tolist()
    ens = EnsemblRelease(93)
    pcp.download('CSV', 'ismilesref.csv', c_id, operation='property/IsomericSMILES', overwrite=True)
    smileb = pd.read_csv('ismilesref.csv')
    smib = []
    for j in smileb['IsomericSMILES']:
        smib.append(Chem.MolToSmiles(Chem.MolFromSmiles(j), kekuleSmiles=False, isomericSmiles=True))
        toseq = []
    for j in p_id:
        seq = ens.protein_sequence(j)  # get amino acid seq from ENSP using pyensembl
        toseq.append(seq)
    with open('sequence.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_id', 'sequence'])
        for i in range(len(p_id)):
            writer.writerow([str(p_id[i]), str(toseq[i])])

def saveData(dataSetPath):
    data = pd.read_csv(dataSetPath)
    id = data['CID']
    smiles = data['IsomericSMILES']
    with open('ismilesref.txt', 'w') as f:
        for i in range(len(id)):
            f.writelines([str(id[i]), ' ', str(smiles[i]), '\n'])

def screen(dataSetPath):
    try:
        i = 0
        dm = []
        # data = open(dataSetPath, 'r')
        data = pd.read_csv(dataSetPath)
        id = data['CID']
        data = data['IsomericSMILES']
        # id = [id.strip().split()[0] for id in data]
        # data = [smile.strip().split()[1] for smile in data]
        max = len(data)
        while True:
            if i < max:
                mol = Chem.MolFromSmiles(data[i])
                bm = molDG.GetMoleculeBoundsMatrix(mol)
                AllChem.EmbedMolecule(mol, randomSeed=1, useRandomCoords=True)
                AllChem.Get3DDistanceMatrix(mol)
                dm.append(AllChem.Get3DDistanceMatrix(mol))
                i += 1
            else:
                break
        np.save('dataset_filter_2_smiles.npy', dm)
    except BaseException:
        print(i)
        print(id[i])




# if __name__ == "__main__":
#    screen('ismilesref.csv')