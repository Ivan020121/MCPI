import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import sys
import pickle

from utils import *

# torch.cuda.set_device(1)
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # "cuda:0"
# else:
#     device = torch.device("cpu")


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)

    return (data - mu) / sigma


class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataSet, matrix, proteins, compoundNet, proteinNet, bits, method='n', pad=False):
        self.padding = pad
        self.dataSet = dataSet  # list:[[smile,seq,label],....]
        self.len = len(dataSet)
        # self.dict = seqContactDict  # dict:{seq:contactMap,....}
        self.properties = [int(x[2]) for x in dataSet]  # labels
        self.property_list = list(sorted(set(self.properties)))
        self.proteins = proteins
        self.matrix = matrix
        self.compoundNet = compoundNet
        self.proteinNet = proteinNet

        self.method = method

        self.bits = bits

    def __getitem__(self, index):
        smiles, seq, label = self.dataSet[index]
        # contactMap = self.dict[seq]     # 为tensor
        smiles = pd.read_csv('ismilesref.csv')
        smiles = smiles['IsomericSMILES'][index]
        protein = self.proteins[index].numpy()
        dm = self.matrix[index].numpy()
        n2vc = self.compoundNet[index].numpy()
        n2vp = self.proteinNet[index].numpy()
        mol = Chem.MolFromSmiles(smiles)


        # 指纹信息
        #  m = Chem.MolFromSmiles(s)
        finger_info = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.bits))  # 分子指纹

        # 数据标准化, 添加于19：22
        if self.method == 'n':
            dm = normalization(dm)
            n2vc = normalization(n2vc)
            n2vp = normalization(n2vp)
            # finger_info = normalization(finger_info)
            protein = normalization(protein)
        elif self.method == 's':
            dm = standardization(dm)
            n2vc = standardization(n2vc)
            n2vp = standardization(n2vp)
            # finger_info = standardization(finger_info)
            protein = standardization(protein)
        else:
            dm = dm

        return dm, finger_info, protein, n2vc, n2vp, int(label)

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)


def load_tensor(file_name, dtype):
    if "protein" in file_name:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]
        # return [dtype(d).transpose(1,0) for d in np.load(file_name + '.npy')]
    else:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


