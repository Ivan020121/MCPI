"""

finger information
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from spp_layer import spatial_pyramid_pool
from residual_block import *
from word2vec import *
from proteinCNN import *


# if torch.cuda.is_available():
#     device = torch.device("cuda")  # "cuda:0"
# else:
#     device = torch.device("cpu")


class SPP_CPI(torch.nn.Module):
    """
    The class is an implementation of the SPP_CPI model including regularization and without pruning.
    Slight modifications have been done for speedup

    """

    def __init__(self, args, block, proteinCNN, train=False):

        super(SPP_CPI, self).__init__()
        self.batch_size = args['batch_size']
        self.train_f = train
        self.r = args['r']
        self.type = args['task_type']
        self.in_channels = args['in_channels']
        self.dim_pro = args['protein_fc']
        self.output_num = [4, 2, 1]
        self.spp_out_dim = args['spp_out_dim']
        self.finger = args['finger']
        self.hidden_nodes = args['hidden_nodes']
        self.compound_network = args['compound_network']
        self.protein_network = args['protein_network']

        # self.blocks_num = [args['block_num1'], args['block_num2']]

        self.fc1 = nn.Linear(args['cnn_channel_block2'] * (1+4+16), self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, self.spp_out_dim)

        self.linear_first_seq = torch.nn.Linear(args['cnn_channel_block2'], args['d_a'])
        self.linear_second_seq = torch.nn.Linear(args['d_a'], self.r)

        # protein_cnn
        self.protein_dim = args['protein_dim']
        self.hid_dim = args['hid_dim']
        self.n_layers = args['n_layers']
        self.kernel_size = args['kernel_size']
        self.dropout = args['dropout']
        self.proteinCNN = proteinCNN(self.protein_dim, self.hid_dim, self.n_layers, self.kernel_size, self.dropout)


        # cnn
        self.conv = conv3x3(1, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.r_elu = nn.ELU(inplace=False)
        self.layer1 = self.make_layer(block, args['cnn_channel_block1'], args['block_num1'])
        self.layer2 = self.make_layer(block, args['cnn_channel_block2'],  args['block_num2'])
        # self.linear_final_step = torch.nn.Linear(self.lstm_hid_dim*2+args['d_a'],args['dense_hid'])
        self.linear_final_step = torch.nn.Linear(self.spp_out_dim + self.finger + self.compound_network
                                                 + self.dim_pro + self.protein_network, args['fc_final'])
        self.linear_final = torch.nn.Linear(args['fc_final'], args['n_classes'])

        # cos_similarity
        self.fc3 = torch.nn.Linear(self.spp_out_dim+self.finger+self.compound_network, 239)
        self.fc4 = torch.nn.Linear(239, 200)
        self.fc5 = torch.nn.Linear(self.dim_pro+self.protein_network, 214)
        self.fc6 = torch.nn.Linear(214, 200)
        self.fc7 = torch.nn.Linear(200, 1)

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def make_layer(self, block, out_channels, block_num, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, block_num):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    # x1 = smiles , x3= proteins
    def forward(self, x1, finger, x3, compoundNet, proteinNet):

        # protein = x3.view(x3.size(0), -1).to(device)
        prosize = len(x3[0])
        model = self.proteinCNN
        # model = model.to(device)
        protein = model(x3)
        protein = torch.squeeze(protein, 0)
        # x1 = x1.type(torch.FloatTensor)
        x1 = torch.unsqueeze(x1, 1)
        pic = self.conv(x1)
        # print(x1.shape, pic.shape)
        pic = self.bn(pic)
        pic = self.r_elu(pic)
        pic = self.layer1(pic)
        pic = self.layer2(pic)

        # print(pic.shape, "pic.shape")

        spp = spatial_pyramid_pool(pic, pic.size(0), [int(pic.size(2)), int(pic.size(3))], self.output_num)
        # print(spp.shape, "spp.shape")

        # print(spp.shape, "spp.shape")
        fc1 = F.relu(self.fc1(spp))
        fc2 = F.relu(self.fc2(fc1))
        # print(fc2.shape)
        # print(finger.shape)
        # print(compoundNet.shape)
        # print(protein.shape)
        x_compound = torch.cat([fc2, finger, compoundNet], dim=1)
        x_protein = torch.cat([protein, proteinNet], dim=1)
        x_compound = self.fc3(x_compound)
        x_compound = F.dropout(F.leaky_relu(x_compound), p=0.2)
        x_compound = F.dropout(F.leaky_relu(self.fc4(x_compound)), p=0.2)
        x_protein = self.fc5(x_protein)
        x_protein = F.dropout(F.leaky_relu(x_protein), p=0.2)
        x_protein = F.dropout(F.leaky_relu(self.fc6(x_protein)), p=0.2)
        y = x_compound*x_protein
        similarity = self.fc7(y)
        # sscomplex = torch.cat([fc2, finger, compoundNet, protein, proteinNet], dim=1)
        # print(sscomplex.shape)
        # print(self.spp_out_dim + self.finger + self.compound_network + self.dim_pro + self.protein_network)
        # sscomplex = torch.relu(self.linear_final_step(sscomplex))
        similarity = torch.sigmoid(similarity)
        return similarity
        # if not bool(self.type):
        #     pred = self.linear_final(sscomplex)
        #     pic_output = torch.sigmoid(pred)
        #     return pic_output
