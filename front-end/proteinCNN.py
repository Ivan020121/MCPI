import torch
import torch.nn as nn
import torch.nn.functional as F

# if torch.cuda.is_available():
#     device = torch.device("cuda")  # "cuda:0"
# else:
#     device = torch.device("cpu")


class CNN(torch.nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim # 100
        self.hid_dim = hid_dim # 100
        self.kernel_size = kernel_size # 5
        self.dropout = dropout # 0.1
        self.n_layers = n_layers # 3
        # self.device = device # cuda:0
        # self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)

    def forward(self, protein):
        # pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        # protein = protein + self.pos_embedding(pos)
        # protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2*hid dim, protein len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, protein len]

            # apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, protein len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = F.max_pool1d(conved, int(conved.shape[2]))
        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        return conved
    # def __init__(self, plensize, s1, sa1, s2, sa2, s3, sa3, j1,
    #              pf1, ja1, j2, pf2, ja2, j3, pf3, ja3, n_hid):
    #     # prosize, plensize_20 = size of protein one hot feature matrix 5762 20  776 100
    #     # batchsize = 100
    #     # j1, s1, pf1 = window-size, stride-step, No. of filters of first protein-CNN convolution layer 33 1 64
    #     # ja1, sa1 = window-size, stride-step of first protein-CNN average-pooling layer 17 1
    #     # j2, s2, pf2 = window-size, stride-step, No. of filters of second protein-CNN convolution layer 23 1 64
    #     # ja2, sa2 = window-size, stride-step of second protein-CNN average-pooling layer 11 1
    #     # j3, s3, pf3 = window-size, stride-step, No. of filters of third protein-CNN convolution layer 33 1 32
    #     # ja3, sa3 = window-size, stride-step of third protein-CNN average-pooling layer 17 1
    #
    #
    #     super(CNN, self).__init__()
    #     self.conv1_pro = nn.Conv1d(1, pf1, j1, stride=s1, padding=(j1 // 2, 0))
    #     self.bn1_pro = nn.BatchNorm2d(pf1)
    #     self.conv2_pro = nn.Conv1d(pf1, pf2, j2, stride=s2, padding=(j2 // 2, 0))
    #     self.bn2_pro = nn.BatchNorm2d(pf2)
    #     self.conv3_pro = nn.Conv1d(pf2, pf3, j3, stride=s3, padding=(j3 // 2, 0))
    #     self.bn3_pro = nn.BatchNorm2d(pf3)
    #     self.fc3_pro=nn.Linear(100, n_hid)
    #     self.n_hid = n_hid
    #     self.plensize = plensize
    #     self.s1, self.sa1, self.s2, self.sa2, self.s3, self.sa3 = s1, sa1, s2, sa2, s3, sa3
    #     self.j1, self.ja1, self.j2, self.ja2, self.j3, self.ja3 = j1, ja1, j2, ja2, j3, ja3
    #
    #
    # def protein_cnn(self, seq, prosize):
    #     self.m1 = (prosize + (self.j1 // 2 * 2) - self.j1) // self.s1 + 1
    #     self.m2 = (self.m1 + (self.ja1 // 2 * 2) - self.ja1) // self.sa1 + 1
    #     self.m3 = (self.m2 + (self.j2 // 2 * 2) - self.j2) // self.s2 + 1
    #     self.m4 = (self.m3 + (self.ja2 // 2 * 2) - self.ja2) // self.sa2 + 1
    #     self.m5 = (self.m4 + (self.j3 // 2 * 2) - self.j3) // self.s3 + 1
    #     self.m6 = (self.m5 + (self.ja3 // 2 * 2) - self.ja3) // self.sa3 + 1
    #     seq = torch.unsqueeze(seq, 1)
    #     h = F.dropout(F.leaky_relu(self.bn1_pro(self.conv1_pro(seq))), p=0.2)  # 1st conv
    #     # print(h.shape) # torch.Size([1, 8, 776, 100])
    #     h = F.avg_pool2d(h, (self.ja1, 1), stride=self.sa1, padding=(self.ja1 // 2, 0))  # 1st pooling
    #     # print(h.shape) # torch.Size([1, 8, 777, 100])
    #     h = F.dropout(F.leaky_relu(self.bn2_pro(self.conv2_pro(h))), p=0.2)  # 2nd conv
    #     # print(h.shape) # torch.Size([1, 4, 777, 100])
    #     h = F.avg_pool2d(h, (self.ja2, 1), stride=self.sa2, padding=(self.ja2 // 2, 0))  # 2nd pooling
    #     # print(h.shape) # torch.Size([1, 4, 778, 100])
    #     h = F.dropout(F.leaky_relu(self.bn3_pro(self.conv3_pro(h))), p=0.2)  # 3rd conv
    #     # print(h.shape) # torch.Size([1, 1, 778, 100])
    #     h = F.avg_pool2d(h, (self.ja3, 1), stride=self.sa3, padding=(self.ja3 // 2, 0))  # 3rd pooling
    #     # print(h.shape) # torch.Size([1, 1, 779, 100])
    #     # h_pro = F.max_pool2d(h, (self.m6, 1))  # grobal max pooling, fingerprint
    #     h_pro = F.max_pool2d(h, (self.m6, 1))
    #     # print(h_pro.shape) # torch.Size([1, 1, 1, 100])
    #     h_pro = F.dropout(F.leaky_relu(self.fc3_pro(h_pro)), p=0.2)  # fully connected_1
    #     # print(h_pro.shape) # torch.Size([1, 1, 1, 100])
    #     return h_pro


'''
if __name__ == "__main__":
    parser = arg.ArgumentParser(description='protein_word2vec_cnn')
    parser.add_argument('--plensize', '-p', type=int, default=20)
    parser.add_argument('--s1', type=int, default=1)
    parser.add_argument('--sa1', type=int, default=1)
    parser.add_argument('--s2', type=int, default=1)
    parser.add_argument('--sa2', type=int, default=1)
    parser.add_argument('--s3', type=int, default=1)
    parser.add_argument('--sa3', type=int, default=1)
    parser.add_argument('--j1', type=int, default=3)
    parser.add_argument('--pf1', type=int, default=8)
    parser.add_argument('--ja1', type=int, default=2)
    parser.add_argument('--j2', type=int, default=3)
    parser.add_argument('--pf2', type=int, default=4)
    parser.add_argument('--ja2', type=int, default=2)
    parser.add_argument('--j3', type=int, default=3)
    parser.add_argument('--pf3', type=int, default=1)
    parser.add_argument('--ja3', type=int, default=2)
    parser.add_argument('--n_hid3', type=int, default=100)
    args = parser.parse_args(args=[])

    # 3 in channels
    # seqDataSet = get_proteinSeq('dataset_filter_2')
    # w2c_protein = word2vec(seqDataSet)

    protein_embedded = []
    w2c_protein = get_protein_embedding('dataset_filter_2')
    w2c_protein = torch.FloatTensor(w2c_protein)# 这里有错误！！！
    w2c_protein = torch.unsqueeze(w2c_protein, 1)
    print(w2c_protein.shape) # torch.size([4, 1, 776, 100])
    for w2c in w2c_protein:
        prosize = len(w2c[0])
        model = CNN(prosize, args.plensize, args.s1, args.sa1, args.s2, args.sa2, args.s3,args.sa3,
                    args.j1, args.pf1, args.ja1, args.j2, args.pf2, args.ja2, args.j3, args.pf3, args.ja3, args.n_hid3)
        protein_vec = model.protein_cnn(w2c)
        protein_vec = torch.squeeze(protein_vec, 1)
        protein_embedded.append(protein_vec)
    protein_embedded = np.array(protein_embedded)
    np.save('3_100_4_15.npy', protein_embedded)
'''
