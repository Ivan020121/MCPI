"""
finger information
"""
import gc

from sklearn import metrics

import datetime

from sppDatapre import *
from sppFingerModel import *
from word2vec import *

import torch.utils.data

# torch.cuda.set_device(1)
# torch.cuda.set_device(1)


def train(trainArgs):

    losses = []
    accs = []
    testResults = {}
    loss_total = []
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'

    # n_time_str = datetime.datetime.now().strftime(ISOTIMEFORMAT) # datetime.datetime.now() 获取当前时间    strftime(ISOTIMEFORMAT) 格式转换
    # # -----
    # info = "spp_4_2_1" + "_p_" + str(p_input_dim) + "_" + str(doc2vec_epoch) + "_finger_" + str(finger_len) + "_in_" \
    #        + str(modelArgs['in_channels']) + "_cnn_" + str(cnn_b1) + "_" + str(cnn_b2) \
    #        + "_layer_" + str(modelArgs['block_num1']) + "_" + str(modelArgs['block_num2']) \
    #        + "_hidden_size_" + str(modelArgs['hidden_nodes']) \
    #        + "_spp_out_" + str(modelArgs['spp_out_dim']) + "_fc_" + str(modelArgs['fc_final']) \
    #        + "_lr_" + str(learning_rate)
    # -----
    # p_input_dim = 200 蛋白质嵌入向量维度
    # doc2vec_epoch = 15 蛋白质处理轮次
    # finger_len = 50 指纹处理向量维度
    # modelArgs['in_channels'] = 64
    # cnn_b1 = 128
    # cnn_b2 = 128
    # modelArgs['block_num1'] = 4
    # modelArgs['block_num2'] = 4
    # modelArgs['hidden_nodes'] = 256
    # modelArgs['spp_out_dim'] = 100 距离矩阵嵌入维度
    # modelArgs['fc_final'] = 100
    # learning_rate = 0.0001

    # test_result = n_time_str + "_" + info + "_Test_" + DataName + ".txt"
    # dev_result = n_time_str + "_" + info + "_Dev_" + DataName + ".txt"
    # train_loss = n_time_str + "_" + info + "_Train_loss" + DataName + ".log"
    # with open(test_result, "a+") as test_f:
    #     test_f.write("testAuc, testPrecision, testRecall, testAcc, testLoss\n") #
    # with open(dev_result, "a+") as dev_f:
    #     dev_f.write("devAuc, devPrecision, devRecall, devAcc, devLoss\n")

    for i in range(trainArgs['epochs']):
        print("Running EPOCH", i + 1)
        total_loss = 0
        n_batches = 0
        correct = 0
        all_pred = np.array([])
        all_target = np.array([])
        train_loader = trainArgs['train_loader'] # torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        optimizer = trainArgs['optimizer'] # torch.optim.Adam(trainArgs['model'].parameters(), lr=trainArgs['lr'])
        criterion = trainArgs["criterion"] # torch.nn.BCELoss()
        attention_model = trainArgs['model'] # SPP_CPI(modelArgs, block=ResidualBlock).to(device)

        print("数据总数", len(train_loader))
        # -----
        for batch_idx, (cmp_dm, finger, word2vecProtein, n2vc, n2vp, y) in enumerate(train_loader):
            cmp_dm = cmp_dm.type(torch.FloatTensor)
            cmp_dm = create_variable(cmp_dm)

            finger = finger.type(torch.FloatTensor)
            finger = create_variable(finger)
            # print(lines.shape, "lines.shape")

            word2vecProtein = create_variable(word2vecProtein)

            n2vc = n2vc.type(torch.FloatTensor)
            n2vc = create_variable(n2vc)

            n2vp = n2vp.type(torch.FloatTensor)
            n2vp = create_variable(n2vp)

            # print(contactmap.shape, "contactmap.shape")
            y_pred = attention_model.forward(cmp_dm, finger, word2vecProtein, n2vc, n2vp)
            all_pred = np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis=0)
            all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis=0)

            # print('y_pred.shape: ', y_pred.shape)
            # print('y.shape: ', y.shape)
            # penalization AAT - I
        # -----
                # binary classification
                # Adding a very small value to prevent BCELoss from outputting NaN's
            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                    y.type(torch.DoubleTensor)).data.sum()


            loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))

            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()  # retain_graph=True

            # gradient clipping
            if trainArgs['clip']:
                torch.nn.utils.clip_grad_norm_(attention_model.parameters(), 0.5)
            optimizer.step()
            n_batches += 1
            # if batch_idx % 10 == 0:
            #     with open(train_loss, "a+") as t_loss:
            #         t_loss.write("Epoch " + str(i + 1) + "\t" + str(batch_idx)+ "\t" + str(loss.item())+"\n")
            #     print(DataName, " [epoch %d]" % (i+1), batch_idx, loss.item())

        avg_loss = total_loss / n_batches
        acc = correct.numpy() / (len(train_loader.dataset))

        losses.append(avg_loss)
        accs.append(acc)

        recall = round(metrics.recall_score(all_target, np.round(all_pred)), 3)


        print("avg_loss is", avg_loss)
        print("train ACC = ", acc)
        print("train recall = ", recall)

        if (trainArgs['doTest']):
            testArgs = {}
            testArgs['model'] = attention_model
            testArgs['criterion'] = trainArgs['criterion']
            testArgs['use_regularizer'] = trainArgs['use_regularizer']
            testArgs['penal_coeff'] = trainArgs['penal_coeff']
            testArgs['clip'] = trainArgs['clip']

            testResult = testPerProteinDataset72(testArgs)
            print(
                "test [len(dev_loader) %d] [Epoch %d/%d] [AUC : %.3f] "
                "[precision : %.3f] [recall : %.3f] [acc : %.3f]"
                % (len(test_loader), i+1, trainArgs['epochs'], testResult[0], testResult[1], testResult[2], testResult[3])
            )
            # with open(test_result, "a+") as test_f:
            #     test_f.write('\t'.join(map(str, testResult)) + '\n')

            # devResult = valPerProteinDataset72(testArgs)
            # print(
            #     "validate [len(dev_loader) %d] [Epoch %d/%d] "
            #     "[AUC : %.3f] [precision : %.3f] [recall : %.3f] [loss : %.3f]"
            #     % (len(dev_loader), i+1, trainArgs['epochs'], devResult[0], devResult[1], devResult[2], devResult[3])
            # )
            # with open(dev_result, "a+") as dev_f:
            #     dev_f.write('\t'.join(map(str, devResult)) + '\n')


    return losses, accs, testResults


def testPerProteinDataset72(testArgs):
    testArgs['test_loader'] = test_loader
    testAcc, testRecall, testPrecision, testAuc, testLoss = test(testArgs)
    # result = [testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3, roce4]
    result = [testAuc, testPrecision, testRecall, testAcc, testLoss]

    return result



def test(testArgs):
    test_loader = testArgs['test_loader']
    criterion = testArgs["criterion"]
    attention_model = testArgs['model']
    losses = []
    accuracy = []
    print('test begin ...')
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    with torch.no_grad():
        # -----
        print(enumerate(test_loader))
        for batch_idx, (cmp_dm, finger, word2vecProtein, n2vc, n2vp, y) in enumerate(test_loader):
            # input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            # attention_model.hidden_state = attention_model.init_hidden()
            cmp_dm = cmp_dm.type(torch.FloatTensor)
            cmp_dm = create_variable(cmp_dm)

            finger = finger.type(torch.FloatTensor)
            finger = create_variable(finger)

            protein = create_variable(word2vecProtein)

            n2vc = n2vc.type(torch.FloatTensor)
            n2vc = create_variable(n2vc)

            n2vp = n2vp.type(torch.FloatTensor)
            n2vp = create_variable(n2vp)

            y_pred = attention_model(cmp_dm, finger, protein, n2vc, n2vp)
        # -----
            if not bool(attention_model.type):
                # binary classification
                # Adding a very small value to prevent BCELoss from outputting NaN's
                pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
                correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                    y.type(torch.DoubleTensor)).data.sum()
                all_pred = np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis=0)
                all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis=0)

                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))
            total_loss += loss.data
            n_batches += 1
    testSize = round(len(test_loader.dataset), 3)
    testAcc = round(correct.numpy() / (n_batches * test_loader.batch_size), 3)
    testRecall = round(metrics.recall_score(all_target, np.round(all_pred)), 3)
    testPrecision = round(metrics.precision_score(all_target, np.round(all_pred)), 3)
    testAuc = round(metrics.roc_auc_score(all_target, all_pred), 3)
    print("AUPR = ", metrics.average_precision_score(all_target, all_pred))
    testLoss = round(total_loss.item() / n_batches, 5)
    # print("test size =", testSize, "  test acc =", testAcc, "  test recall =", testRecall, "  test precision =",
    #       testPrecision, "  test auc =", testAuc, "  test loss = ", testLoss)

    return testAcc, testRecall, testPrecision, testAuc, testLoss

if __name__ == "__main__":

    print('get train datas....')

    print('get seq-contact dict....')
    # seqContactDict = getSeqContactDict(contactPath, contactDictPath)
    print('get letters....')

    print('train loader....')


    DataName = "human"
    dataset_name = "dataset_filter_2"
    trainFoldPath = "train.csv"
    TotalDataset = getTrainDataSet(trainFoldPath)    # [[smiles, sequence, interaction],.....]

    vector_len = 100
    compound_network = 128
    protein_network = 128
    k_gram = 3
    w_dows = 4
    doc2vec_epoch = 15
    p_name = 'sequence.csv'
    # ../data/human/dataset_file_2_protein_doc2vec_3_200_4_15
    p_sequence = get_protein_embedding(p_name)
    trian_proteinDataset = [torch.FloatTensor(seq) for seq in p_sequence]    # protein doc2vec
    del p_sequence
    gc.collect()
    # ../data/human/dataset_file_2_smiles
    smiles_file_path = dataset_name + "_smiles"

    matrix_tensor = load_tensor(smiles_file_path, torch.FloatTensor)  # compound distanceMatrix

    #-----
    # ../data/human/dataset_file_2_proteins_node2vec
    protein_n2v_path = dataset_name + "_proteins_node2vec"
    n2vp = load_tensor(protein_n2v_path, torch.FloatTensor)

    # ../data/human/dataset_file_2_compound_node2vec
    compound_n2v_path = dataset_name + "_compounds_node2vec"
    n2vc = load_tensor(compound_n2v_path, torch.FloatTensor)
    #-----

    b_size = 1
    finger_len = 50
    data_process_method = 'n'

    total_dataset = ProDataset(dataSet=TotalDataset, matrix=matrix_tensor, proteins=trian_proteinDataset,
                               compoundNet=n2vc, proteinNet=n2vp, bits=finger_len, method=data_process_method)
    # train_loader = DataLoader(dataset=total_dataset, batch_size=1, shuffle=True, drop_last=True)

    # 划分数据集
    train_size = int(0.9 * len(total_dataset))
    other_size = len(total_dataset) - train_size
    # 训练数据集
    train_dataset, other_dataset = torch.utils.data.random_split(total_dataset, [train_size, other_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=False)
    # 验证数据集， 测试数据集

    test_loader = torch.utils.data.DataLoader(other_dataset, batch_size=b_size, shuffle=False)

    print('model args...')

    modelArgs = {}
    modelArgs['batch_size'] = 1
    modelArgs['protein_input_dim'] = vector_len
    modelArgs['protein_dim'] = vector_len
    modelArgs['hid_dim'] = vector_len
    modelArgs['n_layers'] = 3
    modelArgs['kernel_size'] = 5
    modelArgs['dropout'] = 0.1
    modelArgs['protein_fc'] = vector_len
    modelArgs['finger'] = finger_len
    modelArgs['compound_network'] = compound_network
    modelArgs['protein_network'] = protein_network
    modelArgs['d_a'] = 32
    # d_a = modelArgs['d_a']
    modelArgs['in_channels'] = 64
    modelArgs['cnn_channel_block1'] = 128
    modelArgs['cnn_channel_block2'] = 128
    cnn_b1 = modelArgs['cnn_channel_block1']
    cnn_b2 = modelArgs['cnn_channel_block2']
    modelArgs['block_num1'] = 4       # resual block， CNN
    modelArgs['block_num2'] = 4

    modelArgs['r'] = 20
    modelArgs['cnn_layers'] = 4
    modelArgs['hidden_nodes'] = 256
    modelArgs['spp_out_dim'] = 100
    modelArgs['fc_final'] = 100

    p_input_dim = modelArgs['protein_input_dim']
    modelArgs['task_type'] = 0
    modelArgs['n_classes'] = 1


    print('train args...')

    trainArgs = {}
    trainArgs['model'] = SPP_CPI(modelArgs, block=ResidualBlock, proteinCNN=CNN)
    trainArgs['epochs'] = 70
    trainArgs['lr'] = 0.0001
    learning_rate = trainArgs['lr']
    trainArgs['train_loader'] = train_loader
    trainArgs['doTest'] = True
    trainArgs['use_regularizer'] = False
    trainArgs['penal_coeff'] = 0.03
    trainArgs['clip'] = True
    trainArgs['criterion'] = torch.nn.BCELoss()
    trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(), lr=trainArgs['lr'])
    trainArgs['doSave'] = True
    print('train args over...')
    # for name, parameter in trainArgs['model'].named_parameters():
    #     print(name, ':', parameter.size())
    losses, accs, testResults = train(trainArgs)
    print(losses)
    print(accs)
    print(testResults)
    torch.save(trainArgs['model'], 'model.pkl')