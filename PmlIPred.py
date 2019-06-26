# PmlIPred

import numpy as np
import re
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, TimeDistributed, RNN, Bidirectional, normalization
from keras import optimizers, regularizers
from sklearn import ensemble

np.random.seed(1337) # seed
TotalSequenceLength = 0 # the total sequence length
splant = 'aly'

# 参数初始化
sequencepath = 'data\\' + splant + '\\Training-test dataset.fasta' # raw sequence information
listsequence = open(sequencepath, 'r').readlines()
featurepath = 'data\\' + splant + '\\Training-test feature.fasta' # feature information
listfeature = open(featurepath,'r').readlines()
for linelength in listsequence:
    miRNAname, lncRNAname, sequence, label = linelength.split(',')
    if len(sequence) > TotalSequenceLength:
        TotalSequenceLength = len(sequence)

times = 10 # 模型训练次数
K = 10 # K折交叉验证
dp = 0.5 # Dropout率
fuzzy = 0.5 # 模糊决策阈值
strategy = 1 # 混合策略选择，1：完全取代策略(Cor)，2：平均权重策略(Avw)，3：自适应加权策略(Adw)

# 相关指标数据初始化
losssum = 0
accuracysum = 0
TPsum = 0
FPsum = 0
TNsum = 0
FNsum = 0
TPRsum = 0
TNRsum = 0
PPVsum = 0
NPVsum = 0
FNRsum = 0
FPRsum = 0
FDRsum = 0
FORsum = 0
ACCsum = 0
F1sum = 0
MCCsum = 0
BMsum = 0
MKsum = 0
rocsum = []


# 输入文件路径和阈值，输出序列集合和标签集合
def onehot(list, TotalSequenceLength):
    onehotsequence = []
    onehotlabel = []
    ATCG = 'ATCG'  # 字母表
    char_to_int = dict((c, j) for j, c in enumerate(ATCG))  # 设定'A': 0, 'T': 1, 'C': 2, 'G': 3

    for line in list:
        miRNAname, lncRNAname, sequence, label = line.split(',')  # 用','来分离序列

        #  对序列进行编码
        integer_encoded = [char_to_int[char] for char in sequence]  # 对sequence每个字符进行整数编码

        #  one-hot编码
        hot_encoded = []
        # 正常编码
        for value in integer_encoded:
            letter = [0 for _ in range(len(ATCG))]
            letter[value] = 1
            hot_encoded.append(letter)
        # 不足补零
        if len(hot_encoded) < TotalSequenceLength:
            zero = TotalSequenceLength - len(hot_encoded)
            letter = [0 for _ in range(len(ATCG))]
            for i in range(zero):
                hot_encoded.append(letter)

        hot_encoded_array = np.array(hot_encoded).reshape(-1, 4)  # 转置

        onehotsequence.append(hot_encoded_array)

        # 处理标签
        onehotlabel.append(label.strip('\n'))

    X = np.array(onehotsequence).reshape(-1, TotalSequenceLength, 4, 1)  # 序列转为二维矩阵
    X = X.astype('float32')
    Y = np.array(onehotlabel).astype('int').reshape(-1, 1)
    Y = np_utils.to_categorical(Y, num_classes = 2)  # 标签转为二维矩阵

    return X, Y

# 输入X、Y、当前迭代次数和K值，输出训练数据和测试数据
def creatdatadeeplearning(X, Y, iteration, K):

    # 训练集和测试集分离
    totalpartX = len(X)
    partX = int(totalpartX / K)
    totalpartY = len(Y)
    partY = int(totalpartY / K)

    partXstart = iteration * partX # 测试集起点位置
    partXend = partXstart + partX # 测试集终点位置

    partYstart = iteration * partY # 测试集标签起点位置
    partYend = partYstart + partY # 测试集标签终点位置

    traindataP = np.array(X[0 : partXstart])
    traindataL = np.array(X[partXend : totalpartX])
    traindata = np.concatenate((traindataP, traindataL))
    testdata = np.array(X[partXstart : partXend])

    trainlabelP = np.array(Y[0 : partYstart])
    trainlabelL = np.array(Y[partYend : totalpartY])
    trainlabel = np.concatenate((trainlabelP, trainlabelL))
    testlabel = np.array(Y[partYstart : partYend])

    return traindata, trainlabel, testdata, testlabel

# 输入特征标签数据，当前迭代次数，K折，输出训练特征集，训练标签集，测试特征集，测试标签集
def creatdatamachinelearning(data, iteration, K):

    # 训练集和测试集分离
    totalpartdata = len(data)
    partdata = int(totalpartdata / K)
    partdatastart = iteration * partdata  # 测试集起点位置
    partdataend = partdatastart + partdata  # 测试集终点位置
    traindataP = data[0 : partdatastart]
    traindataL = data[partdataend : totalpartdata]
    traindata = traindataP + traindataL
    testdata = data[partdatastart : partdataend]

    # 特征数据和标签分离
    rowtraindata = len(traindata)  # 训练样本数量
    columntraindata = len(traindata[0].split()) - 1  # 训练特征数量，最后减1是因为有一标签列
    rowtestdata = len(testdata)  # 测试样本数量
    columntestdata = len(testdata[0].split()) - 1  # 测试特征数量，最后减1是因为有一标签列

    trainfeature = [([0] * columntraindata) for p in range(rowtraindata)] # 训练特征初始化
    trainlabel = [([0] * 1) for p in range(rowtraindata)] # 训练标签初始化
    for linetraindata in traindata:
        setraindata = re.split(r'\s', linetraindata)
        indextraindata = traindata.index(linetraindata)
        for itraindata in range(len(setraindata) - 1):
            if itraindata < len(setraindata) - 2:
                trainfeature[indextraindata][itraindata] = float(setraindata[itraindata])
            else:
                trainlabel[indextraindata][0] = float(setraindata[itraindata])

    testfeature = [([0] * columntestdata) for p in range(rowtestdata)] # 测试特征初始化
    testlabel = [([0] * 1) for p in range(rowtestdata)] # 测试标签初始化
    for linetestdata in testdata:
        setestdata = re.split(r'\s', linetestdata)
        indextestdata = testdata.index(linetestdata)
        for itestdata in range(0, len(setestdata) - 1):
            if itestdata < len(setestdata) - 2:
                testfeature[indextestdata][itestdata] = float(setestdata[itestdata])
            else:
                testlabel[indextestdata][0] = float(setestdata[itestdata])

    return trainfeature, trainlabel, testfeature, testlabel

# CNN+RNN模型
def CNNRNN(traindata, trainlabel, testdata, testlabel, TotalSequenceLength, times):

    # 构建模型
    model = Sequential()

    # 卷积层1 output shape (16,4,max)
    model.add(Convolution2D(
        batch_input_shape=(None, TotalSequenceLength, 4, 1),
        filters=32,
        kernel_size=4,
        strides=1,
        padding='same',
        data_format='channels_last'))

    # Batch Normalization(BN)层
    normalization.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                     gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                     beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

    # 激励层1
    model.add(Activation('relu'))

    # 卷积层2 output shape (32,4,max)
    model.add(Convolution2D(64, 4, strides=1, padding='same', data_format='channels_first'))

    # Batch Normalization(BN)层
    normalization.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                     gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                     beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

    # 激励层2
    model.add(Activation('relu'))

    # 池化层2 (maxPooling) output shape (64, 7, 7)
    model.add(MaxPooling2D(4, 4, 'same', data_format='channels_last'))

    model.add(TimeDistributed(Flatten())) # 将特征铺平用来接RNN

    # 双向GRU
    model.add(Bidirectional(GRU(units=64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                  dropout=0, recurrent_dropout=0, implementation=1, return_sequences=False, return_state=False, go_backwards=False,
                  stateful=False, unroll=False, reset_after=False)))

    # Drouout层
    model.add(Dropout(dp))

    # 全连接层
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # 优化器选择
    sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # 训练模型
    print('Training --------------')
    model.fit(traindata, trainlabel, epochs = times, batch_size = 64, verbose = 1)

    # 测试模型
    print('\nTesting---------------')
    loss, accuracy = model.evaluate(testdata, testlabel)
    # score = model.evaluate(x_test,y_test,batch_size=128)

    # 获得结果标签
    resultslabel = model.predict(testdata)

    return loss, accuracy, resultslabel

# 输入训练特征集，训练标签集，测试特征集，通过RF输出测试结果
def RF(trainfeature, trainlabel, testfeature):

    RFStruct = ensemble.RandomForestClassifier()
    RFStruct.fit(trainfeature, trainlabel) # 训练
    group = RFStruct.predict(testfeature) # 测试
    score = RFStruct.predict_proba(testfeature) # 获取得分

    return group, score

# 输入测试集标签和结果标签进行对比
def comparisondeeplearning(testlabel, resultslabel):

    # 初始化
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # 将resultslabel格式化
    for row1 in range(resultslabel.shape[0]):
        for column1 in range(resultslabel.shape[1]):
            if resultslabel[row1][column1] < 0.5:
                resultslabel[row1][column1] = 0
            else:
                resultslabel[row1][column1] = 1

    # 计算TP, FP, TN, FN
    for row2 in range(testlabel.shape[0]):

        # TP：testlabel为1(本来为正)，resultslabel为1(鉴定为正)
        if testlabel[row2][0] == 0 and testlabel[row2][1] == 1 and testlabel[row2][0] == resultslabel[row2][0] and testlabel[row2][1] == resultslabel[row2][1]:
            TP = TP + 1
        # FP：testlabel为0(本来为负)，resultslabel为1(鉴定为正)
        if testlabel[row2][0] == 1 and testlabel[row2][1] == 0 and testlabel[row2][0] != resultslabel[row2][0] and testlabel[row2][1] != resultslabel[row2][1]:
            FP = FP + 1
        # TN：testlabel为0(本来为负)，resultslabel为0(鉴定为负)
        if testlabel[row2][0] == 1 and testlabel[row2][1] == 0 and testlabel[row2][0] == resultslabel[row2][0] and testlabel[row2][1] == resultslabel[row2][1]:
            TN = TN + 1
        # FN：testlabel为1(本来为正)，resultslabel为0(鉴定为负)
        if testlabel[row2][0] == 0 and testlabel[row2][1] == 1 and testlabel[row2][0] != resultslabel[row2][0] and testlabel[row2][1] != resultslabel[row2][1]:
            FN = FN + 1

    # TPR：sensitivity, recall, hit rate or true positive rate
    if TP + FN != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 999999

    # TNR：specificity, selectivity or true negative rate
    if TN + FP != 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 999999

    # PPV：precision or positive predictive value
    if TP + FP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 999999

    # NPV：negative predictive value
    if TN + FN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 999999

    # FNR：miss rate or false negative rate
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 999999

    # FPR：fall-out or false positive rate
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 999999

    # FDR：false discovery rate
    if FP + TP != 0:
        FDR = FP / (FP + TP)
    else:
        FDR = 999999

    # FOR：false omission rate
    if FN + TN != 0:
        FOR = FN / (FN + TN)
    else:
        FOR = 999999

    # ACC：accuracy
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 999999

    # F1 score：is the harmonic mean of precision and sensitivity
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 999999

    # MCC：Matthews correlation coefficient
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        MCC = (TP * TN + FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC = 999999

    # BM：Informedness or Bookmaker Informedness
    if TPR != 999999 and TNR != 999999:
        BM = TPR + TNR - 1
    else:
        BM = 999999

    # MK：Markedness
    if PPV != 999999 and NPV != 999999:
        MK = PPV + NPV - 1
    else:
        MK = 999999

    return TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK

# 输入测试标签集，测试结果，输出所有指标对比结果
def comparisonmachinelearning(testlabel, group):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for row in range(len(testlabel)):
        # TP：testlabel为1(本来为正)，group为1(鉴定为正)
        if testlabel[row][0] == 1 and testlabel[row][0] == group[row]:
            TP = TP + 1
        # FP：testlabel为0(本来为负)，group为1(鉴定为正)
        if testlabel[row][0] == 0 and testlabel[row][0] != group[row]:
            FP = FP + 1
        # TN：testlabel为0(本来为负)，group为0(鉴定为负)
        if testlabel[row][0] == 0 and testlabel[row][0] == group[row]:
            TN = TN + 1
        # FN：testlabel为1(本来为正)，group为0(鉴定为负)
        if testlabel[row][0] == 1 and testlabel[row][0] != group[row]:
            FN = FN + 1

    # TPR：sensitivity, recall, hit rate or true positive rate
    if TP + FN != 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 999999

    # TNR：specificity, selectivity or true negative rate
    if TN + FP != 0:
        TNR = TN / (TN + FP)
    else:
        TNR = 999999

    # PPV：precision or positive predictive value
    if TP + FP != 0:
        PPV = TP / (TP + FP)
    else:
        PPV = 999999

    # NPV：negative predictive value
    if TN + FN != 0:
        NPV = TN / (TN + FN)
    else:
        NPV = 999999

    # FNR：miss rate or false negative rate
    if FN + TP != 0:
        FNR = FN / (FN + TP)
    else:
        FNR = 999999

    # FPR：fall-out or false positive rate
    if FP + TN != 0:
        FPR = FP / (FP + TN)
    else:
        FPR = 999999

    # FDR：false discovery rate
    if FP + TP != 0:
        FDR = FP / (FP + TP)
    else:
        FDR = 999999

    # FOR：false omission rate
    if FN + TN != 0:
        FOR = FN / (FN + TN)
    else:
        FOR = 999999

    # ACC：accuracy
    if TP + TN + FP + FN != 0:
        ACC = (TP + TN) / (TP + TN + FP + FN)
    else:
        ACC = 999999

    # F1 score：is the harmonic mean of precision and sensitivity
    if TP + FP + FN != 0:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    else:
        F1 = 999999

    # MCC：Matthews correlation coefficient
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        MCC = (TP * TN + FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC = 999999

    # BM：Informedness or Bookmaker Informedness
    if TPR != 999999 and TNR != 999999:
        BM = TPR + TNR - 1
    else:
        BM = 999999

    # MK：Markedness
    if PPV != 999999 and NPV != 999999:
        MK = PPV + NPV - 1
    else:
        MK = 999999

    return TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK

# 调用onehot函数，获得X, Y
X, Y = onehot(listsequence, TotalSequenceLength)

# K折交叉获得结果
for iteration in range(K):
    # 调用creatdatadeeplearning函数，获得traindata, trainlabel, testdata, testlabel
    traindata, trainlabeldl, testdata, testlabeldl = creatdatadeeplearning(X, Y, iteration, K)

    # 调用CNNRNN函数，获得损失loss、精度accuracy和标签resultslabel
    loss, accuracy, resultslabel = CNNRNN(traindata, trainlabeldl, testdata, testlabeldl, TotalSequenceLength, times)

    # 调用creatdatamachinelearning函数，获得traindata, trainlabel, testdata, testlabel
    trainfeature, trainlabelml, testfeature, testlabelml = creatdatamachinelearning(listfeature, iteration, K)

    # 调用RandomForest函数，获得测试结果group
    RFgroup, RFscore = RF(trainfeature, trainlabelml, testfeature)

    # 模糊决策
    for rowfuz in range(resultslabel.shape[0]):
        if abs(resultslabel[rowfuz][0] - resultslabel[rowfuz][1]) < abs(RFscore[rowfuz][0] - RFscore[rowfuz][1]): # 自适应模糊阈值(At)
        # if abs(resultslabel[rowfuz][0] - resultslabel[rowfuz][1]) <= fuzzy and abs(resultslabel[rowfuz][0] - resultslabel[rowfuz][1]) < abs(RFscore[rowfuz][0] - RFscore[rowfuz][1]): # 人工设置模糊阈值与自适应模糊阈值同时满足
        # if abs(resultslabel[rowfuz][0] - resultslabel[rowfuz][1]) <= fuzzy: # 人工设置模糊阈值

            if strategy == 1: # 完全取代策略
                resultslabel[rowfuz][0] = RFscore[rowfuz][0]
                resultslabel[rowfuz][1] = RFscore[rowfuz][1]

            if strategy == 2: # 平均权重策略
                resultslabel[rowfuz][0] = resultslabel[rowfuz][0] * 0.5 + RFscore[rowfuz][0] * 0.5
                resultslabel[rowfuz][1] = resultslabel[rowfuz][1] * 0.5 + RFscore[rowfuz][1] * 0.5

            if strategy == 3: # 自适应加权策略
                TPDL, FPDL, TNDL, FNDL, TPRDL, TNRDL, PPVDL, NPVDL, FNRDL, FPRDL, FDRDL, FORDL, ACCDL, F1DL, MCCDL, BMDL, MKDL = comparisondeeplearning(testlabeldl, resultslabel)
                TPML, FPML, TNML, FNML, TPRML, TNRML, PPVML, NPVML, FNRML, FPRML, FDRML, FORML, ACCML, F1ML, MCCML, BMML, MKML = comparisonmachinelearning(testlabeldl, RFgroup)
                wdl = (TPRDL + PPVDL + ACCDL)/(TPRDL + PPVDL + ACCDL + TPRML + PPVML + ACCML)
                wml = (TPRML + PPVML + ACCML)/(TPRDL + PPVDL + ACCDL + TPRML + PPVML + ACCML)
                resultslabel[rowfuz][0] = resultslabel[rowfuz][0] * wdl + RFscore[rowfuz][0] * wml
                resultslabel[rowfuz][1] = resultslabel[rowfuz][1] * wdl + RFscore[rowfuz][1] * wml

            ######################################################################################


    # 调用conparison函数，获得指标数据
    TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK = comparisondeeplearning(testlabeldl, resultslabel)

    print('第', iteration + 1, '折')
    print('第',iteration + 1,'折  test loss:', loss, ' test accuracy:', accuracy)
    print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
    print('TPR:', TPR, 'TNR:', TNR, 'PPV:', PPV, 'NPV:', NPV, 'FNR:', FNR, 'FPR:', FPR, 'FDR:', FDR, 'FOR:', FOR)
    print('ACC:', ACC, 'F1:', F1, 'MCC:', MCC, 'BM:', BM, 'MK:', MK)

    # 迭加所有评价指标
    losssum = losssum + loss
    accuracysum = accuracysum + accuracy
    TPsum = TPsum + TP
    FPsum = FPsum + FP
    TNsum = TNsum + TN
    FNsum = FNsum + FN
    TPRsum = TPRsum + TPR
    TNRsum = TNRsum + TNR
    PPVsum = PPVsum + PPV
    NPVsum = NPVsum + NPV
    FNRsum = FNRsum + FNR
    FPRsum = FPRsum + FPR
    FDRsum = FDRsum + FDR
    FORsum = FORsum + FOR
    ACCsum = ACCsum + ACC
    F1sum = F1sum + F1
    MCCsum = MCCsum + MCC
    BMsum = BMsum + BM
    MKsum = MKsum + MK

# 计算平均值
lossaverage = losssum / K
accuracyaverage = accuracysum / K
TPaverage = TPsum / K
FPaverage = FPsum / K
TNaverage = TNsum / K
FNaverage = FNsum / K
TPRaverage = TPRsum / K
TNRaverage = TNRsum / K
PPVaverage = PPVsum / K
NPVaverage = NPVsum / K
FNRaverage = FNRsum / K
FPRaverage = FPRsum / K
FDRaverage = FDRsum / K
FORaverage = FORsum / K
ACCaverage = ACCsum / K
F1average = F1sum / K
MCCaverage = MCCsum / K
BMaverage = BMsum / K
MKaverage = MKsum / K

# 输出结果
print('\ntest average loss: ', lossaverage)
print('\ntest average accuracy: ', accuracyaverage)
print('\ntest average TP: ', TPaverage)
print('\ntest average FP: ', FPaverage)
print('\ntest average TN: ', TNaverage)
print('\ntest average FN: ', FNaverage)
print('\ntest average TPR: ', TPRaverage)
print('\ntest average TNR: ', TNRaverage)
print('\ntest average PPV: ', PPVaverage)
print('\ntest average NPV: ', NPVaverage)
print('\ntest average FNR: ', FNRaverage)
print('\ntest average FPR: ', FPRaverage)
print('\ntest average FDR: ', FDRaverage)
print('\ntest average FOR: ', FORaverage)
print('\ntest average ACC: ', ACCaverage)
print('\ntest average F1: ', F1average)
print('\ntest average MCC: ', MCCaverage)
print('\ntest average BM: ', BMaverage)
print('\ntest average MK: ', MKaverage)
