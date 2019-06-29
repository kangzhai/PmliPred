# PmliPred for K-fold cross validation

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
splant = 'aly' # plant species

# Load data
sequencepath = 'data\\' + splant + '\\Training-test dataset.fasta' # raw sequence information
listsequence = open(sequencepath, 'r').readlines()
featurepath = 'data\\' + splant + '\\Training-test feature.fasta' # feature information
listfeature = open(featurepath,'r').readlines()

# Get the maximum length of the sequence
for linelength in listsequence:
    miRNAname, lncRNAname, sequence, label = linelength.split(',')
    if len(sequence) > TotalSequenceLength:
        TotalSequenceLength = len(sequence)

times = 10 # epochs
K = 10 # K-fold cross validation
dp = 0.5 # Dropout rate
threshold = 0.5 # threshold
strategy = 1 # weight strategy selection 1: Complete replacement 2: Average weight 3: Adaption weight

# Evaluation criteria
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

# one-hot encoding
def onehot(list, TotalSequenceLength):
    onehotsequence = []
    onehotlabel = []
    ATCG = 'ATCG'  # alphabet
    char_to_int = dict((c, j) for j, c in enumerate(ATCG))  # set 'A': 0, 'T': 1, 'C': 2, 'G': 3

    for line in list:
        miRNAname, lncRNAname, sequence, label = line.split(',')

        #  integer encoding
        integer_encoded = [char_to_int[char] for char in sequence]

        #  one-hot encoding
        hot_encoded = []
        
		# encoding
        for value in integer_encoded:
            letter = [0 for _ in range(len(ATCG))]
            letter[value] = 1
            hot_encoded.append(letter)
        # zero-padding
        if len(hot_encoded) < TotalSequenceLength:
            zero = TotalSequenceLength - len(hot_encoded)
            letter = [0 for _ in range(len(ATCG))]
            for i in range(zero):
                hot_encoded.append(letter)

        hot_encoded_array = np.array(hot_encoded).reshape(-1, 4)

        onehotsequence.append(hot_encoded_array)
		
        onehotlabel.append(label.strip('\n'))

    X = np.array(onehotsequence).reshape(-1, TotalSequenceLength, 4, 1)
    X = X.astype('float32')
    Y = np.array(onehotlabel).astype('int').reshape(-1, 1)
    Y = np_utils.to_categorical(Y, num_classes = 2)

    return X, Y

# creat deep learning data
def creatdatadeeplearning(X, Y, iteration, K):

    # separate the data
    totalpartX = len(X)
    partX = int(totalpartX / K)
    totalpartY = len(Y)
    partY = int(totalpartY / K)

    partXstart = iteration * partX
    partXend = partXstart + partX

    partYstart = iteration * partY
    partYend = partYstart + partY

    traindataP = np.array(X[0 : partXstart])
    traindataL = np.array(X[partXend : totalpartX])
    traindata = np.concatenate((traindataP, traindataL))
    testdata = np.array(X[partXstart : partXend])

    trainlabelP = np.array(Y[0 : partYstart])
    trainlabelL = np.array(Y[partYend : totalpartY])
    trainlabel = np.concatenate((trainlabelP, trainlabelL))
    testlabel = np.array(Y[partYstart : partYend])

    return traindata, trainlabel, testdata, testlabel

# creat machine learning data
def creatdatamachinelearning(data, iteration, K):

    # separate the data
    totalpartdata = len(data)
    partdata = int(totalpartdata / K)
    partdatastart = iteration * partdata
    partdataend = partdatastart + partdata
    traindataP = data[0 : partdatastart]
    traindataL = data[partdataend : totalpartdata]
    traindata = traindataP + traindataL
    testdata = data[partdatastart : partdataend]

    # separate the label
    rowtraindata = len(traindata)
    columntraindata = len(traindata[0].split()) - 1
    rowtestdata = len(testdata)
    columntestdata = len(testdata[0].split()) - 1
	
	# get the training data and label
    trainfeature = [([0] * columntraindata) for p in range(rowtraindata)]
    trainlabel = [([0] * 1) for p in range(rowtraindata)]
    for linetraindata in traindata:
        setraindata = re.split(r'\s', linetraindata)
        indextraindata = traindata.index(linetraindata)
        for itraindata in range(len(setraindata) - 1):
            if itraindata < len(setraindata) - 2:
                trainfeature[indextraindata][itraindata] = float(setraindata[itraindata])
            else:
                trainlabel[indextraindata][0] = float(setraindata[itraindata])
	
	# get the test data and label
    testfeature = [([0] * columntestdata) for p in range(rowtestdata)]
    testlabel = [([0] * 1) for p in range(rowtestdata)]
    for linetestdata in testdata:
        setestdata = re.split(r'\s', linetestdata)
        indextestdata = testdata.index(linetestdata)
        for itestdata in range(0, len(setestdata) - 1):
            if itestdata < len(setestdata) - 2:
                testfeature[indextestdata][itestdata] = float(setestdata[itestdata])
            else:
                testlabel[indextestdata][0] = float(setestdata[itestdata])

    return trainfeature, trainlabel, testfeature, testlabel

# CNN-Bi-GRU
def CNNBiGRU(traindata, trainlabel, testdata, testlabel, TotalSequenceLength, times):

    # Model
    model = Sequential()

    # Convolution layer
    model.add(Convolution2D(batch_input_shape=(None, TotalSequenceLength, 4, 1), filters=32, kernel_size=4, strides=1, padding='same', data_format='channels_last'))

    # Batch Normalization layer
    normalization.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                     gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                     beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

    # Activation function
    model.add(Activation('relu'))

    # Convolution layer
    model.add(Convolution2D(64, 4, strides=1, padding='same', data_format='channels_first'))

    # Batch Normalization layer
    normalization.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                                     gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                     beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

    # Activation function
    model.add(Activation('relu'))

    # MaxPooling layer
    model.add(MaxPooling2D(4, 4, 'same', data_format='channels_last'))
	
	# Flatten layer
    model.add(TimeDistributed(Flatten()))

    # Bi-GRU
    model.add(Bidirectional(GRU(units=64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                  dropout=0, recurrent_dropout=0, implementation=1, return_sequences=False, return_state=False, go_backwards=False,
                  stateful=False, unroll=False, reset_after=False)))

    # Drouout layer
    model.add(Dropout(dp))

    # fully-connected layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # optimizer
    sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # training
    print('Training --------------')
    model.fit(traindata, trainlabel, epochs = times, batch_size = 64, verbose = 1)

    # test
    print('\nTesting---------------')
    loss, accuracy = model.evaluate(testdata, testlabel)

    # get the confidence coefficient
    resultslabel = model.predict(testdata)

    return resultslabel

# RF
def RF(trainfeature, trainlabel, testfeature):

    RFStruct = ensemble.RandomForestClassifier()
    RFStruct.fit(trainfeature, trainlabel) # training
    group = RFStruct.predict(testfeature) # test
    score = RFStruct.predict_proba(testfeature) # get the confidence coefficient

    return group, score

# calculate the results of deep learning model
def comparisondeeplearning(testlabel, resultslabel):

    # initialization
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # formatting
    for row1 in range(resultslabel.shape[0]):
        for column1 in range(resultslabel.shape[1]):
            if resultslabel[row1][column1] < 0.5:
                resultslabel[row1][column1] = 0
            else:
                resultslabel[row1][column1] = 1

    # TP, FP, TN, FN
    for row2 in range(testlabel.shape[0]):
        # TP
        if testlabel[row2][0] == 0 and testlabel[row2][1] == 1 and testlabel[row2][0] == resultslabel[row2][0] and testlabel[row2][1] == resultslabel[row2][1]:
            TP = TP + 1
        # FP
        if testlabel[row2][0] == 1 and testlabel[row2][1] == 0 and testlabel[row2][0] != resultslabel[row2][0] and testlabel[row2][1] != resultslabel[row2][1]:
            FP = FP + 1
        # TN
        if testlabel[row2][0] == 1 and testlabel[row2][1] == 0 and testlabel[row2][0] == resultslabel[row2][0] and testlabel[row2][1] == resultslabel[row2][1]:
            TN = TN + 1
        # FN
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

# calculate the results of machine learning model
def comparisonmachinelearning(testlabel, group):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for row in range(len(testlabel)):
        # TP
        if testlabel[row][0] == 1 and testlabel[row][0] == group[row]:
            TP = TP + 1
        # FP
        if testlabel[row][0] == 0 and testlabel[row][0] != group[row]:
            FP = FP + 1
        # TN
        if testlabel[row][0] == 0 and testlabel[row][0] == group[row]:
            TN = TN + 1
        # FN
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

# one-hot encoding
X, Y = onehot(listsequence, TotalSequenceLength)

# K-fold cross validation
for iteration in range(K):
    # creat deep learning data
    traindata, trainlabeldl, testdata, testlabeldl = creatdatadeeplearning(X, Y, iteration, K)

    # CNN-Bi-GRU
    resultslabel = CNNBiGRU(traindata, trainlabeldl, testdata, testlabeldl, TotalSequenceLength, times)

    # creat machine learning data
    trainfeature, trainlabelml, testfeature, testlabelml = creatdatamachinelearning(listfeature, iteration, K)

    # RF
    RFgroup, RFscore = RF(trainfeature, trainlabelml, testfeature)

    # fuzzy decision
    for rowfuz in range(resultslabel.shape[0]):
        if abs(resultslabel[rowfuz][0] - resultslabel[rowfuz][1]) < abs(RFscore[rowfuz][0] - RFscore[rowfuz][1]): # variable threshold
        # if abs(resultslabel[rowfuz][0] - resultslabel[rowfuz][1]) <= threshold: # constant threshold

            if strategy == 1: # Complete replacement strategy
                resultslabel[rowfuz][0] = RFscore[rowfuz][0]
                resultslabel[rowfuz][1] = RFscore[rowfuz][1]

            if strategy == 2: # Average weight strategy
                resultslabel[rowfuz][0] = resultslabel[rowfuz][0] * 0.5 + RFscore[rowfuz][0] * 0.5
                resultslabel[rowfuz][1] = resultslabel[rowfuz][1] * 0.5 + RFscore[rowfuz][1] * 0.5

            if strategy == 3: # Adaption weight strategy
                TPDL, FPDL, TNDL, FNDL, TPRDL, TNRDL, PPVDL, NPVDL, FNRDL, FPRDL, FDRDL, FORDL, ACCDL, F1DL, MCCDL, BMDL, MKDL = comparisondeeplearning(testlabeldl, resultslabel)
                TPML, FPML, TNML, FNML, TPRML, TNRML, PPVML, NPVML, FNRML, FPRML, FDRML, FORML, ACCML, F1ML, MCCML, BMML, MKML = comparisonmachinelearning(testlabeldl, RFgroup)
                wdl = (TPRDL + PPVDL + ACCDL)/(TPRDL + PPVDL + ACCDL + TPRML + PPVML + ACCML)
                wml = (TPRML + PPVML + ACCML)/(TPRDL + PPVDL + ACCDL + TPRML + PPVML + ACCML)
                resultslabel[rowfuz][0] = resultslabel[rowfuz][0] * wdl + RFscore[rowfuz][0] * wml
                resultslabel[rowfuz][1] = resultslabel[rowfuz][1] * wdl + RFscore[rowfuz][1] * wml

            ######################################################################################


    # obtain the results
    TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK = comparisondeeplearning(testlabeldl, resultslabel)
	
	# pring the results of each fold
    print('the', iteration + 1, 'fold')
    print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
    print('TPR:', TPR, 'TNR:', TNR, 'PPV:', PPV, 'NPV:', NPV, 'FNR:', FNR, 'FPR:', FPR, 'FDR:', FDR, 'FOR:', FOR)
    print('ACC:', ACC, 'F1:', F1, 'MCC:', MCC, 'BM:', BM, 'MK:', MK)

    # add the results
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

# obtain the average results
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

# pring the results
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
