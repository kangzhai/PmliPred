# PmliPred for plant miRNA-lncRNA prediction

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
splant = 'osa' # plant species

# Load data
TrainSequencePath = 'data\\' + splant + '\\Training-test dataset.fasta' # raw sequence information for training
ListTrainSequence = open(TrainSequencePath, 'r').readlines()
TrainFeaturePath = 'data\\' + splant + '\\Training-test feature.fasta' # feature information for training
ListTrainFeature = open(TrainFeaturePath,'r').readlines()
TestSequencePath = 'data\\' + splant + '\\Validation dataset.fasta' # raw sequence information for validation
ListTestSequence = open(TestSequencePath, 'r').readlines()
TestFeaturePath = 'data\\' + splant + '\\Validation feature.fasta' # feature information for validation
ListTestFeature = open(TestFeaturePath,'r').readlines()

# Get the maximum length of the sequence
for linelength1 in ListTrainSequence:
    miRNAname, lncRNAname, sequence, label = linelength1.split(',')
    if len(sequence) > TotalSequenceLength:
        TotalSequenceLength = len(sequence)
for linelength2 in ListTestSequence:
    miRNAname, lncRNAname, sequence, label = linelength2.split(',')
    if len(sequence) > TotalSequenceLength:
        TotalSequenceLength = len(sequence)

epo = 10 # epochs
dp = 0.5 # Dropout rate
threshold = 0.5 # threshold
strategy = 1 # weight strategy selection 1: Complete weight 2: Average weight

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
def creatdatadeeplearning(ListTrainSequence, ListTestSequence, TotalSequenceLength):

	Xtrain, Ytrain = onehot(ListTrainSequence, TotalSequenceLength)
	TrainDataDl = np.array(Xtrain)
	TrainLabelDl = np.array(Ytrain)
	Xtest, Ytest = onehot(ListTestSequence, TotalSequenceLength)
	TestDataDl = np.array(Xtest)
	TestLabelDl = np.array(Ytest)

    return TrainDataDl, TrainLabelDl, TestDataDl, TestLabelDl

# creat machine learning data
def creatdatamachinelearning(ListTrainFeature, ListTestFeature):

    # separate the label
    rowtraindata = len(ListTrainFeature)
    columntraindata = len(ListTrainFeature[0].split()) - 1
    rowtestdata = len(ListTestFeature)
    columntestdata = len(ListTestFeature[0].split()) - 1
	
	# get the training data and label
    TrainDataMl = [([0] * columntraindata) for p in range(rowtraindata)]
    TrainLabelMl = [([0] * 1) for p in range(rowtraindata)]
    for linetraindata in ListTrainFeature:
        setraindata = re.split(r'\s', linetraindata)
        indextraindata = ListTrainFeature.index(linetraindata)
        for itraindata in range(len(setraindata) - 1):
            if itraindata < len(setraindata) - 2:
                TrainDataMl[indextraindata][itraindata] = float(setraindata[itraindata])
            else:
                TrainLabelMl[indextraindata][0] = float(setraindata[itraindata])
	
	# get the validation data and label
    TestDataMl = [([0] * columntestdata) for p in range(rowtestdata)]
    for linetestdata in ListTestFeature:
        setestdata = re.split(r'\s', linetestdata)
        indextestdata = ListTestFeature.index(linetestdata)
        for itestdata in range(0, len(setestdata) - 1):
            if itestdata < len(setestdata) - 2:
                TestDataMl[indextestdata][itestdata] = float(setestdata[itestdata])

    return TrainDataMl, TrainLabelMl, TestDataMl

# CNN-Bi-GRU
def CNNBiGRU(TrainDataDl, TrainLabelDl, TestDataDl, TotalSequenceLength, epo):

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
    model.fit(TrainDataDl, TrainLabelDl, epochs = epo, batch_size = 64, verbose = 1)

    # get the confidence coefficient
    ResultLabel = model.predict(TestDataDl)

    return ResultLabel

# RF
def RF(TrainDataMl, TrainLabelMl, TestDataMl):

    RFStruct = ensemble.RandomForestClassifier()
    RFStruct.fit(TrainDataMl, TrainLabelMl) # training
    RFscore = RFStruct.predict_proba(TestDataMl) # get the confidence coefficient

    return RFscore

# calculate the results
def comparison(TestLabelDl, FinaLabel):

    # initialization
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # formatting
    for row1 in range(FinaLabel.shape[0]):
        for column1 in range(FinaLabel.shape[1]):
            if FinaLabel[row1][column1] < 0.5:
                FinaLabel[row1][column1] = 0
            else:
                FinaLabel[row1][column1] = 1

    # TP, FP, TN, FN
    for row2 in range(TestLabelDl.shape[0]):
        # TP
        if TestLabelDl[row2][0] == 0 and TestLabelDl[row2][1] == 1 and TestLabelDl[row2][0] == FinaLabel[row2][0] and TestLabelDl[row2][1] == FinaLabel[row2][1]:
            TP = TP + 1
        # FP
        if TestLabelDl[row2][0] == 1 and TestLabelDl[row2][1] == 0 and TestLabelDl[row2][0] != FinaLabel[row2][0] and TestLabelDl[row2][1] != FinaLabel[row2][1]:
            FP = FP + 1
        # TN
        if TestLabelDl[row2][0] == 1 and TestLabelDl[row2][1] == 0 and TestLabelDl[row2][0] == FinaLabel[row2][0] and TestLabelDl[row2][1] == FinaLabel[row2][1]:
            TN = TN + 1
        # FN
        if TestLabelDl[row2][0] == 0 and TestLabelDl[row2][1] == 1 and TestLabelDl[row2][0] != FinaLabel[row2][0] and TestLabelDl[row2][1] != FinaLabel[row2][1]:
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

# creat deep learning data
TrainDataDl, TrainLabelDl, TestDataDl, TestLabelDl = creatdatadeeplearning(ListTrainSequence, ListTestSequence, TotalSequenceLength)

# creat machine learning data
TrainDataMl, TrainLabelMl, TestDataMl = creatdatamachinelearning(ListTrainFeature, ListTestFeature)

# CNN-Bi-GRU
ResultsLabel = CNNBiGRU(TrainDataDl, TrainLabelDl, TestDataDl, TotalSequenceLength, epo)

# RF
RFscore = RF(TrainDataMl, TrainLabelMl, TestDataMl)

# fuzzy decision
FinaLabel = ResultsLabel
for rowfuz in range(FinaLabel.shape[0]):
    if abs(ResultLabel[rowfuz][0] - ResultLabel[rowfuz][1]) < abs(RFscore[rowfuz][0] - RFscore[rowfuz][1]): # variable threshold
    # if abs(ResultLabel[rowfuz][0] - ResultLabel[rowfuz][1]) <= threshold: # constant threshold
        if strategy == 1: # Complete weight strategy
            FinaLabel[rowfuz][0] = RFscore[rowfuz][0]
            FinaLabel[rowfuz][1] = RFscore[rowfuz][1]
        if strategy == 2: # Average weight strategy
            FinaLabel[rowfuz][0] = ResultLabel[rowfuz][0] * 0.5 + RFscore[rowfuz][0] * 0.5
            FinaLabel[rowfuz][1] = ResultLabel[rowfuz][1] * 0.5 + RFscore[rowfuz][1] * 0.5

# obtain the results
TP, FP, TN, FN, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK = comparison(TestLabelDl, FinaLabel)
	
# print the results of each fold
print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
print('TPR:', TPR, 'TNR:', TNR, 'PPV:', PPV, 'NPV:', NPV, 'FNR:', FNR, 'FPR:', FPR, 'FDR:', FDR, 'FOR:', FOR)
print('ACC:', ACC, 'F1:', F1, 'MCC:', MCC, 'BM:', BM, 'MK:', MK)


