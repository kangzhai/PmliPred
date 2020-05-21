# PmliPred for test

import numpy as np
import re
import math
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, TimeDistributed, RNN, Bidirectional, normalization
from keras import optimizers, regularizers
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
import argparse
# np.random.seed(1337)  # seed

parser = argparse.ArgumentParser(description="PmliPred for test")
args = parser.parse_args()

##########the parameter which can be adjusted#######################################################################################################
PlantName = 'Arabidopsis lyrata'  # plant species: 'Arabidopsis lyrata' or 'Solanum lycopersicum'
##########the parameter which can be adjusted#######################################################################################################

TotalSequenceLength = 0  # the total sequence length

# Load data
TrainSequencePath = 'Datasets\\Training-validation dataset\\Sequence.fasta' # raw sequence information for training
ListTrainSequence = open(TrainSequencePath, 'r').readlines()
TrainFeaturePath = 'Datasets\\Training-validation dataset\\Feature.fasta' # feature information for training
ListTrainFeature = open(TrainFeaturePath,'r').readlines()
TestSequencePath = 'Datasets\\Test dataset\\' + PlantName + '\\Sequence.fasta' # raw sequence information for test
ListTestSequence = open(TestSequencePath, 'r').readlines()
TestFeaturePath = 'Datasets\\Test dataset\\' + PlantName + '\\Feature.fasta' # feature information for test
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

# one-hot encoding
def onehot(list, TotalSequenceLength):
    onehotsequence = []
    onehotlabel = []
    ATCG = 'ATCG'  # alphabet
    char_to_int = dict((c, j) for j, c in enumerate(ATCG))  # set 'A': 0, 'T': 1, 'C': 2, 'G': 3

    for line in list:
        miRNAname, lncRNAname, sequence, label = line.split(',')
        sequence = sequence.upper()
        sequence = sequence.replace('U', 'T')

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
    Y = np_utils.to_categorical(Y, num_classes=2)

    return X, Y

# create deep learning data
def createdatadeeplearning(ListTrainSequence, ListTestSequence, TotalSequenceLength):

    Xtrain, Ytrain = onehot(ListTrainSequence, TotalSequenceLength)
    TrainDataDl = np.array(Xtrain)
    TrainLabelDl = np.array(Ytrain)
    Xtest, Ytest = onehot(ListTestSequence, TotalSequenceLength)
    TestDataDl = np.array(Xtest)
    TestLabelDl = np.array(Ytest)

    return TrainDataDl, TrainLabelDl, TestDataDl, TestLabelDl

# create machine learning data
def createdatamachinelearning(ListTrainFeature, ListTestFeature):

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

# CNN-BiGRU
def CNNBiGRU(TrainDataDl, TrainLabelDl, TestDataDl, TotalSequenceLength):

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

    # MaxPooling layer
    model.add(MaxPooling2D(pool_size=4, strides=4, padding='same', data_format='channels_last'))

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

    # BiGRU
    model.add(Bidirectional(GRU(units=64, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
                  bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
                  dropout=0, recurrent_dropout=0, implementation=1, return_sequences=False, return_state=False, go_backwards=False,
                  stateful=False, unroll=False, reset_after=False)))

    # Dropout layer
    model.add(Dropout(0.5))

    # fully-connected layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # optimizer
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # training
    print('Training --------------')
    model.fit(TrainDataDl, TrainLabelDl, epochs=10, batch_size=64, verbose=1)

    # get the confidence probability
    ResultsLabel = model.predict(TestDataDl)

    return ResultsLabel

# RF
def RF(TrainDataMl, TrainLabelMl, TestDataMl):

    RFStruct = ensemble.RandomForestClassifier()
    RFStruct.fit(TrainDataMl, TrainLabelMl) # training
    RFscore = RFStruct.predict_proba(TestDataMl) # get the confidence probability

    return RFscore

# create deep learning data
TrainDataDl, TrainLabelDl, TestDataDl, TestLabelDl = createdatadeeplearning(ListTrainSequence, ListTestSequence, TotalSequenceLength)

# create machine learning data
TrainDataMl, TrainLabelMl, TestDataMl = createdatamachinelearning(ListTrainFeature, ListTestFeature)

# CNN-BiGRU
ResultsLabel = CNNBiGRU(TrainDataDl, TrainLabelDl, TestDataDl, TotalSequenceLength)

# RF
RFscore = RF(TrainDataMl, TrainLabelMl, TestDataMl)

# fuzzy decision
FinaLabel = ResultsLabel
for rowfuz in range(FinaLabel.shape[0]):
    if abs(ResultsLabel[rowfuz][0] - ResultsLabel[rowfuz][1]) < abs(RFscore[rowfuz][0] - RFscore[rowfuz][1]): # variable threshold
        FinaLabel[rowfuz][0] = RFscore[rowfuz][0]
        FinaLabel[rowfuz][1] = RFscore[rowfuz][1]

# obtain AUC
print('AUC is ')
print(roc_auc_score(TestLabelDl, FinaLabel))
