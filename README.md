# PmliPred
The related datasets and scoure codes of PmliPred privided by Q. Kang

The latest version is updated on 2019.12.09.

# Introduction
PmliPred is a method for plant miRNA-lncRNA interaction prediction based on hybrid model and fuzzy decision. It is implemented by Keras  and all main scripts are written by Python on PC under a Microsoft Windows 10 operating system.

The repository can be downloaded locally by clicking "clone or download" button. PmliPred can be applied directly without installation. 

# Dependency
python 3.6.5

Keras 2.2.4

# Datasets
The example datasets can be obtained by unzipping "Datasets.zip".

"Training-validation dataset" file includes the raw sequences and manually features of the samples for model training and validation. 

"Test dataset" file includes the raw sequences and manually features of the samples for model test.

"miR399-lnc1077" file includes the raw sequence and manually features of the interaction between miRNA (miR399) and lncRNA (lnc1077) in solanum lycopersicum. 

"miR482b-TCONS_00023468" file includes the raw sequence and manually features of the interaction between miRNA (miR482b) and lncRNA (TCONS_00023468) in solanum lycopersicum.

# Feature description
Feature description.xlsx lists the number of extracted features of each miRNA-lncRNA interaction sample, where 23 features are from miRNA and the other 87 features are from lncRNA.

Install-ViennaRNA-2.4.10_64bit.exe is the installation of ViennaRNA parckage that contains the RNA sequence secondary structure extraction tool RNAfold. It can be also downloaded from https://www.tbi.univie.ac.at/RNA/.

# Usage



# Reference
Wait for update
