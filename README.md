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

"Training-validation dataset" includes the raw sequences and manually features of the samples for model training and validation. 

"Test dataset" includes the raw sequences and manually features of the samples for model test.

"miR399-lnc1077" includes the raw sequence and manually features of the interaction between miRNA (miR399) and lncRNA (lnc1077) in solanum lycopersicum. 

"miR482b-TCONS_00023468" includes the raw sequence and manually features of the interaction between miRNA (miR482b) and lncRNA (TCONS_00023468) in solanum lycopersicum.

# Feature description
Feature description.xlsx lists the number of extracted features of each miRNA-lncRNA interaction sample, where 23 features are from miRNA and the other 87 features are from lncRNA.

# Usage
(1) perform k-fold cross validation

python PmliPredForCrossValidation.py

Explanation: there are four parameters can be adjusted manually, where "K" means K-fold cross validation, "WeightStrategy = 1" means the complete weight is selected and "WeightStrategy = 2" means the average weight is selected, "ThresholdStrategy = 1" means the variable threshold is selected and "ThresholdStrategy = 2" means the constant threshold is selected, "threshold" is the value of threshold and it can be used just on canstant threshold strategy. The outputs are the results of each fold and the average results of K folds.

(2) test PmliPred

python PmliPredForTest.py

Explanation: there is just one parameter can be adjusted manually, where "PlantName" means the test species. Here we provide two selections of the species, such as 'Arabidopsis lyrata' or 'Solanum lycopersicum'. The output it the AUC value.

(3) predict plant miRNA-lncRNA interaction

python PmliPredForPrediction.py

Explanation: there are two parameters can be adjusted manually, where "InteractionName" means the name of miRNA-lncRNA interaction, "IndependentTimes" means the times of independent prediction. Here we provide two selections of the name of miRNA-lncRNA interaction, such as 'miR482b-TCONS_00023468' or 'miR399-lnc1077'. The outputs are the confidence probability that there is an interaction in the sample (pc) of each time of independent prediction and the average pc value of IndependentTimes times of predictions

# Note
The above three files can be opened using python IDE, such as pyCharm and so on. They can be also opened as the txt or fasta files. Then the parameters can be adjusted manually. We will try to make these parameters directly available as the input in future updates.

"PmliPredForPridiction.py" can be also used to predict a new plant miRNA-lncRNA interaction. The input must be the miRNA-lncRNA sequence and manually features at this version. Here k-mer frequency and GC content are extracted by Python scripts, and number of base pairs and minimum free energy are extracted by Python scripts and RNAfold in ViennaRNA package. Install-ViennaRNA-2.4.10_64bit.exe is the installation of ViennaRNA package that contains RNAfold (a RNA secondary structure extraction tool). ViennaRNA package can be also downloaded from https://www.tbi.univie.ac.at/RNA/. Next we will organize these Python scripts to minimize the user's work.

# Reference
Wait for update
