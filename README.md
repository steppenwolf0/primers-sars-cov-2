# Code and data for "Classification and Specific Primer Design for Accurate Detection of SARS-CoV-2 Using Deep Learning"

This repository contains the code of the scripts, the information to retrieve the data, and the instructions to reproduce the experiments for the paper "Classification and Specific Primer Design for Accurate Detection of SARS-CoV-2 Using Deep Learning", currently submitted to Nature Scientific Reports.

## Files description
- Instructions.docx : all the necessary instructions to run the experiments presented in the paper.
- sample_IDs : all IDs of the virus samples used in the experiments (storing the data would have required GBs).
- candidates.csv : sequences obtained after training the Convolutional Neural Network (CNN) and performing feature selection. These are candidate primers for SARS-CoV-2.
- CNN/ : folder containing code and data to run the CNN classification in a cross-validation.
- data_1503_featureSpace/ : folder with the code to run the validation.
- Primer3Plus/ : folder with the code to run validation of candidate primers using the web application Primer3Plus.

## Original manuscript and supplementary information
Once the paper will finish the review process, link to the paper and its supplementary materials will be displayed here. For the moment, here is a link to the BiorXiv version of the paper: https://www.biorxiv.org/content/10.1101/2020.03.13.990242v5
