# Bitcoin Fraudulent Account Detection

The work of this project was done in conjunction with Cristina Susanu (CristinBSE) and Andres Virguez (andresvBSE) as the final project for the Masters in Data Science for Decision Making at BSE. 

Financial institutions are starting to integrate their digital payment systems with with cryptocurrency markets. A challenge that these institutions face when operating in crypto environments consists in avoiding carrying out transactions with accounts that are involved with fraudulent activities (drug dealing, terrorism, ransomware, etc.).

In light of this challenge, a number of credit rating agencies have started to provide fraudulent scores of Bitcoin addresses to help financial institutions decide whether they should engage in a transaction with a given address or not.

The objective of this master thesis is to collaborate with cryptocurrency security firm, Clovrlabs, in the development of risk measurement models for Bitcoin addresses. Specifically, the project consists in sampling the Bitcoin transactions network, extracting covariates associated with Bitcoin addresses via graph embeddings from the bitcoin blockchain using two embedding methods (Node2Vec and Trans2Vec), adding non-embedding features and then to use these covariates to predict fraudulent addresses using a sample of labelled addresses.

This repository contains
1. Our data analysis,
2. The sampling methodology, and
3. The classification pipeline

For reference:

Repository with the provided data: https://github.com/LlucPuigCodina/bse_clovrlabs_btc_fraud

Origional GitHub repository of the Trans2Vec paper: https://github.com/wuzhy1ng/Phishing-Detection-on-Ethereum (code provided in folder above, adjustments required)
