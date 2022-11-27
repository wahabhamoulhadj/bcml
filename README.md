This repository contains the datasets, results, and scripts used for the project "DeepTracing: Using AI to detect anomalies in large execution traces." We developed ensembles using Boolean Combination Algorithms(BCAs) and six heterogeneous Machine Learning (ML) classifiers to predict software defects.

We used five BCAs: BBC2, IBC, PBC2, WPBC2, and WPIBC.

We used six ML classifiers: Random Forest, Decision Tree, Logistic Regression, Naive Bayes, KNN, and SVM.

The repository is organised as follows:

1. PROMIS: Contains the datasets that we've used in this project. It contains six metrics: CK, CK_NET, CK_PROC, NET, NET_PROC, and CK_NET_PROC, each containing 27 datasets.

2. Results_Dataframes: Contains all the result dataframes and Reciever Operating Characteristics (ROC) curves.

3. Tables: Scripts for cross-validation and training for all ML classifiers are included.

4. BBC2_Algorithm: Contains the Python script related to Pair-wise Brute-force Boolean Combination (BBC2)

5. IBC_Algorithm : Contains the Python script related to Iterative Boolean Combination (IBC)

6. PBC2_Algorithm : Contains the Python script related to Pruning Boolean Combination (PBC)

7. WPBC2_Algorithm : Contains the Python script related to Weighted Pruning Boolean Combination (WPBC2)

8. WPIBC_Algorithms : Contains the Python script related to Weighted Iterative Boolean Combination (WPIBC)

9. The slides and other resources provide a high-level overview of the algorithms' internal workings and the end-to-end pipeline.


Authors: Mohammed A. Shehab, Venkata Sai Gunda, and Prof. Abdelwahab Hamou-Lhadj

Institutions: Concordia University and IIT Kharagpur




For any questions regarding the repository and/or the paper,

please contact:

Professor Wahab Hamou-Lhadj

wahab.hamou-lhadj@concordia.ca
