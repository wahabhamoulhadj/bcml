import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

df = pd.read_csv('PROC_all_six_models_auc.csv')
df1 = pd.read_csv('PROC_all_auc.csv')
indexs = df1['Unnamed: 0']

df.drop(df.filter('Unnamed: 0'), axis=1, inplace=True)
df1.drop(df1.filter('Unnamed: 0'), axis=1, inplace=True)
print(df1)
roc_curves_tpr_list = np.load('PROC_all_six_models_tpr.npy', allow_pickle=True)
roc_curves_fpr_list = np.load('PROC_all_six_models_fpr.npy', allow_pickle=True)
bbc_fpr = np.load('PROC_bbc_roc_fpr.npy', allow_pickle=True)
bbc_tpr = np.load('PROC_bbc_roc_tpr.npy', allow_pickle=True)
# bbc_tpr = np.array(bbc_tpr)
# bbc_fpr = np.array(bbc_fpr)

ibc_fpr = np.load('PROC_ibc_roc_fpr.npy', allow_pickle=True)
ibc_tpr = np.load('PROC_ibc_roc_tpr.npy', allow_pickle=True)

pbc_fpr = np.load('PROC_pbc_roc_fpr.npy', allow_pickle=True)
pbc_tpr = np.load('PROC_pbc_roc_tpr.npy', allow_pickle=True)

wpibc_fpr = np.load('PROC_wpibc_roc_fpr.npy', allow_pickle=True)
wpibc_tpr = np.load('PROC_wpibc_roc_tpr.npy', allow_pickle=True)

wpbc2_fpr = np.load('PROC_WPBC2_roc_fpr.npy', allow_pickle=True)
wpbc2_tpr = np.load('PROC_WPBC2_roc_tpr.npy', allow_pickle=True)

# print(df)

for count in range(len(df)):
    plt.figure()

    plt.plot(roc_curves_fpr_list[count][0], roc_curves_tpr_list[count][0], label="Naive Bayes")
    plt.plot(roc_curves_fpr_list[count][1], roc_curves_tpr_list[count][1], label="RandomForest")
    plt.plot(roc_curves_fpr_list[count][2], roc_curves_tpr_list[count][2], label="DecisionTree")
    plt.plot(roc_curves_fpr_list[count][3], roc_curves_tpr_list[count][3], label="LogisticRegression")
    plt.plot(roc_curves_fpr_list[count][4], roc_curves_tpr_list[count][4], label="KNN")
    plt.plot(roc_curves_fpr_list[count][5], roc_curves_tpr_list[count][5], label="SVM")
    plt.plot(bbc_fpr[count], bbc_tpr[count], label="BBC2", marker='s')
    plt.plot(ibc_fpr[count], ibc_tpr[count], label="IBC", marker='o')
    plt.plot(pbc_fpr[count], pbc_tpr[count], label="PBC", marker='o')
    plt.plot(wpbc2_fpr[count], wpbc2_tpr[count], label="WPBC2", marker='D')
    plt.plot(wpibc_fpr[count], wpibc_tpr[count], label="WPIBC", marker='^')
    plt.plot([0, 1], [0, 1], 'k--')

    plt.legend(["Naive Bayes AUC : {:.3f}".format(df.iloc[count][1]),
                "Random Forest AUC : {:.3f}".format(df.iloc[count][2]),
                "Decision Tree AUC : {:.3f}".format(df.iloc[count][3]),
                "Logistic Regression AUC : {:.3f}".format(df.iloc[count][4]),
                "KNN AUC : {:.3f} ".format(df.iloc[count][5]),
                "SVC AUC : {:.3f} ".format(df.iloc[count][6]),
                "BBC2 AUC : {:.3f} ".format(pd.read_csv('PROC_BBC_AUC_Table.csv').iloc[count][1]),
                "IBC AUC : {:.3f} ".format(pd.read_csv('PROC_AUC_Table_IBC.csv').iloc[count][1]),
                "PBC AUC : {:.3f} ".format(pd.read_csv('PROC_PBC_AUC_Table.csv').iloc[count][1]),
                "WPBC2 AUC : {:.3f} ".format(pd.read_csv('PROC_WPBC2_AUC_Table.csv').iloc[count][1]),
                "WPIBC AUC : {:.3f} ".format(pd.read_csv('PROC_AUC_Table_WPIBC.csv').iloc[count][1]),], loc='lower right')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(indexs[count])
    g = indexs[count][:-4]
    plt.savefig(r"PROC_ROC_Curves/ {}.png".format(g), format='png')

    # plt.show()
# directory = r'PROC_ROC_Curves'
# for filename in os.listdir(directory):
#     os.rename(filename, filename+".png")