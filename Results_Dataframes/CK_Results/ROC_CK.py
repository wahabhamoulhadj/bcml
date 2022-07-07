import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('CK.csv')
indexs =df['Unnamed: 0']
df.drop(df.filter('Unnamed: 0'), axis=1, inplace=True)

roc_curves_tpr_list = np.load('../../Tables/all_six_models_tpr.npy', allow_pickle=True)
roc_curves_fpr_list = np.load('../../Tables/all_six_models_fpr.npy', allow_pickle=True)
bbc_fpr = np.load('../../BBC_Algorithms/bbc_roc_fpr.npy', allow_pickle=True)
bbc_tpr = np.load('../../BBC_Algorithms/bbc_roc_tpr.npy', allow_pickle=True)
bbc_tpr = np.array(bbc_tpr)
bbc_fpr = np.array(bbc_fpr)

ibc_fpr = np.load('../../IBC_Algorithm/ibc_roc_fpr.npy', allow_pickle=True)
ibc_tpr = np.load('../../IBC_Algorithm/ibc_roc_tpr.npy', allow_pickle=True)
print(df)

for count in range(len(df)):
    plt.figure()

    plt.plot(roc_curves_fpr_list[count][0], roc_curves_tpr_list[count][0], label="Naive Bayes")
    plt.plot(roc_curves_fpr_list[count][1], roc_curves_tpr_list[count][1], label="RandomForest")
    plt.plot(roc_curves_fpr_list[count][2], roc_curves_tpr_list[count][2], label="DecisionTree")
    plt.plot(roc_curves_fpr_list[count][3], roc_curves_tpr_list[count][3], label="LogisticRegression")
    plt.plot(roc_curves_fpr_list[count][4], roc_curves_tpr_list[count][4], label="KNN")
    plt.plot(roc_curves_fpr_list[count][5], roc_curves_tpr_list[count][5], label="SVM")
    plt.plot(bbc_fpr[count], bbc_tpr[count], label="BBC2", marker = 's')
    plt.plot(ibc_fpr[count], ibc_tpr[count], label="IBC", marker = 'o')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(["Naive Bayes AUC : {}".format(df.iloc[count][1]),
                "Random Forest AUC : {}".format(df.iloc[count][2]),
                "Decision Tree AUC : {}".format(df.iloc[count][3]),
                "Logistic Regression AUC : {}".format(df.iloc[count][4]),
                "KNN AUC : {} ".format(df.iloc[count][5]),
                "SVC AUC : {} ".format(df.iloc[count][6]),
                "BBC2 AUC : {} ".format(pd.read_csv('../../BBC_Algorithms/AUC_Table_BBC.csv').iloc[count][1]),
                "IBC AUC : {} ".format(pd.read_csv('../../IBC_Algorithm/AUC_Table_IBC.csv').iloc[count][1], loc='lower right')])


    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(indexs[count])
    g = indexs[count][:-4]
    plt.savefig("CK_ROC_Curves/ {}.png".format(g), format='png')

    # plt.show()








