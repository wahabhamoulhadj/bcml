import numpy as np
import pandas as pd

# df = pd.read_csv('../Tables/AUC_Table.csv')
# l = pd.Series(['project_' + str(i + 1) for i in range(27)])
# df = pd.concat([l, df], axis=1, join='outer')
# df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)
# df.set_index(0)
#
# df.to_csv('../Tables/AUC_Table.csv', index=False)

# import pandas as pd
# df = pd.read_csv('../Tables/AUC_Table.csv')
# l = pd.Series(['project_' + str(i+1) for i in range(27)])
# df = pd.concat([l , df], axis=1, join='outer' )
# df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
# df.set_index(0)
#
# df.to_csv('../Tables/AUC_Table.csv', index=False)
#
# import pandas as pd
#
from matplotlib import pyplot as plt

df = pd.read_csv('../Tables/AUC_Table.csv')
df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

print(df.iloc[0][1])
print(pd.read_csv('AUC_Table_BBC.csv').iloc[0][1])
# df.set_index(0)
# df = df.iloc[:, 8:]
# print(df)
# df.to_csv('../Tables/AUC_Table.csv', index=False)
import numpy as np
#
roc_curves_tpr_list = np.load('../Tables/all_six_models_tpr.npy', allow_pickle=True)
roc_curves_fpr_list = np.load('../Tables/all_six_models_fpr.npy', allow_pickle=True)
bbc_fpr = np.load('bbc_roc_fpr.npy', allow_pickle=True)
bbc_tpr = np.load('bbc_roc_tpr.npy', allow_pickle=True)
bbc_tpr = np.array(bbc_tpr)
bbc_fpr = np.array(bbc_fpr)
for count in range(4):
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(roc_curves_fpr_list[count][0], roc_curves_tpr_list[count][0], label="Naive Bayes")
    plt.plot(roc_curves_fpr_list[count][1], roc_curves_tpr_list[count][1], label="RandomForest")
    plt.plot(roc_curves_fpr_list[count][2], roc_curves_tpr_list[count][2], label="DecisionTree")
    plt.plot(roc_curves_fpr_list[count][3], roc_curves_tpr_list[count][3], label="LogisticRegression")
    plt.plot(roc_curves_fpr_list[count][4], roc_curves_tpr_list[count][4], label="KNN")
    plt.plot(roc_curves_fpr_list[count][5], roc_curves_tpr_list[count][5], label="SVM")
    plt.plot(bbc_fpr[count], bbc_tpr[count], label="BBC2")
    plt.legend(["Naive Bayes AUC : {}".format(df.iloc[count][0]),
                "Random Forest AUC : {}".format(df.iloc[count][1]),
                "Decision Tree AUC : {}".format(df.iloc[count][2]),
                "Logistic Regression AUC : {}".format(df.iloc[count][3]),
                "KNN AUC{} : ".format(df.iloc[count][4]),
                "SVC AUC : {} ".format(df.iloc[count][5]),
                "BBC2 AUC : {} ".format(pd.read_csv('AUC_Table_BBC.csv').iloc[count][1])], loc='lower right')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('Receiver Operating Characteristic')
    plt.show()








