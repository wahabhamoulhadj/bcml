import pandas as pd
import seaborn
import matplotlib.pyplot as plt
df1 = pd.read_csv('CK_NET_Results/CK_NET_all_auc.csv').drop(columns=['Unnamed: 0'])
df2 = pd.read_csv('CK_PROC_Results/CK_PROC_all_auc.csv').drop(columns=['Unnamed: 0'])
df3 = pd.read_csv('CK_Results/CK_all_auc.csv').drop(columns=['Unnamed: 0'])
df4 = pd.read_csv('NET_Results/NET_all_auc.csv').drop(columns=['Unnamed: 0'])
df5 = pd.read_csv('NET_PROC_Results/NET_PROC_all_auc.csv').drop(columns=['Unnamed: 0'])
df6 = pd.read_csv('PROC_Results/PROC_all_auc.csv').drop(columns=['Unnamed: 0'])
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
frames = [df1, df2, df3, df4, df5, df6]

result = pd.concat(frames)

for i in range(len(result)):
    print(i,result.iloc[i].idxmax())
# plt.boxplot(result)
# plt.ylabel('AUC')
# plt.xticks([1,2,3,4,5,6,7,8,9,10,11], ['NB', 'RF',
#                                     'DT', 'LR', 'KNN', 'SVM', 'BBC', 'IBC', 'PBC', 'WPBC2', 'WPIBC'])
#
# plt.show()