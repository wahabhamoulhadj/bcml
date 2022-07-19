import glob
import os
import pandas as pd
df = pd.read_csv('CK_NET_all_six_models_auc.csv')
path = "../../PROMIS/CK_NET/*.csv"
l = []
for fname in glob.glob(path):
    l.append(os.path.basename(fname))
l = pd.Series(l)
CK_NET_BBC = pd.read_csv('CK_NET_BBC_AUC_Table.csv')
CK_NET_IBC = pd.read_csv('CK_NET_AUC_Table_IBC.csv')
CK_NET_WPBC2 = pd.read_csv('CK_NET_WPBC2_AUC_Table.csv')
CK_NET_WPIBC = pd.read_csv('CK_NET_AUC_Table_WPIBC.csv')
df = pd.concat([df , CK_NET_BBC, CK_NET_IBC, CK_NET_WPBC2, CK_NET_WPIBC], axis=1, join='outer')

df = df.set_index(l)
df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

df.to_csv('CK_NET_all_auc.csv')
print(df)