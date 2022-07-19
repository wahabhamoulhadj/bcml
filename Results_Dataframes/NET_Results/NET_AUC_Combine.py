import glob
import os
import pandas as pd
df = pd.read_csv('NET_all_six_models_auc.csv')
path = "../../PROMIS/NET/*.csv"
l = []
for fname in glob.glob(path):
    l.append(os.path.basename(fname))
l = pd.Series(l)
NET_BBC = pd.read_csv('NET_BBC_AUC_Table.csv')
NET_IBC = pd.read_csv('NET_AUC_Table_IBC.csv')
NET_WPBC2 = pd.read_csv('NET_WPBC2_AUC_Table.csv')
NET_WPIBC = pd.read_csv('NET_AUC_Table_WPIBC.csv')
df = pd.concat([df , NET_BBC, NET_IBC, NET_WPBC2, NET_WPIBC], axis=1, join='outer')

df = df.set_index(l)
df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

df.to_csv('NET_all_auc.csv')
print(df)