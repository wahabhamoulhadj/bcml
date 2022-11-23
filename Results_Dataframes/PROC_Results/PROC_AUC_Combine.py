import glob
import os
import pandas as pd
df = pd.read_csv('PROC_all_six_models_auc.csv')
path = "../../PROMIS/PROC/*.csv"
l = []
for fname in glob.glob(path):
    l.append(os.path.basename(fname))
l = pd.Series(l)
PROC_BBC = pd.read_csv('PROC_BBC_AUC_Table.csv')
PROC_IBC = pd.read_csv('PROC_AUC_Table_IBC.csv')
PROC_PBC = pd.read_csv('PROC_PBC_AUC_Table.csv')
PROC_WPBC2 = pd.read_csv('PROC_WPBC2_AUC_Table.csv')
PROC_WPIBC = pd.read_csv('PROC_AUC_Table_WPIBC.csv')
df = pd.concat([df , PROC_BBC, PROC_IBC, PROC_PBC, PROC_WPBC2, PROC_WPIBC], axis=1, join='outer')

df = df.set_index(l)
df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

df.to_csv('PROC_all_auc.csv')
print(df)