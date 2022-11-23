import glob
import os
import pandas as pd
df = pd.read_csv('CK_all_six_models_auc.csv')
path = "../../PROMIS/CK/*.csv"
l = []
for fname in glob.glob(path):
    l.append(os.path.basename(fname))
l = pd.Series(l)
CK_BBC = pd.read_csv('CK_BBC_AUC_Table.csv')
CK_IBC = pd.read_csv('CK_AUC_Table_IBC.csv')
CK_PBC = pd.read_csv('CK_PBC_AUC_Table.csv')
CK_WPBC2 = pd.read_csv('CK_WPBC2_AUC_Table.csv')
CK_WPIBC = pd.read_csv('CK_AUC_Table_WPIBC.csv')
df = pd.concat([df , CK_BBC, CK_IBC, CK_PBC, CK_WPBC2, CK_WPIBC], axis=1, join='outer')

df = df.set_index(l)
df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)

df.to_csv('CK_all_auc.csv')
print(df)