import glob
import os
import pandas as pd
df = pd.read_csv('Tables/AUC_Table.csv')
path = "PROMIS/CK/*.csv"
l = []
for fname in glob.glob(path):
    l.append(os.path.basename(fname))
l = pd.Series(l)
df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)
df = df.set_index(l)
df.to_csv('Results_Dataframes/CK_all_auc.csv')
print(df)