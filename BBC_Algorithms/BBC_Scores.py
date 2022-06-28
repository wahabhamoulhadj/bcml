import glob
import numpy as np
import pandas as pd
from BBC_Algorithms.auc_bbc_algo import auc_bbc
from Tables.splitTable import split_table

path = "../PROMIS/CK/*.csv"
all_predict_proba = np.load('../Tables/all_six_models_predict_proba.npy', allow_pickle=True)
random_seed = 42
bbc_list = []
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    bbc_list.append(auc_bbc(split_table(fname, random_seed)[3], soft_detectors, 12))
    print(count, bbc_list)

columns = ['BBC']
df = pd.DataFrame(bbc_list, columns=columns)

df.to_csv('AUC_Table_BBC.csv')






