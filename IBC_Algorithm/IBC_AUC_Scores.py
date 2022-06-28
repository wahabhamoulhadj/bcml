import glob
import numpy as np
import pandas as pd
from IBC_Algorithm.IBC_ibcvr import ibcvr
from Tables.splitTable import split_table

path = "../PROMIS/CK/*.csv"
ibc_list = []
all_predict_proba = np.load('../Tables/all_six_models_predict_proba.npy', allow_pickle=True)
for count, fname in enumerate(glob.glob(path)):
    soft_detectors = all_predict_proba[count]
    ibc_list.append(ibcvr(soft_detectors, split_table(fname, 42)[3], 12)[2])
    print(count, ibc_list)

columns = ['IBC']
df = pd.DataFrame(ibc_list, columns=columns)

df.to_csv('AUC_Table_IBC.csv')
