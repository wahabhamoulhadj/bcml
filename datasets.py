import glob

import numpy as np

from Tables.splitTable import split_table

path = "../PROMIS/CK/*.csv"
all_predict_proba = np.load('Tables/all_six_models_predict_proba.npy', allow_pickle=True)
for i in all_predict_proba:
    print('[' ,len(i),' x ',len(i[0]))