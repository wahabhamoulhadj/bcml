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
df = pd.read_csv('../Tables/AUC_Table.csv')
# df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)
# df.set_index(0)
df = df.iloc[:, 8:]
print(df)
df.to_csv('../Tables/AUC_Table.csv', index=False)
# import numpy as np
#
# print(np.load('/Tables/all_six_models_predict_proba.npy', allow_pickle=True)[0])
