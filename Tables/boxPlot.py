import pandas as pd
df = pd.concat([pd.read_csv('../Tables/AUC_Table.csv'),pd.read_csv('../BBC_Algorithms/AUC_Table_BBC.csv') ,pd.read_csv('../IBC_Algorithm/AUC_Table_IBC.csv')], axis=1, join='outer' )
df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
print(df)
df.to_csv('../Tables/AUC_Table.csv', index=False)