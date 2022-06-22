import pandas as pd
df = pd.concat([pd.read_csv('../Tables/AUC_Table.csv'), pd.read_csv('AUC_Table_BBC.csv')], axis=1, join='outer' )
df.drop(df.filter(regex="Unnamed"),axis=1, inplace=True)
df.to_csv('../Tables/AUC_Table.csv', index=False)
