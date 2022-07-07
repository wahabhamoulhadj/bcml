import os
directory = r'CK_PROC_ROC_Curves'
for filename in os.listdir(directory):
    os.rename(filename, filename+".png")