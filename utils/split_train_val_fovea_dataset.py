import glob
import os

import pandas as pd


path = "/home/brani/STORAGE/DATA/refugee/"
table = pd.DataFrame()
excel_files = glob.glob(os.path.join(path, "fovea/*.xlsx"))

for excel_file in excel_files:
    table = pd.concat([table, pd.read_excel(excel_file)], ignore_index=True)

table = table.reindex()
table_train = table.sample(frac=0.85)
table_eval = table.sample(frac=0.15)
print(table_eval)
print(table_train)

os.makedirs(os.path.join(path, "fovea_train"), exist_ok=True)
os.makedirs(os.path.join(path, "fovea_eval"), exist_ok=True)

table_train.to_excel(os.path.join(path, "fovea_train", "locations.xlsx"))
table_eval.to_excel(os.path.join(path, "fovea_eval", "locations.xlsx"))