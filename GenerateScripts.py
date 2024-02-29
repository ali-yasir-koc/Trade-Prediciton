########### GENERATE SCRIPT ################
# These codes are used to prepare sh files.
# Two empty sh files that already exist are filled by using Hs Codes in scope.
# These sh file contain main python codes in this project for HS Code and trade type pairs.
# M represents imports and X represents exports.

###############################
# Import Library and Settings
###############################
import pandas as pd


###############################
# Loading Data
###############################
all_import_hs = pd.read_parquet("datasets/all_hs_import.parquet")
all_export_hs = pd.read_parquet("datasets/all_hs_export.parquet")


###############################
# Fill Sh Files
###############################
with open("run_import.sh", "w") as f:
    for i in all_import_hs["HS_Code"]:
        f.write(f"python main.py {i} M\n")

with open("run_export.sh", "w") as f:
    for i in all_export_hs["HS_Code"]:
        f.write(f"python main.py {i} X\n")
