import pandas as pd

all_import_hs = pd.read_parquet("datasets/all_hs_import.parquet")
all_export_hs = pd.read_parquet("datasets/all_hs_export.parquet")


with open("run_import.sh", "w") as f:
    for i in all_import_hs["HS_Code"]:
        f.write(f"python main.py {i} M\n")

with open("run_export.sh", "w") as f:
    for i in all_export_hs["HS_Code"]:
        f.write(f"python main.py {i} X\n")
