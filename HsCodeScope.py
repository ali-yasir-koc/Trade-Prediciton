import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

############################
# Import Data HS-Code Scope
#############################
df_M = pd.read_parquet("datasets/comtrade_yearly_import.parquet", engine ="pyarrow")
df_M.head()

df_M = df_M.sort_values(["period", "primaryValue"], ascending = False)
df_M = df_M[df_M["cmdCode"] != "999999"].reset_index(drop = True)
df_M["cumsum"] = df_M.groupby("period")["primaryValue"].cumsum()
df_M["total_all"] = df_M.groupby("period")["primaryValue"].transform("sum")
df_M["rate"] = (df_M["cumsum"] / df_M["total_all"]) * 100
df_M.loc[df_M["rate"] <= 80.0].groupby("period")["cmdCode"].count()
df_M.head()

cmdCode_dict = {}

for year in range(2016, 2023):
    condition = (df_M["period"] == year) & (df_M["rate"] <= 80.0)
    cmdCode_values = df_M.loc[condition, "cmdCode"].tolist()
    cmdCode_dict[f"cmdCode_list_{year}"] = cmdCode_values

for i in range(2016, 2023):
    current_year_key = f"cmdCode_list_{i}"
    next_year_key = f"cmdCode_list_{i + 1}" if i + 1 <= 2022 else None

    if next_year_key:
        difference_length = len(set(cmdCode_dict[current_year_key]).difference(set(cmdCode_dict[next_year_key])))
        print(f"Difference between {current_year_key} and {next_year_key}: {difference_length}")

all_hs = []
for i in cmdCode_dict.keys():
    all_hs.extend(cmdCode_dict[i])

all_hs = list(set(all_hs))
len(all_hs)

all_hs_import = pd.DataFrame({"HS_Code": all_hs})
all_hs_import.to_parquet("datasets/all_hs_import.parquet", engine = "pyarrow")

############################
# Export Data HS-Code Scope
#############################
df_X = pd.read_parquet("datasets/comtrade_yearly_export.parquet", engine = "pyarrow")
df_X.head()

df_X = df_X.sort_values(["period", "primaryValue"], ascending = False)
df_X = df_X[df_X["cmdCode"] != "999999"].reset_index(drop=True)
df_X["cumsum"] = df_X.groupby("period")["primaryValue"].cumsum()
df_X["total_all"] = df_X.groupby("period")["primaryValue"].transform("sum")
df_X["rate"] = (df_X["cumsum"] / df_X["total_all"]) * 100
df_X.loc[df_X["rate"] <= 80.0].groupby("period")["cmdCode"].count()
df_X.head()

cmdCode_dict = {}

for year in range(2016, 2023):
    condition = (df_X["period"] == year) & (df_X["rate"] <= 80.0)
    cmdCode_values = df_X.loc[condition, "cmdCode"].tolist()
    cmdCode_dict[f"cmdCode_list_{year}"] = cmdCode_values

for i in range(2016, 2023):
    current_year_key = f"cmdCode_list_{i}"
    next_year_key = f"cmdCode_list_{i + 1}" if i + 1 <= 2022 else None

    if next_year_key:
        difference_length = len(set(cmdCode_dict[current_year_key]).difference(set(cmdCode_dict[next_year_key])))
        print(f"Difference between {current_year_key} and {next_year_key}: {difference_length}")

all_hs = []
for i in cmdCode_dict.keys():
    all_hs.extend(cmdCode_dict[i])

all_hs = list(set(all_hs))
len(all_hs)

all_hs_export = pd.DataFrame({"HS_Code": all_hs})
all_hs_export.to_parquet("datasets/all_hs_export.parquet", engine ="pyarrow")