########### DETERMINE HS CODE SCOPE ################
# These codes are used to determine the Hs Codes that will be included in the scope of the project.
# For this purpose, import and export data on an annual basis are used.
# We have set 80 percent of the annual trade volume as threshold.
# We will cover the products above this limit.
# For this purpose, we started by ranking the products in descending order according to their annual trade volume.
# We took the second step by collecting the transaction volumes of the products cumulatively on an annual basis.
# Then we calculated the total transaction volume of each year.
# We continued the transactions by dividing the cumulative total by the total volume.
# Thus, we calculated the 80 percent limit we wanted.
# The cmdCode in the data represents the HS code. Those with HS code 99999 are outside the scope of trade.


###############################
# Import Library and Settings
###############################
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


###############################
# Data Loading
###############################
df_M = pd.read_parquet("datasets/comtrade_yearly_import.parquet", engine ="pyarrow")
df_X = pd.read_parquet("datasets/comtrade_yearly_export.parquet", engine = "pyarrow")
df_M.head()
df_X.head()


############################
# Scope Function
#############################
def determine_scope(dataframe, trade_type, threshold = 80.0):
    """
    It takes the annual data and applies some filters.
    It finds out which products are above the specified trade volume threshold.
    Args:
        threshold: limit value for the scope
        dataframe: data including annual trade data
        trade_type: export or import

    Returns: prints the length of the list containing hs cods
             saves the list containing hs cods
    """
    dataframe = dataframe.sort_values(["period", "primaryValue"], ascending = False)
    dataframe = dataframe[dataframe["cmdCode"] != "999999"].reset_index(drop = True)
    dataframe["cumsum"] = dataframe.groupby("period")["primaryValue"].cumsum()
    dataframe["total_all"] = dataframe.groupby("period")["primaryValue"].transform("sum")
    dataframe["rate"] = (dataframe["cumsum"] / dataframe["total_all"]) * 100

    cmdCode_dict = {}
    for year in range(2016, 2023):
        condition = (dataframe["period"] == year) & (dataframe["rate"] <= threshold)
        cmdCode_values = dataframe.loc[condition, "cmdCode"].tolist()
        cmdCode_dict[f"cmdCode_list_{year}"] = cmdCode_values

    all_hs = []
    for i in cmdCode_dict.keys():
        all_hs.extend(cmdCode_dict[i])

    all_hs = list(set(all_hs))
    globals()[f"all_hs_{trade_type}"] = pd.DataFrame({"HS_Code": all_hs})
    globals()[f"all_hs_{trade_type}"].to_parquet(f"datasets/all_hs_{trade_type}.parquet", engine = "pyarrow")

    print(len(all_hs))
    print("###############################")
    print("Process is completed")


determine_scope(df_M, "import")
determine_scope(df_X, "export")



