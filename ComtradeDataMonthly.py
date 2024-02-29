########### WEB SCRAPING MONTHLY DATA ################
# These codes allow import and export data to be received from Comtrade on a monthly basis.
# Firstly, data is loaded from the parquet file containing the Hs Codes within the scope.
# Secondly, the URL of the site where the data is obtained and the API key obtained from the site are defined.
# Some parameters are set in the scraping function and the get function of the request library is used.
# A period list containing all months in the desired range is prepared.
# Then the function is executed with a for loop.
# The results were saved as a parquet file.
# M represents imports and X represents exports.
# Comtrade is the site from which the data is taken.

##############################
# Import Library and Settings
##############################
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


##############################
# Get Hs Code Data
##############################
df_M_HS = pd.read_parquet("datasets/all_hs_import.parquet", engine = "pyarrow")
df_X_HS = pd.read_parquet("datasets/all_hs_export.parquet", engine = "pyarrow")


##############################
# Scraping Function
##############################
request_url = "https://comtradeapi.un.org/data/v1/get/C/M/HS"
headers = {'Ocp-Apim-Subscription-Key': '4741ec0a2a7145a0b8db02d914cc58a9'}


def get_data(period, hs_code, flowcode):
    params = {
        'reporterCode': 276,
        'period': period,
        "flowCode": flowcode,
        'cmdCode': hs_code,
        "motCode": 0,
        "customsCode": "C00",
        "partner2Code": 0
    }

    response = requests.get(url = request_url, params = params, headers = headers)
    try:
        data = pd.DataFrame(response.json()['data'])

        summary_data = data[(data['cmdCode'].str.len() == 6) &
                            (data['partnerCode'] != 0)].reset_index(drop = True)

        required_columns = ['reporterCode', 'period', 'cmdCode', 'partnerCode', 'qty', 'netWgt', 'grossWgt',
                            'cifvalue', 'fobvalue', 'primaryValue', 'refYear', 'flowCode']
        summary_data = summary_data[required_columns]
    except:
        required_columns = ['reporterCode', 'period', 'cmdCode', 'partnerCode', 'qty', 'netWgt', 'grossWgt',
                            'cifvalue', 'fobvalue', 'primaryValue', 'refYear', 'flowCode']
        summary_data = pd.DataFrame(columns = required_columns)
    return summary_data


##############################
# Date Scope Generation
##############################
start_date = '2010-01-01'
end_date = '2023-11-30'

date_list = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y%m').tolist()
print(date_list)


##############################
# Data Scraping Process
##############################
for i in df_M_HS["HS_Code"]:
    df_temp = pd.DataFrame()
    for j in date_list:
        temp = get_data(j, i, "M")
        df_temp = pd.concat([df_temp, temp], ignore_index = True)
        print(f"{i} HSCODE {j} is downloaded {datetime.now()}")
        time.sleep(10)
    df_temp.to_parquet(f'datasets/import/{i}_M_comtrade.parquet', engine= "pyarrow")

for i in df_X_HS["HS_Code"]:
    df_temp = pd.DataFrame()
    for j in date_list:
        temp = get_data(j, i, "X")
        df_temp = pd.concat([df_temp, temp], ignore_index = True)
        print(f"{i} HSCODE {j} is downloaded {datetime.now()}")
        time.sleep(10)
    df_temp.to_parquet(f'datasets/export/{i}_X_comtrade.parquet', engine= "pyarrow")


