########### WEB SCRAPING YEARLY DATA ################
# These codes allow import and export data to be received from Comtrade on an annual basis.
# Firstly, the URL of the site where the data is obtained and the API key obtained from the site are defined.
# Some parameters are set in the scraping function and the get function of the request library is used.
# The function is then executed with a for loop for the year and trade type lists.
# The results were saved as a parquet file.
# M represents imports and X represents exports.
# Comtrade is the site from which the data is taken.


###############################
# Import Library and Settings
###############################
import requests
import pandas as pd
import time

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


###############################
# Scraping Function
###############################
request_url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
# ...get/commodity code:C/ monthly:M, annually:A/ Hs code:HS
headers = {'Ocp-Apim-Subscription-Key': '7fd766b8d3c64a95bc21114588796f3b'}

def get_data(period, flowcode):
    """
    This is a web scraping request code.
    Args:
        period:
        flowcode:

    Returns:
        dataframe or fail response
    """
    params = {
        'reporterCode': 276,
        'period': period,
        "flowCode": flowcode,
        "partnerCode": 0,
        "motCode": 0,
        "customsCode": "C00",
        "partner2Code": 0
    }
    response = requests.get(url = request_url, params = params, headers = headers)

    if response.status_code == 200:
        data = pd.DataFrame(response.json()['data'])

        summary_data = data[(data['cmdCode'].str.len() == 6)].reset_index(drop = True)

        required_columns = ['reporterCode', 'period', 'cmdCode', 'partnerCode', 'qty', 'netWgt', 'grossWgt',
                            'cifvalue', 'fobvalue', 'primaryValue', 'refYear', 'flowCode']
        summary_data = summary_data[required_columns]

        return summary_data
    else:
        return response


###############################
# Scraping Process
###############################
year = range(2016, 2023)
types = ["M", "X"]

df_M = pd.DataFrame()
df_X = pd.DataFrame()

for i in year:
    for j in types:
        temp = get_data(i, j)
        if j == "M":
            df_M = pd.concat([df_M, temp], ignore_index = True)
        else:
            df_X = pd.concat([df_X, temp], ignore_index = True)
        print(f"In {i} {j} {len(temp)} rows data are downloaded.")
        time.sleep(10)


df_M.to_parquet('datasets/comtrade_yearly_import.parquet', engine="pyarrow")
df_X.to_parquet('datasets/comtrade_yearly_export.parquet', engine="pyarrow")

