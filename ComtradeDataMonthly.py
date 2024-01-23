########### Import Library and Settings  ############
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


########### Get Data ############
df_M_HS = pd.read_parquet("datasets/all_hs_import.parquet", engine = "pyarrow")
df_X_HS = pd.read_parquet("datasets/all_hs_export.parquet", engine = "pyarrow")


########### Request Generation ############
request_url = "https://comtradeapi.un.org/data/v1/get/C/M/HS"
headers = {'Ocp-Apim-Subscription-Key': '4741ec0a2a7145a0b8db02d914cc58a9', } # 7fd766b8d3c64a95bc21114588796f3b


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


sd = get_data(202201, "950490", "M")  # örnek


########### Date Scope Generation ###########
start_date = datetime(2016, 1, 1)
end_date = datetime(2022, 12, 31)

date_list = []
current_date = start_date

while current_date <= end_date:
    date_list.append(current_date.strftime('%Y%m'))
    current_date = current_date + timedelta(days = 32)

print(date_list)


############## Data Scraping ############
for i in df_M_HS["HS_Code"][7:]:
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


