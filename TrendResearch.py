import pandas as pd


predictions = pd.read_csv('datasets/predictions.csv', sep='|', dtype={'hs_code': 'str'})
mae_values = pd.read_csv('datasets/mae_values.csv', sep='|', dtype={'hs_code': 'str'})

trends = mae_values[['hs_code', 'trade_type', 'trend_label', 'trend_value']]

trends['trend_label'] = trends['trend_label'].apply(lambda x: x.strip('[]').split(','))
trends['label_3'] = trends['trend_label'].apply(lambda x: int(x[0]))
trends['label_5'] = trends['trend_label'].apply(lambda x: int(x[1]))
trends['label_all'] = trends['trend_label'].apply(lambda x: int(x[2]))
del trends['trend_label']

trends['trend_value'] = trends['trend_value'].apply(lambda x: x.strip('[]').split(','))
trends['trend_3'] = trends['trend_value'].apply(lambda x: float(x[0]))
trends['trend_5'] = trends['trend_value'].apply(lambda x: float(x[1]))
trends['trend_all'] = trends['trend_value'].apply(lambda x: float(x[2]))
del trends['trend_value']

trends.head()

three_years_first_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_3', ascending=False).head(50)
five_years_first_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_5', ascending=False).head(50)
all_years_first_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_all', ascending=False).head(50)

three_years_first_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_3', ascending=False).head(50)
five_years_first_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_5', ascending=False).head(50)
all_years_first_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_all', ascending=False).head(50)

three_years_last_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_3', ascending=True).head(50)
five_years_last_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_5', ascending=True).head(50)
all_years_last_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_all', ascending=True).head(50)

three_years_last_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_3', ascending=True).head(50)
five_years_last_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_5', ascending=True).head(50)
all_years_last_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_all', ascending=True).head(50)


hs_list = list(three_years_first_50_M["hs_code"])
hs_list.extend(list(five_years_first_50_M["hs_code"]))
hs_list.extend(list(all_years_first_50_M["hs_code"]))
hs_list.extend(list(three_years_first_50_X["hs_code"]))
hs_list.extend(list(five_years_first_50_X["hs_code"]))
hs_list.extend(list(all_years_first_50_X["hs_code"]))
hs_list.extend(list(three_years_last_50_M["hs_code"]))
hs_list.extend(list(five_years_last_50_M["hs_code"]))
hs_list.extend(list(all_years_last_50_M["hs_code"]))
hs_list.extend(list(three_years_last_50_X["hs_code"]))
hs_list.extend(list(five_years_last_50_X["hs_code"]))
hs_list.extend(list(all_years_last_50_X["hs_code"]))

hs_desc = pd.read_csv("datasets/hs_descriptions.csv", sep="|", dtype={"hs_code": str})
hs_desc.head()
hs_interested = hs_desc[hs_desc["hs_code"].isin(hs_list)].reset_index(drop=True)
hs_interested.head()
hs_interested.to_csv("datasets/hs_interested.csv", sep="|", index=False)


m_scope = pd.read_parquet('datasets/all_hs_import.parquet')
x_scope = pd.read_parquet('datasets/all_hs_export.parquet')

m_list = list(m_scope["HS_Code"])
x_list = list(x_scope["HS_Code"])

fark = set(m_list).difference(x_list)
sec = []
for i in fark:
    if i in hs_list:
        sec.append(i)
