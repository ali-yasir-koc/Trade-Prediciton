import sys
import pandas as pd
import functions as f

predictions = pd.read_csv('datasets/predictions.csv', sep='|', dtype={'hs_code': 'str'})
mae_values = pd.read_csv('datasets/mae_values.csv', sep='|', dtype={'hs_code': 'str'})

trends = mae_values[['hs_code', 'trade_type', 'trend_label', 'trend_value']]

trends['trend_label'] = trends['trend_label'].apply(lambda x: x.strip('[]').split(','))
trends['label_1'] = trends['trend_label'].apply(lambda x: int(x[0]))
trends['label_3'] = trends['trend_label'].apply(lambda x: int(x[1]))
trends['label_all'] = trends['trend_label'].apply(lambda x: int(x[2]))
del trends['trend_label']

trends['trend_value'] = trends['trend_value'].apply(lambda x: x.strip('[]').split(','))
trends['trend_1'] = trends['trend_value'].apply(lambda x: float(x[0]))
trends['trend_3'] = trends['trend_value'].apply(lambda x: float(x[1]))
trends['trend_all'] = trends['trend_value'].apply(lambda x: float(x[2]))
del trends['trend_value']

trends.head()

one_years_first_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_1', ascending=False).head(50)
three_years_first_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_3', ascending=False).head(50)
all_years_first_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_all', ascending=False).head(50)

one_years_first_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_1', ascending=False).head(50)
three_years_first_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_3', ascending=False).head(50)
all_years_first_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_all', ascending=False).head(50)

one_years_last_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_1', ascending=True).head(50)
three_years_last_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_3', ascending=True).head(50)
all_years_last_50_M = trends[trends['trade_type'] == 'M'].sort_values('trend_all', ascending=True).head(50)

one_years_last_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_1', ascending=True).head(50)
three_years_last_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_3', ascending=True).head(50)
all_years_last_50_X = trends[trends['trade_type'] == 'X'].sort_values('trend_all', ascending=True).head(50)


hs_list = list(one_years_first_50_M["hs_code"])
hs_list.extend(list(three_years_first_50_M["hs_code"]))
hs_list.extend(list(all_years_first_50_M["hs_code"]))
hs_list.extend(list(one_years_first_50_X["hs_code"]))
hs_list.extend(list(three_years_first_50_X["hs_code"]))
hs_list.extend(list(all_years_first_50_X["hs_code"]))
hs_list.extend(list(one_years_last_50_M["hs_code"]))
hs_list.extend(list(three_years_last_50_M["hs_code"]))
hs_list.extend(list(all_years_last_50_M["hs_code"]))
hs_list.extend(list(one_years_last_50_X["hs_code"]))
hs_list.extend(list(three_years_last_50_X["hs_code"]))
hs_list.extend(list(all_years_last_50_X["hs_code"]))

hs_list = list(set(hs_list))

hs_desc = pd.read_csv("datasets/hs_descriptions.csv", sep="|", dtype={"hs_code": str})
hs_desc.head()
hs_interested = hs_desc[hs_desc["hs_code"].isin(hs_list)].reset_index(drop=True)
hs_interested.head()
hs_interested.to_csv("datasets/hs_interested.csv", sep="|", index=False)


hs_interested = pd.read_csv("datasets/hs_interested.csv", sep="|", dtype={'hs_code': str})

trends['sel'] = trends['trend_1'] * trends['trend_3']


trends[(trends['hs_code'].isin(hs_interested['hs_code'])) & (trends['trade_type'] == 'M') &
       (trends['label_3'] == 1)].\
    sort_values('trend_1', ascending=False).head(10)
trends[(trends['hs_code'].isin(hs_interested['hs_code'])) & (trends['trade_type'] == 'M') &
       (trends['label_3'] == -1)].\
    sort_values('trend_1', ascending=True).head(10)


trends[(trends['hs_code'].isin(hs_interested['hs_code'])) & (trends['trade_type'] == 'X') &
       (trends['label_3'] == 1)].\
    sort_values('trend_1', ascending=False).head(10)
trends[(trends['hs_code'].isin(hs_interested['hs_code'])) & (trends['trade_type'] == 'X') &
       (trends['label_3'] == -1)].\
    sort_values('trend_1', ascending=True).head(10)

