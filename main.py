import sys
import pandas as pd
import functions as f


HsCode = sys.argv[1]
TradeType = sys.argv[2]

# HsCode = '271600'
# TradeType = 'M'
try:
    RawData = f.read_data(HsCode, TradeType)
    print(f'Data read for {HsCode} & {TradeType}')

    StatsData = f.stats_input(RawData)
    if not isinstance(StatsData, pd.DataFrame):
        print(f'Hs Code added to elimination for {HsCode} & {TradeType}')
        print('########################################################################')
        exit()
    print(f'Stats data prepared for {HsCode} & {TradeType}')

    MlData = f.ml_input(RawData)
    print(f'ML data prepared read for {HsCode} & {TradeType}')

    TrendLabels, TrendValues = f.linear_trend(StatsData)
    print(f'Trend determined for {HsCode} & {TradeType}')

    Results = f.calculate_mae(HsCode, TradeType, StatsData, MlData, TrendLabels, TrendValues)
    print(f'MAE values calculated for {HsCode} & {TradeType}')

    Predictions, Forecast = f.generate_predictions(Results, StatsData, MlData)
    print(f'Predictions created for {HsCode} & {TradeType}')

    Results.to_csv("datasets/mae_values.csv", sep="|", index = False, mode='a', header=False)
    Predictions.to_csv("datasets/predictions.csv", sep="|", index = False, mode='a', header=False)

    if Results.loc[0]['selected_model'] != 'smoothing_model':
        Forecast.to_csv("datasets/prophet_metadata.csv", sep="|", index = False, mode='a', header=False)
    print('########################################################################')

except Exception as e:
    print(e)
