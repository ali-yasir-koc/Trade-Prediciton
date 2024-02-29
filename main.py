########### MAIN FUNCTION ################
# This is the main code of the project.
# It executes by using of helper functions.
# The raw data is prepared with two different functions for different paths.
# The prepared data are exposed to the trend calculation function and
# the function in which different models are used and errors are calculated.
# The results and predictions for each hs code and trade type pair are added as rows to the appropriate csv files.


##############################
# Import Library and Settings
##############################
import sys
import pandas as pd
import functions as f


##############################
# Main Algorithm
##############################
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
