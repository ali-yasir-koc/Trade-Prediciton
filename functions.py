## 1- Import Data and Settings
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima import auto_arima
from pmdarima.arima import ARIMA
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
import logging
import sys

logging.disable(sys.maxsize)

warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('mode.chained_assignment', None)

## 2- Read Data
def read_data(hs_code, trade_type):
    if trade_type == "M":
        data = pd.read_parquet(f"datasets/imports/{hs_code}_{trade_type}_comtrade.parquet")
    else:
        data = pd.read_parquet(f"datasets/exports/{hs_code}_{trade_type}_comtrade.parquet")
    return data


def read_helper_data():
    helpers_data = pd.read_csv('datasets/helpers_data.csv', sep='|')

    return helpers_data


## 3- For Stats Models Input Creation
def stats_input(dataframe):
    df = dataframe.copy()
    df["period"] = pd.to_datetime(df["period"], format='%Y%m')
    df = df[["period", "primaryValue"]]
    df = df.groupby("period").agg({"primaryValue": "sum"}).reset_index()

    if (len(df) >= 72) & (df['period'].max() >= pd.to_datetime('2023-01-01')):
        df.set_index('period', inplace=True)
        df = df.resample('M').mean().fillna(method = "ffill")
        df["primaryValue"] = np.log1p(df["primaryValue"])
        return df
    elif df['period'].max() < pd.to_datetime('2023-01-01'):
        temp = pd.DataFrame({"hs_code": dataframe["cmdCode"].unique(),
                             "trade_type": dataframe["flowCode"].unique(),
                             "elimination_reason": "There is no data in 2023"})
        temp.to_csv("datasets/eliminated.csv", sep="|", index = False, mode='a', header=False)
    else:
        temp = pd.DataFrame({"hs_code": dataframe["cmdCode"].unique(),
                             "trade_type": dataframe["flowCode"].unique(),
                             "elimination_reason": "There is no enough data"})
        temp.to_csv("datasets/eliminated.csv", sep = "|", index = False, mode = 'a', header = False)


def ml_input(dataframe):
    helpers_data = read_helper_data()
    df = dataframe.copy()
    df["period"] = pd.to_datetime(df["period"], format='%Y%m')
    helpers_data["period"] = pd.to_datetime(helpers_data["period"], format="%Y%m")
    df = df.groupby("period").agg({"primaryValue": "sum",
                                   "netWgt": "sum"}).reset_index()
    max_date = df.period.max()
    min_date = df.period.min()
    full_date_range = pd.date_range(start = min_date,
                                    end = max_date + pd.DateOffset(months = 12),
                                    freq = 'MS')
    full_df = pd.DataFrame({"period": full_date_range})
    df = pd.merge(full_df, df, how = "left", on = "period")
    mask = df["period"] <= max_date
    df.loc[mask] = df.loc[mask].fillna(method = 'ffill')
    df["unit_price"] = df["primaryValue"] / df["netWgt"]
    df['year'] = df['period'].dt.year
    df['month'] = df['period'].dt.month
    df = pd.merge(df, helpers_data, how="left", on="period")

    ss = StandardScaler()
    standard_col = ['netWgt', "unit_price", 'parity', 'base_index', 'yearly_index', 'petrol_price', 'gdp', 'un_rate']
    for col in standard_col:
        df[col] = ss.fit_transform(df[[col]])
    df["primaryValue"] = np.log1p(df["primaryValue"])

    lag = 12
    for col in standard_col:
        df[f'{col}_lag_{str(lag)}'] = df[col].transform(lambda x: x.shift(lag))
        df[f'{col}_roll_mean_{str(lag)}'] = df[col].transform(
            lambda x: x.shift(1).rolling(window=lag, min_periods=1, win_type="triang").mean())

    df.drop(columns=['netWgt', 'unit_price', 'parity', 'base_index',
                     'yearly_index', 'petrol_price', 'gdp', 'un_rate'], inplace=True)
    na_cols = df.columns[4:]

    for col in na_cols:
        size = len(df[df[col].isna()])
        mean_val = df[col].mean()
        scale = df[col].std()/4
        replace_val = [i + mean_val for i in np.random.normal(scale=scale, size=(size,))]
        df.loc[df[col].isna(), col] = replace_val
    return df


def ml_data_prepare(dataframe, final = False):
    test_start = 12 if final else 24
    test_finish = None if final else -12

    train = dataframe.iloc[:-test_start]
    test = dataframe.iloc[-test_start:test_finish]

    return train, test


## 4- Trend Tests
def linear_trend(dataframe):
    trend_labels = []
    trend_values = []
    for i in (12, 36, len(dataframe)):
        X = np.arange(i).reshape(-1, 1)
        # Create a value with the number of dates on the x-axis
        y = dataframe.iloc[-i:].values.reshape(-1, 1)
        mms = MinMaxScaler()
        y = mms.fit_transform(y)
        # Set values to the desired shape

        model = LinearRegression()
        model.fit(X, y)
        trend_slope = model.coef_[0][0]
        trend_values.append(trend_slope)
        if trend_slope > 0.0001:
            trend_labels.append(1)
        elif trend_slope < -0.0001:
            trend_labels.append(-1)
        else:
            trend_labels.append(0)

    return trend_labels, trend_values


## 5- Test/Train Split
def stats_data_prepare(dataframe, test_size = 12):
    train = dataframe.iloc[:-test_size]
    test = dataframe.iloc[-test_size:]

    return train, test


## 6- Smoothing Models
def tes_optimizer(train, test, abg, step=12):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12). \
            fit(smoothing_level=comb[0], smoothing_trend=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        # print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    return best_alpha, best_beta, best_gamma, best_mae


def smoothing_model(dataframe, final=False, best_params=None):
    if final:
        model = ExponentialSmoothing(dataframe, trend="add", seasonal="add", seasonal_periods=12). \
            fit(**best_params)
        y_fitted = model.fittedvalues
        y_pred = model.forecast(12)
        y_pred = pd.concat([y_fitted, y_pred])

        y_pred = pd.DataFrame(y_pred)
        y_pred = y_pred.reset_index()
        y_pred.columns = ['period', 'primary_value']

        confidence_level = 1.96
        std_dev = dataframe['primaryValue'].std()

        y_pred['primary_value_lower'] = y_pred['primary_value'] - confidence_level * std_dev
        y_pred['primary_value_upper'] = y_pred['primary_value'] + confidence_level * std_dev

        return y_pred, 1
    else:
        train, test = stats_data_prepare(dataframe)

        alphas = betas = gammas = np.arange(0.20, 1, 0.10)
        abg = list(itertools.product(alphas, betas, gammas))

        best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, test, abg)
        best_params = {"smoothing_level": best_alpha,
                       "smoothing_trend": best_beta,
                       "smoothing_seasonal": best_gamma}
        final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12). \
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

        y_pred = final_tes_model.forecast(12)
        mae = mean_absolute_error(test, y_pred)
        return mae, best_params


## 7- Stats Models
def adfuller_test(dataframe):
    adf_result = adfuller(dataframe)
    return adf_result[1]


def auto_arima_optimizer(dataframe):
    train, test = stats_data_prepare(dataframe)
    p_value = adfuller_test(dataframe)
    if p_value > 0.05:
        d = 0
        df_diff = dataframe["primaryValue"]
        while p_value > 0.05:
            df_diff = np.diff(df_diff, n = 1)
            p_value = adfuller_test(df_diff)
            d += 1
    else:
        d = 0
    # ARIMA_model = auto_arima(train,
    #                          start_p = 1,
    #                          start_q = 1,
    #                          max_q = 6,  # maximum p and q
    #                          max_p = 6,
    #                          d = d,
    #                          m = 1,  # frequency of series (if m==1, seasonal is set to FALSE automatically)
    #                          trace = False,  # logs
    #                          error_action = 'warn',  # shows errors ('ignore' silences these)
    #                          suppress_warnings = True,
    #                          stepwise = True,
    #                          scoring = "mae")
    SARIMA_model = auto_arima(train,
                              start_p = 1, start_q = 1,
                              max_p = 6, max_q = 6,
                              m = 12,  # 12 is the frequency of the cycle
                              start_P = 1, start_Q = 1,
                              max_P = 6, max_Q = 6,
                              seasonal = True,  # set to seasonal
                              d = d,
                              D = d,
                              trace = False,
                              error_action = 'ignore',
                              suppress_warnings = True,
                              stepwise = True,
                              scoring = "mae")
    return SARIMA_model


def stats_model(dataframe, final=False, best_params=None):
    if final:
        model = ARIMA(**best_params)
        model = model.fit(dataframe)
        y_fitted = model.fittedvalues()
        y_pred = model.predict(n_periods = 12)
        y_pred = pd.concat([y_fitted, y_pred])
        return y_pred
    else:
        train, test = stats_data_prepare(dataframe)
        SARIMA_model = auto_arima_optimizer(dataframe)
        # ARIMA_model = ARIMA_model.fit(train)
        SARIMA_model = SARIMA_model.fit(train)
        # predictions_arima = ARIMA_model.predict(n_periods = 12)
        # mae_arima = mean_absolute_error(test, predictions_arima)
        predictions_sarima = SARIMA_model.predict(n_periods = 12)
        mae = mean_absolute_error(test, predictions_sarima)
        best_params = {"order": SARIMA_model.order,
                       "seasonal_order": SARIMA_model.seasonal_order}

        return mae, best_params


## Prophet One Column
def prophet_optimizer_one_col(dataframe):
    best_params, best_mae = None, float("inf")

    params_grid = {'seasonality_mode': ('multiplicative', 'additive'),
                   'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
                   'n_changepoints': [10, 20, 30],
                   'growth': ['linear', 'logistic']}
    grid = ParameterGrid(params_grid)

    df = dataframe.reset_index()
    df.columns = ['ds', 'y']
    df['cap'] = df['y'].max() + df['y'].std()

    train_data, test_data = stats_data_prepare(df)

    for p in grid:
        np.random.seed(0)
        train_model = Prophet(**p,
                              yearly_seasonality=True,
                              interval_width=0.95)
        train_model.fit(train_data)
        train_forecast = train_model.make_future_dataframe(periods=12, freq='m', include_history=False)
        train_forecast['cap'] = train_data['cap'].max()
        train_forecast = train_model.predict(train_forecast)
        test_pred = train_forecast[['ds', 'yhat']]
        mae = mean_absolute_error(test_data['y'], test_pred['yhat'])
        if mae < best_mae:
            best_mae = mae
            best_params = p

    return best_params, best_mae


def prophet_one_col_model(dataframe, final=False, best_params=None):
    if final:
        df = dataframe.copy()
        df = df.reset_index()
        df.columns = ['ds', 'y']
        df['cap'] = df['y'].max() + df['y'].std()

        final_model = Prophet(**best_params,
                              yearly_seasonality=True,
                              interval_width=0.95)
        final_model.fit(df)

        future = final_model.make_future_dataframe(periods=12, freq='m')
        future['cap'] = df['cap'].max()
        forecast = final_model.predict(future)

        y_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        y_pred.columns = ['period', 'primary_value', 'primary_value_lower', 'primary_value_upper']
        return y_pred, forecast
    else:
        best_params, mae = prophet_optimizer_one_col(dataframe)

        return mae, best_params


def prophet_optimizer_multiple_col(dataframe):
    best_params, best_mae = None, float("inf")

    params_grid = {'seasonality_mode': ('multiplicative', 'additive'),
                   'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],
                   'n_changepoints': [10, 20, 30],
                   'growth': ['linear', 'logistic']}
    grid = ParameterGrid(params_grid)

    df = dataframe.rename(columns={"period": "ds", "primaryValue": "y"})
    df['cap'] = df['y'].max() + df["y"].std()

    train_data, test_data = ml_data_prepare(df)

    for p in grid:
        np.random.seed(0)
        train_model = Prophet(**p,
                              yearly_seasonality=True,
                              interval_width=0.95)

        for col in df.columns[2:-1]:
            train_model.add_regressor(col, standardize=False)
        train_model.fit(train_data)
        train_forecast = train_model.predict(test_data)
        test_pred = train_forecast[['ds', 'yhat']]
        mae = mean_absolute_error(test_data['y'], test_pred['yhat'])
        if mae < best_mae:
            best_mae = mae
            best_params = p

    return best_params, best_mae


def prophet_multiple_col_model(dataframe, final=False, best_params=None):
    if final:
        df = dataframe.copy()
        df = df.rename(columns={"period": "ds", "primaryValue": "y"})
        df['cap'] = df['y'].max() + df["y"].std()
        df_train, df_predict = ml_data_prepare(df, final=True)

        final_model = Prophet(**best_params,
                              yearly_seasonality=True,
                              interval_width=0.95)
        for col in df.columns[2:-1]:
            final_model.add_regressor(col, standardize=False)

        final_model.fit(df_train)
        forecast = final_model.predict(df)

        y_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        y_pred.columns = ['period', 'primary_value', 'primary_value_lower', 'primary_value_upper']
        return y_pred, forecast
    else:
        best_params, mae = prophet_optimizer_multiple_col(dataframe)

        return mae, best_params


## 8- Calculate MAE
def calculate_mae(hs_code, trade_type, stats_df, ml_df, trend_label, trend_value):
    mae1, params_1 = smoothing_model(stats_df)
    print(f'Smoothing MAE: {mae1}')
    mae2, params_2 = prophet_one_col_model(stats_df)
    print(f'Prophet One Column MAE: {mae2}')
    mae3, params_3 = prophet_multiple_col_model(ml_df)
    print(f'Prophet Multiple Columns MAE: {mae3}')

    mae_dict = {"hs_code": [hs_code],
                "trade_type": [trade_type],
                "trend_label": [trend_label],
                "trend_value": [trend_value],
                "smoothing_model": [mae1],
                "smoothing_params": [params_1],
                "prophet_one_col_model": [mae2],
                "prophet_one_col_params": [params_2],
                "prophet_multiple_col_model": [mae3],
                "prophet_multiple_col_params": [params_3]
                }
    results = pd.DataFrame(mae_dict)

    numeric_columns = ['smoothing_model', 'prophet_one_col_model', 'prophet_multiple_col_model']
    results['selected_model'] = results[numeric_columns].idxmin(axis=1)
    results['min_error'] = results[numeric_columns].min(axis=1)
    return results


## 9- Prediction
def generate_predictions(result, stats_df, ml_df):
    model_type = result.loc[0]['selected_model']

    input_df = ml_df.copy() if model_type == 'prophet_multiple_col_model' else stats_df.copy()

    best_params = result.loc[0][f'{"_".join(model_type.split("_")[:-1])}_params']

    predictions, forecast = eval(model_type)(input_df, final=True, best_params=best_params)

    for col in predictions.columns[1:]:
        predictions.loc[:, col] = np.expm1(predictions.loc[:, col])

    predictions.loc[:, 'hs_code'] = result.loc[0]['hs_code']
    predictions.loc[:, 'trade_type'] = result.loc[0]['trade_type']
    predictions.loc[:, 'trend_label'] = [result.loc[0]['trend_label']] * len(predictions)
    predictions.loc[:, 'trend_value'] = [result.loc[0]['trend_value']] * len(predictions)
    predictions = predictions[['hs_code', 'trade_type', 'trend_label', 'trend_value', 'period', 'primary_value',
                               'primary_value_lower', 'primary_value_upper']]

    return predictions, forecast
