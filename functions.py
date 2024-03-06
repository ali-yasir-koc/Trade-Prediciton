########### HELPER FUNCTION ################
# This script contains helper functions of project.


##############################
# Import Library and Settings
##############################
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

warnings.simplefilter(action = "ignore", category = ConvergenceWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('mode.chained_assignment', None)


##############################
# 1 - Read Data
##############################
def read_data(hs_code, trade_type):
    """
    It is a function that reads the existing parquet file according to the input hs code and trade type information.
    Args:
        hs_code: it is a string value like "271600"
        trade_type: M or X, it represents import or export
    Returns:
        dataframe
    """
    if trade_type == "M":
        data = pd.read_parquet(f"datasets/imports/{hs_code}_{trade_type}_comtrade.parquet")
    else:
        data = pd.read_parquet(f"datasets/exports/{hs_code}_{trade_type}_comtrade.parquet")
    return data


def read_helper_data():
    """
    It is a function that reads helper data that contains some economical information.
    Returns:
        dataframe
    """
    helpers_data = pd.read_csv('datasets/helpers_data.csv', sep = '|')

    return helpers_data


###############################################
# 2 - Input Preparation For Statistical  Model
###############################################
def stats_input(dataframe):
    """
    It is the function that makes the raw data ready for statistical models by putting it through some processes.
    Statistical models use single-column dataframes with date information in the index.
    Therefore, this function aims to transform the data into a single column dataframe with one value per month.

    Args:
        dataframe: it is raw data for Hs Code-trade type pair

    Returns:
        dataframe or it adds some information to eliminated csv file
    """
    df = dataframe.copy()
    df["period"] = pd.to_datetime(df["period"], format = '%Y%m')
    df = df[["period", "primaryValue"]]

    # Since it deals with the sum of the transactions of all countries in that month on the basis of hs code,
    # primaryValues are summed.
    df = df.groupby("period").agg({"primaryValue": "sum"}).reset_index()

    if (len(df) >= 72) & (df['period'].max() >= pd.to_datetime('2023-01-01')):
        df.set_index('period', inplace = True)

        # Missing values are filled with the next value in the data.
        df = df.resample('M').mean().fillna(method = "ffill")

        # Logarithmic transformation was performed for primaryValue.
        df["primaryValue"] = np.log1p(df["primaryValue"])
        return df

    # If there are no transactions for Hs Code in 2023 or the total number of transactions is less than 72 months,
    # this Hs Code is sent to the eliminated csv file.
    elif df['period'].max() < pd.to_datetime('2023-01-01'):
        temp = pd.DataFrame({"hs_code": dataframe["cmdCode"].unique(),
                             "trade_type": dataframe["flowCode"].unique(),
                             "elimination_reason": "There is no data in 2023"})
        temp.to_csv("datasets/eliminated.csv", sep = "|", index = False, mode = 'a', header = False)
    else:
        temp = pd.DataFrame({"hs_code": dataframe["cmdCode"].unique(),
                             "trade_type": dataframe["flowCode"].unique(),
                             "elimination_reason": "There is no enough data"})
        temp.to_csv("datasets/eliminated.csv", sep = "|", index = False, mode = 'a', header = False)


###############################################
# 3 - Input Preparation For ML  Model
###############################################
def ml_input(dataframe):
    """
    It is the function that makes the raw data ready for machine learning models by putting it through some processes.
    Args:
        dataframe: it is raw data for Hs Code-trade type pair

    Returns:
        dataframe
    """
    helpers_data = read_helper_data()
    df = dataframe.copy()
    df["period"] = pd.to_datetime(df["period"], format = '%Y%m')
    helpers_data["period"] = pd.to_datetime(helpers_data["period"], format = "%Y%m")

    # Since it deals with the sum of the transactions of all countries in that month on the basis of hs code,
    # primaryValues are summed.
    df = df.groupby("period").agg({"primaryValue": "sum",
                                   "netWgt": "sum"}).reset_index()

    # First a time range is created, to create a value for all months for each dataset .
    max_date = df.period.max()
    min_date = df.period.min()
    full_date_range = pd.date_range(start = min_date,
                                    end = max_date + pd.DateOffset(months = 12),
                                    freq = 'MS')
    full_df = pd.DataFrame({"period": full_date_range})

    # The empty dataframe containing the time intervals is merged with the real dataframe.
    df = pd.merge(full_df, df, how = "left", on = "period")
    mask = df["period"] <= max_date

    # Missing values are filled with the next value in the data.
    df.loc[mask] = df.loc[mask].fillna(method = 'ffill')

    # New feature creation steps
    df["unit_price"] = df["primaryValue"] / df["netWgt"]
    df['year'] = df['period'].dt.year
    df['month'] = df['period'].dt.month
    df = pd.merge(df, helpers_data, how = "left", on = "period")

    # Numerical values are standardised for more meaningful and more measurable processing.
    ss = StandardScaler()
    standard_col = ['netWgt', "unit_price", 'parity', 'base_index', 'yearly_index', 'petrol_price', 'gdp', 'un_rate']
    for col in standard_col:
        df[col] = ss.fit_transform(df[[col]])

    # Logarithmic transformation was performed for primaryValue.
    df["primaryValue"] = np.log1p(df["primaryValue"])

    # 1-Each value is assigned as a lag value 12 months into the next month
    # 2-The average of the last 12 months is taken and assigned to the lag value of the next month.
    lag = 12
    for col in standard_col:
        df[f'{col}_lag_{str(lag)}'] = df[col].transform(lambda x: x.shift(lag))
        df[f'{col}_roll_mean_{str(lag)}'] = df[col].transform(
            lambda x: x.shift(1).rolling(window = lag, min_periods = 1, win_type = "triang").mean())

    # The columns without lag are deleted.
    df.drop(columns = ['netWgt', 'unit_price', 'parity', 'base_index',
                       'yearly_index', 'petrol_price', 'gdp', 'un_rate'], inplace = True)

    # Newly created lagged columns contain na values.
    # To fill these missing values, a random count generated over the standard deviation of that column and
    # the mean of that column is used.
    na_cols = df.columns[4:]
    for col in na_cols:
        size = len(df[df[col].isna()])
        mean_val = df[col].mean()
        scale = df[col].std() / 4
        replace_val = [i + mean_val for i in np.random.normal(scale = scale, size = (size,))]
        df.loc[df[col].isna(), col] = replace_val
    return df


#########################
# 4 - Test/Train Split
#########################
def stats_data_prepare(dataframe, test_size = 12):
    """
    This is a function that divides the data obtained for statistical models into training and test data.
    Args:
        dataframe:  data obtained for statistical models
        test_size: by default the last 12 months are determined

    Returns:
        train dataframe, test dataframe
    """
    train = dataframe.iloc[:-test_size]
    test = dataframe.iloc[-test_size:]

    return train, test


def ml_data_prepare(dataframe, final = False):
    """
    This is a function that divides the data obtained for machine learning models into training and test data.
    Args:
        dataframe: data obtained for machine learning models
        final: when setting up the final model, this parameter takes the value True

    Returns:
        train dataframe, test dataframe
    """

    # If final is False;
    # Since we create a 12-month lag in the input prepared for machine learning,
    # we subtract the last 24 months while testing the model and obtain the train set.
    # If final is True;
    # The dates we predict are the test set and the whole data is the train set.
    test_start = 12 if final else 24

    # For the same reason, we take the 12 months following the train set as a test set.
    test_finish = None if final else 12

    train = dataframe.iloc[:-test_start]
    test = dataframe.iloc[-test_start:-test_finish]

    return train, test


#########################
# 5 - Trend Calculate
#########################
def linear_trend(dataframe):
    """
    This is a function that calculates the slope for the last 12 months, last 36 months and all months of the values
     in the given dataframe and labels the calculated slope according to a threshold value.
    Linear regression was used to determine the slopes.
    Trend labels [1,0,-1] represent an increasing steady and decreasing trend.
    Args:
        dataframe: It is one column dataframe

    Returns:
        trends label: example [1,0,-1], trends value: float values
    """
    trend_labels = []
    trend_values = []

    # Independent and dependent variables called X and y are created.
    # Where X represents the indices of the data set, y represents the values in the data set.
    X = np.arange(len(dataframe)).reshape(-1, 1)
    y = dataframe.iloc[:].values.reshape(-1, 1)

    # We standardise values across all data sets to bring them to a common scale.
    mms = MinMaxScaler()
    y = mms.fit_transform(y)

    # For each dataset, the slope is calculated for the last 12 months, last 36 months and all months.
    for i in (12, 36, len(dataframe)):
        model = LinearRegression()
        model.fit(X[-i:], y[-i:])

        # y = b + mx
        # y represents the dependent variable, x represents the independent variable,
        # m represents the slope coefficient, b represents the intercept.
        trend_slope = model.coef_[0][0]
        trend_values.append(trend_slope)
        if trend_slope > 0.001:
            trend_labels.append(1)
        elif trend_slope < -0.001:
            trend_labels.append(-1)
        else:
            trend_labels.append(0)

    return trend_labels, trend_values


#########################
# 6 - Smoothing Model
#########################
def tes_optimizer(train, test, abg, step = 12):
    """
    This is a function for selecting the most appropriate all alpha beta and
    gamma parameters for the Triple Exponential Smoothing model.

    Args:
        train (Dataframe): DataFrame containing training data
        test (Dataframe): DataFrame containing testing data
        abg (list): a list of the combination of alpha, beta and gamma parameters
        step (int): maximum number of steps to be moved in the time unit for the forecasts

    Returns:
        the best alpha, beta, gama: best value of alpha, beta and gama
        best mae: the smallest mae value for these parameters

    """

    # By default, these parameters are assigned as null.
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")

    # TES models are trained by different alpha, beta and gama combination.
    # Trend and seasonal parameters for TES model are additive.(Multiplicative is another choice)
    # The parameter seasonal periods defines 12 months (an annual cycle) as the period of the seasonal component.
    # The comb expression has alpha values in the first index, beta values in the second index and
    # gamma values in the third index.
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend = "add", seasonal = "add", seasonal_periods = 12). \
            fit(smoothing_level = comb[0], smoothing_trend = comb[1], smoothing_seasonal = comb[2])

        # y_pred holds the prediction values of the model for the indicated number of steps.
        y_pred = tes_model.forecast(step)

        # The error between the test values and the prediction values is calculated.
        mae = mean_absolute_error(test, y_pred)

        # If the MAE obtained is smaller than the best MAE found so far,
        # the best parameter values and the MAE value are updated.
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae

    return best_alpha, best_beta, best_gamma, best_mae


def smoothing_model(dataframe, final = False, best_params = None):
    """
    This function creates and trains a Triple Exponential Smoothing model.
    The model is used to smooth time series data in the DataFrame provided
    and can be used to forecast future values

    Args:
        dataframe (DataFrame): DataFrame containing the time series data.
        final (bool): A flag indicating whether the model should be finally trained
                      and predictions made. Defaults to False.
        best_params (dict): Dictionary of the best parameters to be used in the final
                            training of the model. Defaults to None.

    Returns:
         tuple: Returns a tuple in the form (y_pred, 1) if final=True,
               or (best_mae, best_params) if final=False.

    Notes:  If final=True, it returns the y_pred DataFrame and 1.
            The value 1 is added only to be compatible with the output of the prophet model.

    """
    if final:
        # The final model is built with the best parameters previously determined.
        model = ExponentialSmoothing(dataframe, trend = "add", seasonal = "add", seasonal_periods = 12). \
            fit(**best_params)

        # The predicted values of the model for the training data are stored in the variable y_fitted.
        y_fitted = model.fittedvalues

        # The values predicted by the model for the next 12 months were kept in the y_pred variable.
        y_pred = model.forecast(12)

        # The model's predictions for the past and the future were combined.
        y_pred = pd.concat([y_fitted, y_pred])
        y_pred = pd.DataFrame(y_pred)
        y_pred = y_pred.reset_index()
        y_pred.columns = ['period', 'primary_value']

        # In order to create a range around the prediction values of the model,
        # we determined the lower and upper bounds of the prediction values.
        # The tabular equivalent of the 95 per cent confidence interval value of 1.96 was used for this purpose.
        # The standard deviation of the values in the dataframe was also used.
        confidence_level = 1.96
        std_dev = dataframe['primaryValue'].std()

        # The lower confidence limit of the estimated values is calculated
        # by subtracting the product of the confidence level and the standard deviation from the estimated value.
        y_pred['primary_value_lower'] = y_pred['primary_value'] - confidence_level * std_dev

        # The upper confidence limit of the predicted values is calculated
        # by summing the product of the confidence level and the standard deviation from the predicted value.
        y_pred['primary_value_upper'] = y_pred['primary_value'] + confidence_level * std_dev

        return y_pred, 1
    else:
        # Dataframe is divided into test and train.
        train, test = stats_data_prepare(dataframe)

        # A range is created for the alpha beta and gamma parameters.
        # A list containing combinations of these values is obtained.
        alphas = betas = gammas = np.arange(0.20, 1, 0.10)
        abg = list(itertools.product(alphas, betas, gammas))

        # The best values are obtained with the optimisation function.
        best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, test, abg)
        best_params = {"smoothing_level": best_alpha,
                       "smoothing_trend": best_beta,
                       "smoothing_seasonal": best_gamma}
        return best_mae, best_params


################################
# 7 - Prophet One Column Model
################################
def prophet_optimizer_one_col(dataframe):
    """
    This function performs parameter optimization for the Prophet model that uses a single time series column.
    It iterates over a grid of parameter combinations, including seasonality mode, changepoint prior scale,
     number of changepoints, and growth type.
    The grid search is conducted to find the combination of parameters that minimizes the Mean Absolute Error (MAE)
     between the actual and predicted values on a test set.

    Args:
        dataframe (Dataframe): DataFrame containing the time series data.

    Returns:
        tuple: Returns a tuple containing the best parameters and the corresponding MAE.

    """

    # By default, these parameters are assigned as null.
    best_params, best_mae = None, float("inf")

    params_grid = {
        # Indicates that act of the seasonality component
        'seasonality_mode': ('multiplicative', 'additive'),

        # It is a parameter used to detect change points.
        # For example a sudden change of a trend.
        # The larger this value is, the more flexible the change points are predicted by the model.
        'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],

        # This parameter determines the number of change points in the time series.
        # Change points represent structural changes in the time series.
        # The higher this parameter is, more flexible the model is and more change points can be identified.
        # However, too many change points may lead to overfitting.
        'n_changepoints': [10, 20, 30],

        # Indicates the growth type of the time series.
        'growth': ['linear', 'logistic']}
    grid = ParameterGrid(params_grid)

    # The dataframe for the prophet model is reorganised.
    df = dataframe.reset_index()
    df.columns = ['ds', 'y']

    # Here, the maximum value df['y'].max() and the standard deviation df["y"].std() of column 'y' in the DataFrame
    # are summed and assigned to column 'cap'.
    # This is used to ensure that the model does not predict values higher than the maximum value in the data set.
    # Cap is used as a limiting parameter in the Prophet model and determines the maximum value the model can produce.
    df['cap'] = df['y'].max() + df['y'].std()

    # Dataframe is divided into test and train.
    train_data, test_data = stats_data_prepare(df)

    for p in grid:
        np.random.seed(0)
        train_model = Prophet(**p,
                              yearly_seasonality = True,
                              interval_width = 0.95)
        train_model.fit(train_data)

        train_forecast = train_model.make_future_dataframe(periods = 12, freq = 'm', include_history = False)
        train_forecast['cap'] = train_data['cap'].max()
        train_forecast = train_model.predict(train_forecast)
        test_pred = train_forecast[['ds', 'yhat']]

        mae = mean_absolute_error(test_data['y'], test_pred['yhat'])
        if mae < best_mae:
            best_mae = mae
            best_params = p

    return best_params, best_mae


def prophet_one_col_model(dataframe, final = False, best_params = None):
    """
    Constructs a Prophet model for time series forecasting based on the provided single-column dataframe.

    If `final` is True, the function builds the final model using the best parameters (`best_params`)
    obtained from optimization. Otherwise, it performs parameter optimization to determine the best
    model configuration based on the smallest Mean Absolute Error (MAE) on a test set.

    Args:
        dataframe (DataFrame): A DataFrame containing the time series data.
        final (bool): A flag indicating whether to build the final model or perform parameter optimization.
                      Defaults to False.
        best_params (dict): A dictionary containing the best parameter values obtained from optimization.
                            This parameter is required when `final` is True. Defaults to None.

    Returns:
        - If `final` is True:
                 - y_pred (DataFrame): A DataFrame containing the forecasted values along with confidence intervals.
                 - forecast (DataFrame): The complete forecast DataFrame generated by the Prophet model.
        - If `final` is False:
                 - mae (float): The Mean Absolute Error obtained with the best parameter configuration.
                 - best_params (dict): A dictionary containing the best parameter values obtained from optimization.

    """
    if final:
        df = dataframe.copy()

        # The dataframe for the prophet model is reorganised.
        df = df.reset_index()
        df.columns = ['ds', 'y']

        # Determine a cap value
        df['cap'] = df['y'].max() + df['y'].std()

        final_model = Prophet(**best_params,
                              yearly_seasonality = True,
                              interval_width = 0.95)
        final_model.fit(df)

        future = final_model.make_future_dataframe(periods = 12, freq = 'm')
        future['cap'] = df['cap'].max()
        forecast = final_model.predict(future)

        y_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        y_pred.columns = ['period', 'primary_value', 'primary_value_lower', 'primary_value_upper']
        return y_pred, forecast

    else:
        best_params, mae = prophet_optimizer_one_col(dataframe)

        return mae, best_params


#####################################
# 8 - Prophet Multiple Column Model
#####################################
def prophet_optimizer_multiple_col(dataframe):
    """
    This function performs parameter optimization for the Prophet model that uses multiple time series column.
    It iterates over a grid of parameter combinations, including seasonality mode, changepoint prior scale,
     number of changepoints, and growth type.
    The grid search is conducted to find the combination of parameters that minimizes the Mean Absolute Error (MAE)
     between the actual and predicted values on a test set.

    Args:
        dataframe (Dataframe): DataFrame containing the time series data.

    Returns:
        tuple: Returns a tuple containing the best parameters and the corresponding MAE.
    """

    # By default, these parameters are assigned as null.
    best_params, best_mae = None, float("inf")

    params_grid = {
        # Indicates that act of the seasonality component
        'seasonality_mode': ('multiplicative', 'additive'),

        # It is a parameter used to detect change points.
        # For example a sudden change of a trend.
        # The larger this value is, the more flexible the change points are predicted by the model.
        'changepoint_prior_scale': [0.1, 0.2, 0.3, 0.4, 0.5],

        # This parameter determines the number of change points in the time series.
        # Change points represent structural changes in the time series.
        # The higher this parameter is, more flexible the model is and more change points can be identified.
        # However, too many change points may lead to overfitting.
        'n_changepoints': [10, 20, 30],

        # Indicates the growth type of the time series.
        'growth': ['linear', 'logistic']}
    grid = ParameterGrid(params_grid)

    # The dataframe for the prophet model is reorganised.
    df = dataframe.rename(columns = {"period": "ds", "primaryValue": "y"})

    # Here, the maximum value df['y'].max() and the standard deviation df["y"].std() of column 'y' in the DataFrame
    # are summed and assigned to column 'cap'.
    # This is used to ensure that the model does not predict values higher than the maximum value in the data set.
    # Cap is used as a limiting parameter in the Prophet model and determines the maximum value the model can produce.
    df['cap'] = df['y'].max() + df["y"].std()

    # Dataframe is divided into test and train by using ml_data_prepare.
    train_data, test_data = ml_data_prepare(df)

    for p in grid:
        np.random.seed(0)
        train_model = Prophet(**p,
                              yearly_seasonality = True,
                              interval_width = 0.95)

        # In multi-column ML models, other columns are added one by one as regressors.
        for col in df.columns[2:-1]:
            train_model.add_regressor(col, standardize = False)
        train_model.fit(train_data)

        train_forecast = train_model.predict(test_data)
        test_pred = train_forecast[['ds', 'yhat']]

        mae = mean_absolute_error(test_data['y'], test_pred['yhat'])
        if mae < best_mae:
            best_mae = mae
            best_params = p

    return best_params, best_mae


def prophet_multiple_col_model(dataframe, final = False, best_params = None):
    """
    Constructs a Prophet model for time series forecasting based on the provided multi-column dataframe.

    If `final` is True, the function builds the final model using the best parameters (`best_params`)
    obtained from optimization. Otherwise, it performs parameter optimization to determine the best
    model configuration based on the smallest Mean Absolute Error (MAE) on a test set.
    Args:
        dataframe (DataFrame): A DataFrame containing the time series data.
        final (bool): A flag indicating whether to build the final model or perform parameter optimization.
                      Defaults to False.
        best_params (dict): A dictionary containing the best parameter values obtained from optimization.
                            This parameter is required when `final` is True. Defaults to None.

    Returns:
        - If `final` is True:
                 - y_pred (DataFrame): A DataFrame containing the forecasted values along with confidence intervals.
                 - forecast (DataFrame): The complete forecast DataFrame generated by the Prophet model.
        - If `final` is False:
                 - mae (float): The Mean Absolute Error obtained with the best parameter configuration.
                 - best_params (dict): A dictionary containing the best parameter values obtained from optimization.

    """
    if final:
        df = dataframe.copy()

        # The dataframe for the prophet model is reorganised.
        df = df.rename(columns = {"period": "ds", "primaryValue": "y"})

        # Determine a cap value
        df['cap'] = df['y'].max() + df["y"].std()

        # Dataframe is divided into predict and train.
        df_train, df_predict = ml_data_prepare(df, final = True)

        final_model = Prophet(**best_params,
                              # This parameter determines the use of the annual seasonality component in the model.
                              yearly_seasonality = True,
                              # This determines the confidence interval of the estimated values.
                              interval_width = 0.95)

        # In multi-column ML models, other columns are added one by one as regressors.
        for col in df.columns[2:-1]:
            final_model.add_regressor(col, standardize = False)

        final_model.fit(df_train)

        # Forecast contains all metadata obtained with prophet.
        forecast = final_model.predict(df)

        y_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        y_pred.columns = ['period', 'primary_value', 'primary_value_lower', 'primary_value_upper']
        return y_pred, forecast

    else:
        best_params, mae = prophet_optimizer_multiple_col(dataframe)

        return mae, best_params


########################
# 9 - Calculate MAE
########################
def calculate_mae(hs_code, trade_type, stats_df, ml_df, trend_label, trend_value):
    """
    Calculates the Mean Absolute Error (MAE) for different time series forecasting models,
    including smoothing models and Prophet models, and returns the results in a DataFrame.

    Args:
        hs_code (str): The HS code
        trade_type (str): The type of trade (e.g., import or export).
        stats_df (DataFrame): Single column dataframe generated by stats_input
        ml_df (DataFrame): Multiple column dataframe generated by stats_input
        trend_label (list): A list of trend labels obtained from the data.
        trend_value (list): A list of trend values obtained from the data.

    Returns:
        DataFrame: A DataFrame containing the MAE values for different models,
                   along with the associated parameters and selected model information.

    Notes:
        - This function calculates the MAE for three types of models: smoothing models, Prophet models
          with one column, and Prophet models with multiple columns.
        - The function prints the MAE values for each model during execution.
        - The selected model for each data point is determined based on the model with the minimum MAE.
        - The selected model information and the minimum error value are included in the output DataFrame.

    """

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

    # The name of the model with the lowest MAE value is assigned to the selected model column.
    numeric_columns = ['smoothing_model', 'prophet_one_col_model', 'prophet_multiple_col_model']
    results['selected_model'] = results[numeric_columns].idxmin(axis = 1)
    results['min_error'] = results[numeric_columns].min(axis = 1)

    return results


########################
# 10 - Prediction
########################
def generate_predictions(result, stats_df, ml_df):
    """
    Generates predictions using the selected model from the results DataFrame,
    and returns the forecasted values along with associated information.

    Args:
        result (DataFrame): A DataFrame containing the results from model selection,
                            including the selected model type and associated parameters.
        stats_df (DataFrame): Single column dataframe generated by stats_input
        ml_df (DataFrame): Multiple column dataframe generated by stats_input

    Returns:
        tuple: A tuple containing the predictions DataFrame and the forecast DataFrame.

    Notes:
        - This function selects the model type based on the 'selected_model' column in the results DataFrame.
        - It retrieves the best parameters for the selected model from the results DataFrame.
        - It generates predictions using the selected model with the best parameters.
        - If the selected model is 'prophet_multiple_col_model', it uses the machine learning DataFrame (ml_df)
          as input; otherwise, it uses the statistical DataFrame (stats_df).
        - The predictions DataFrame contains the forecasted values along with associated information such as
          HS code, trade type, trend label, trend value, period, primary value, primary value lower bound,
          and primary value upper bound.
        - The forecast DataFrame contains the complete forecast generated by the selected Prophet model.


    """

    # The model type is assigned to a variable
    model_type = result.loc[0]['selected_model']

    # The model data is assigned to a variable based on model type
    input_df = ml_df.copy() if model_type == 'prophet_multiple_col_model' else stats_df.copy()

    # The best parameters are selected based on model type and assigned to a variable
    best_params = result.loc[0][f'{"_".join(model_type.split("_")[:-1])}_params']

    # The appropriate model is created according to the model type, parameters are entered and prediction is made.
    predictions, forecast = eval(model_type)(input_df, final = True, best_params = best_params)

    # The logarithmic transformation performed when preparing the data as input is reversed.
    for col in predictions.columns[1:]:
        predictions.loc[:, col] = np.expm1(predictions.loc[:, col])

    # Some parameters in the result dataframe are transferred to the prediction dataframe.
    predictions.loc[:, 'hs_code'] = result.loc[0]['hs_code']
    predictions.loc[:, 'trade_type'] = result.loc[0]['trade_type']
    predictions.loc[:, 'trend_label'] = [result.loc[0]['trend_label']] * len(predictions)
    predictions.loc[:, 'trend_value'] = [result.loc[0]['trend_value']] * len(predictions)
    predictions = predictions[['hs_code', 'trade_type', 'trend_label', 'trend_value', 'period', 'primary_value',
                               'primary_value_lower', 'primary_value_upper']]

    return predictions, forecast
