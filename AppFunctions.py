########### HELPER FUNCTION of WEB APPLICATION ################
# This script contains helper functions of web application.


#################################
# Import Library and Settings
#################################
import pandas as pd
import seaborn as sns
import io
import matplotlib.pyplot as plt


##############################
# 1 - Read CSV
##############################
def read_csv():
    """
    Reads the required CSV files for the deployment.

    Returns:
        predictions_data (DataFrame): DataFrame containing predictions data.
        mae_values_data (DataFrame): DataFrame containing mean absolute error (MAE) values.
        hs_descriptions_data (DataFrame): DataFrame containing Harmonized System (HS) code descriptions.
        countries (DataFrame): DataFrame containing country information.

    """
    predictions_data = pd.read_csv('datasets/predictions.csv', sep = '|', dtype = {'hs_code': str})
    mae_values_data = pd.read_csv("datasets/mae_values.csv", sep = "|", dtype = {"hs_code": str})
    hs_descriptions_data = pd.read_csv("datasets/hs_interested.csv", sep = "|", dtype = {"hs_code": str})
    countries = pd.read_csv("datasets/helpers data/countries.csv", sep = "|")

    return predictions_data, mae_values_data, hs_descriptions_data, countries


##############################
# 2 - Read Raw Data
##############################
def read_data(hs_code, trade_type):
    """
    Reads the trade data for a specific HS code and trade type.

    Args:
        hs_code (str): The Harmonized System (HS) code for the trade data.
        trade_type (str): The type of trade data, either "M" for imports or "X" for exports.

    Returns:
        DataFrame: DataFrame containing the trade data for the specified HS code and trade ty

    """
    if trade_type == "M":
        data = pd.read_parquet(f"datasets/imports/{hs_code}_{trade_type}_comtrade.parquet")
    else:
        data = pd.read_parquet(f"datasets/exports/{hs_code}_{trade_type}_comtrade.parquet")

    return data


###################################
# 3 - Input Preparation For Graph
###################################
def graph_input(dataframe):
    """
     Preprocesses input DataFrame for graphs.

    Args:
        dataframe (DataFrame): The input DataFrame containing period and primary value data.

    Returns:
        DataFrame: DataFrame with preprocessed data, aggregated monthly mean primary values.

    """
    df = dataframe.copy()
    df["period"] = pd.to_datetime(df["period"], format = '%Y%m')
    df = df[["period", "primaryValue"]]

    # Since it deals with the sum of the transactions of all countries in that month on the basis of hs code,
    # primaryValues are summed.
    df = df.groupby("period").agg({"primaryValue": "sum"}).reset_index()

    # This step is done to use in filling of missing values step.
    df.set_index('period', inplace = True)

    # Missing values are filled with the next value in the data.
    df = df.resample('M').mean().fillna(method = "ffill")
    df = df.rename(columns = {'primaryValue': 'primary_value'})

    # A two-column dataframe is obtained again.
    df = df.reset_index()

    return df


##############################
# 4 - Extract Prediction
##############################
def get_prediction(graph_df, hs_code, trade_type, predictions_df):
    """
    Retrieves predictions for a specific HS code and trade type and preprocesses them.

    Args:
        graph_df (DataFrame): DataFrame containing graph data.
        hs_code (str): The Harmonized System (HS) code for the trade data.
        trade_type (str): The type of trade data, either "M" for imports or "X" for exports.
        predictions_df (DataFrame): DataFrame containing predictions data.

    Returns:
        DataFrame: DataFrame containing predictions for the specified HS code and trade type,
                   with preprocessed period data and a calculated 'cap' value

    """

    # Prediction values are selected based on Hs Code and Trade type.
    pred = predictions_df[(predictions_df['hs_code'] == hs_code) &
                          (predictions_df['trade_type'] == trade_type)].reset_index(drop = True)
    pred['period'] = pd.to_datetime(pred["period"], format = '%Y-%m-%d')

    # The maximum value of prediction was determined by summing the standard deviation of primary values and
    # the maximum value of primary values.
    pred['cap'] = graph_df['primary_value'].max() + graph_df['primary_value'].std()

    return pred


##############################
# 5 - Grab Model Name
##############################
def get_model_name(hs_code, trade_type, mae_df):
    """
    Retrieves the name of the selected model for a given HS code and trade type.

    Args:
        hs_code (str): The Harmonized System (HS) code for the trade data.
        trade_type (str): The type of trade data, either "M" for imports or "X" for exports.
        mae_df (Dataframe): Dataframe contains mae values of all models.
    Returns:
        str: The name of the selected model for the specified HS code and trade type.

    """

    # Selected model is grabbed by using Hs Code and Trade type
    model_name = mae_df.loc[(mae_df['hs_code'] == hs_code) &
                            (mae_df['trade_type'] == trade_type), 'selected_model'].values[0]

    # The first letters of the model name have been capitalized.
    model_name = model_name.replace('_', ' ').title()

    return model_name


##############################
# 6 - Generate Trend Images
##############################
def trend_images(pred):
    """
    Generates image paths for trend labels.

    Args:
        pred (DataFrame): DataFrame containing trend label predictions.

    Returns:
        tuple: A tuple containing image paths for trend labels: trend1, trend3, trend

    """

    # This dictionary was created to show the images corresponding to the labels assigned to the trends.
    # 1 indicates an increasing trend, 0 indicates constant trend and -1 indicates a decreasing trend.
    image_dict = {1: "green_arrow.png",
                  0: "yellow_arrow.png",
                  -1: "red_arrow.png"}

    # Trend labels are got.
    trends = pred['trend_label'].values[0]

    # Trend labels in string format have been converted to list.
    # Then, the image equivalents of the values in this list were selected according to the dictionary above.
    trends = [image_dict[i] for i in list(map(int, trends.strip('[]').split(',')))]

    # Images are assigned an object one by one.
    trend1, trend3, trend_all = trends[0], trends[1], trends[2]

    return trend1, trend3, trend_all


##############################
# 7 - Find Top Partner Country
##############################
def trend_country(raw_data_df, countries):
    """
    Determines the top trading partner countries for different time periods based on primary value.

    Args:
        raw_data_df (DataFrame): DataFrame containing raw trade data.
        countries (DataFrame): DataFrame containing country information.

    Returns:
        tuple: A tuple containing the names of the top trading partner countries for the following time periods:
               one_year_name, three_year_name, all_year_name.

    """

    # The codes of the country with the highest trade volume in the specified date range are assigned a value.
    # We collect trade volumes by country and rank them from largest to smallest.
    one_year = raw_data_df[raw_data_df["period"] > 202211]. \
        groupby("partnerCode"). \
        agg({"primaryValue": "sum"}).reset_index(). \
        sort_values("primaryValue", ascending = False)["partnerCode"].values[0]

    three_year = raw_data_df[raw_data_df["period"] > 202011]. \
        groupby("partnerCode").\
        agg({"primaryValue": "sum"}).reset_index().\
        sort_values("primaryValue", ascending = False)["partnerCode"].values[0]

    all_year = raw_data_df.\
        groupby("partnerCode").\
        agg({"primaryValue": "sum"}).reset_index().\
        sort_values("primaryValue", ascending = False)["partnerCode"].values[0]

    # Country names are assigned according to obtained country codes.
    one_year_name = countries.loc[countries["country_code"] == one_year, "country_name"].values[0]
    three_year_name = countries.loc[countries["country_code"] == three_year, "country_name"].values[0]
    all_year_name = countries.loc[countries["country_code"] == all_year, "country_name"].values[0]

    return one_year_name, three_year_name, all_year_name


##############################
# 8- Plot
##############################
def plot_actual_pred(actual_data, pred_data, model_name, hs_code, trade_type):
    """
    Plots actual and predicted values along with prediction intervals for visualization.

    Args:
        actual_data (DataFrame): DataFrame containing actual trade data.
        pred_data (DataFrame): DataFrame containing predicted trade data and prediction intervals.
        model_name (str): The name of the predictive model.
        hs_code (str): The Harmonized System (HS) code for the trade data.
        trade_type (str): The type of trade data, either "M" for imports or "X" for exports.

    Returns:
        BytesIO: A BytesIO object containing the plotted image.

    """

    # First, a figure framework is created.
    plt.figure(facecolor = 'w', figsize = (10, 6))

    # Prediction values are plotted in the figure as a line.
    sns.lineplot(x = 'period', y = 'primary_value', data = pred_data, color = '#0072B2')
    sns.lineplot(x = 'period', y = 'primary_value_lower', data = pred_data, color = '#0072B2', lw = 0.25)

    # The lineplot function returns a matplotlib.axes._subplots.AxesSubplot object representing the generated line plot.
    # This object can be used to control the properties of the lines and fill areas on the plot.
    # By assigning the return value of sns.lineplot to variable c, we store this object for later use.
    c = sns.lineplot(x = 'period', y = 'primary_value_upper', data = pred_data, color = '#0072B2', lw = 0.25)

    # The space between the lower and upper values  are filled in to better see the range of the estimates.
    # After the line graph is drawn, Line2D objects are created to represent each line.
    # The get_lines() method returns a list of these Line2D objects.
    line = c.get_lines()
    plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color = '#0072B2', alpha = .2)

    # The actual data is plotted as points.
    plt.plot(actual_data['period'], actual_data['primary_value'], 'k.', label = 'Actual')

    # The maximum value that the model can predict is plotted as a dash.
    plt.plot(pred_data['period'], pred_data['cap'], ls = '--', c = 'k')

    # The title of figure is determined.
    plt.title(f'{model_name} - {hs_code} - {trade_type}', color = '#FF7518', fontsize = 20, fontweight = 'bold',
              fontname = 'Arial')

    # The names of the axes are determined.
    plt.xlabel('Year Month', color = '#FF7518', fontsize = 16, fontweight = 'bold', fontname = 'Arial')
    plt.ylabel('Value', color = '#FF7518', fontsize = 16, fontweight = 'bold', fontname = 'Arial')

    # Adjustments were made to the labels on the axes.
    plt.tick_params(axis = 'both', labelcolor = 'black', labelsize = 14)
    plt.xticks(fontweight = 'bold')
    plt.yticks(fontweight = 'bold')

    # gca() (get current axes) is used to change the properties of the current axis.
    # The appearance on the graph of the multiplier of the values on the y-axis was adjusted.
    plt.gca().yaxis.get_offset_text().set_color("#0072B2")
    plt.gca().yaxis.get_offset_text().set_size(14)
    plt.gca().yaxis.get_offset_text().set_weight("bold")

    # The background color of figure is determined.
    plt.gca().set_facecolor('#FCFBF4')

    # The background of figure is grid.
    plt.grid(True, which = 'major', c = 'gray', ls = '-', lw = 1, alpha = 0.2)

    # A BytesIO object is created to use in HTML.
    img = io.BytesIO()

    # Plotted graph is saved in the img object.
    plt.savefig(img, format = 'png')

    # When plt.close() is called, the current figure is closed and freed from memory.
    plt.close()

    # Before using the data for an operation, the read position is returned to the start position.
    img.seek(0)

    return img
