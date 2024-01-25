import pandas as pd
from flask import Flask, request, render_template
import seaborn as sns
import base64
import io
import matplotlib
matplotlib.pyplot.switch_backend('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

def read_csv():
    predictions_data = pd.read_csv('datasets/predictions.csv', sep='|', dtype={'hs_code': str})
    mae_values_data = pd.read_csv("datasets/mae_values.csv", sep="|", dtype={"hs_code": str})
    hs_descriptions_data = pd.read_csv("datasets/hs_interested.csv", sep="|", dtype={"hs_code": str})
    countries = pd.read_csv("datasets/helpers data/countries.csv", sep = "|")
    return predictions_data, mae_values_data, hs_descriptions_data, countries


def read_data(hs_code, trade_type):
    if trade_type == "M":
        data = pd.read_parquet(f"datasets/imports/{hs_code}_{trade_type}_comtrade.parquet")
    else:
        data = pd.read_parquet(f"datasets/exports/{hs_code}_{trade_type}_comtrade.parquet")
    return data


def stats_input(dataframe):
    df = dataframe.copy()
    df["period"] = pd.to_datetime(df["period"], format='%Y%m')
    df = df[["period", "primaryValue"]]
    df = df.groupby("period").agg({"primaryValue": "sum"}).reset_index()

    df.set_index('period', inplace=True)
    df = df.resample('M').mean().fillna(method = "ffill")
    df = df.rename(columns = {'primaryValue': 'primary_value'})
    df = df.reset_index()
    return df


def get_prediction(stats_data, hs_code, trade_type):
    pred = predictions[(predictions['hs_code'] == hs_code) &
                       (predictions['trade_type'] == trade_type)].reset_index(drop=True)
    pred['period'] = pd.to_datetime(pred["period"], format='%Y-%m-%d')
    pred['cap'] = stats_data['primary_value'].max() + stats_data['primary_value'].std()
    return pred


def get_model_name(hs_code, trade_type):
    model_name = mae_values.loc[(mae_values['hs_code'] == hs_code) &
                                (mae_values['trade_type'] == trade_type), 'selected_model'].values[0]
    model_name = model_name.title().replace('_', ' ')
    return model_name


def trend_images(pred):
    image_dict = {1: "green_arrow.png",
                  0: "yellow_arrow.png",
                  -1: "red_arrow.png"}

    trends = pred['trend_label'].values[0]
    trends = [image_dict[i] for i in list(map(int, trends.strip('[]').split(',')))]

    trend1, trend3, trend_all = trends[0], trends[1], trends[2]

    return trend1, trend3, trend_all


def trend_country(hs_code, trade_type, countries):
    if trade_type == "M":
        df = pd.read_parquet(f"datasets/imports/{hs_code}_M_comtrade.parquet")
    else:
        df = pd.read_parquet(f"datasets/exports/{hs_code}_X_comtrade.parquet")

    one_year = df[df["period"] > 202211].groupby("partnerCode").agg({"primaryValue": "sum"}).reset_index().sort_values("primaryValue", ascending = False)["partnerCode"].values[0]
    three_year = df[df["period"] > 202011].groupby("partnerCode").agg({"primaryValue": "sum"}).reset_index().sort_values("primaryValue", ascending = False)["partnerCode"].values[0]
    all_year = df.groupby("partnerCode").agg({"primaryValue": "sum"}).reset_index().sort_values("primaryValue", ascending = False)["partnerCode"].values[0]

    one_year_name = countries.loc[countries["country_code"] == one_year, "country_name"].values[0]
    three_year_name = countries.loc[countries["country_code"] == three_year, "country_name"].values[0]
    all_year_name = countries.loc[countries["country_code"] == all_year, "country_name"].values[0]

    return one_year_name, three_year_name, all_year_name


def plot_actual_pred(actual_data, pred_data, model_name, hs_code, trade_type):
    plt.figure(facecolor='w', figsize=(10, 6))
    sns.lineplot(x='period', y='primary_value', data=pred_data, color='#0072B2')
    sns.lineplot(x='period', y='primary_value_lower', data=pred_data, color='#0072B2', lw=0.25)
    c = sns.lineplot(x='period', y='primary_value_upper', data=pred_data, color='#0072B2', lw=0.25)

    line = c.get_lines()
    plt.fill_between(line[0].get_xdata(), line[1].get_ydata(), line[2].get_ydata(), color='#0072B2', alpha=.2)
    plt.plot(actual_data['period'], actual_data['primary_value'], 'k.', label='Actual')
    plt.plot(pred_data['period'], pred_data['cap'], ls='--', c='k')

    plt.xlabel('Year Month', color='#FF7518', fontsize=16, fontweight='bold',
               fontname='Arial')
    plt.ylabel('Value', color='#FF7518', fontsize=16, fontweight='bold',
               fontname='Arial')
    plt.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    plt.title(f'{model_name} - {hs_code} - {trade_type}', color='#FF7518', fontsize=20, fontweight='bold',
              fontname='Arial')
    plt.tick_params(axis='both', labelcolor='black', labelsize=14)
    plt.gca().set_facecolor('#FCFBF4')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.gca().yaxis.get_offset_text().set_color("#0072B2")
    plt.gca().yaxis.get_offset_text().set_size(14)
    plt.gca().yaxis.get_offset_text().set_weight("bold")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img


predictions, mae_values, hs_descriptions, countries = read_csv()
@app.route('/')
def home():
    hs_codes = list(hs_descriptions["hs_code"])
    return render_template('home.html', hs_codes= hs_codes)


@app.route("/main", methods=["GET"])
def main():
    hs_code = request.args['hs_code']
    trade_type = request.args['trade_type']

    hs_codes = list(hs_descriptions["hs_code"])
    description = hs_descriptions.loc[hs_descriptions["hs_code"] == hs_code, "description"].values[0]
    try:
        raw_data = read_data(hs_code, trade_type)
        stats_data = stats_input(raw_data)
        pred = get_prediction(stats_data, hs_code, trade_type)
        model_name = get_model_name(hs_code, trade_type)

        img = plot_actual_pred(stats_data, pred, model_name, hs_code, trade_type)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        trend1, trend3, trend_all = trend_images(pred)
        country1, country3, country_all = trend_country(hs_code, trade_type, countries)

        return render_template("predict.html", plot_url=plot_url, hs_codes=hs_codes, description=description,
                               trend1=trend1, trend3=trend3, trend_all=trend_all,
                               country1= country1, country3=country3, country_all= country_all)
    except FileNotFoundError:
        trade_str = "Import" if trade_type == "M" else "Export"
        return render_template("error.html",
                               description=description, hs_codes=hs_codes, hs_code=hs_code, trade_str=trade_str)


app.run(host="0.0.0.0", port=8080)
