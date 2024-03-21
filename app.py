########### WEB APPLICATION ################
# This script contains flask web application.
# It uses web application's helper functions.
# Obtained csv files are used as data source and, it is deployed.


#################################
# Import Library and Settings
#################################
from flask import Flask, request, render_template
import base64
import AppFunctions as af


##############################
# Flask
##############################
# A flask object is created.
app = Flask(__name__)

# Some needed csv files are read.
predictions, mae_values, hs_descriptions, countries = af.read_csv()

# Flask calls home(), which has the application / URL path.
@app.route('/')
def home():
    """
    Displays the home page of the web application.

    Returns:
        render_template: The rendered HTML template for the home page, containing a list of HS codes.

    """

    # We want to see the Hs codes we are interested in as a dropdown list on the main page.
    # This list can be expanded and narrowed according to the purpose.
    hs_codes = list(hs_descriptions["hs_code"])

    return render_template('home.html', hs_codes= hs_codes)

# Indicates an endpoint in the Flask application with URL path "/main".
# This endpoint accepts HTTP GET requests.
@app.route("/main", methods=["GET"])
def main():
    """
     Displays the main page of the web application with trade predictions and trends of the input pairs.

    Returns:
        render_template: The rendered HTML template for the main page, including trade predictions, trends,
                         and graphical representations.
    Returns:

    """

    # request function of Flask is used for getting inputs.
    hs_code = request.args['hs_code']
    trade_type = request.args['trade_type']

    # We want to see the Hs codes we are interested in as a dropdown list on this page again.
    hs_codes = list(hs_descriptions["hs_code"])

    # The description of Hs Code is taken from interested file.
    description = hs_descriptions.loc[hs_descriptions["hs_code"] == hs_code, "description"].values[0]

    # If there are no results for the entered binary, it goes to the except block.
    # Some Hs codes are only present in the import data while some are only present in the export data.
    try:
        # The desired image was obtained using helper functions.
        raw_data = af.read_data(hs_code, trade_type)
        graph_data = af.graph_input(raw_data)
        pred = af.get_prediction(graph_data, hs_code, trade_type, predictions)
        model_name = af.get_model_name(hs_code, trade_type, mae_values)

        trend1, trend3, trend_all = af.trend_images(pred)
        country1, country3, country_all = af.trend_country(raw_data, countries)

        img = af.plot_actual_pred(graph_data, pred, model_name, hs_code, trade_type)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template(  # These parameters are determined in HTML script.
                               "predict.html",
                               plot_url = plot_url,
                               hs_codes = hs_codes,
                               description = description,
                               trend1 = trend1,
                               trend3 = trend3,
                               trend_all = trend_all,
                               country1 = country1,
                               country3 = country3,
                               country_all = country_all)

    except FileNotFoundError:
        trade_str = "Import" if trade_type == "M" else "Export"
        return render_template("error.html",
                               description = description,
                               hs_codes = hs_codes,
                               hs_code = hs_code,
                               trade_str = trade_str)


# Flask app is run.
app.run(host="0.0.0.0", port=8080)
