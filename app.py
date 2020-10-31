# This code was based on the resources provided in the Data Mining for Computer and Systems Sciences (DAMI) course 2020 and originally created by luis-eduardo@dsv.su.se
#
# =============================================================================
# Imports
# =============================================================================

import helper_dash_example

import dash
from dash.dependencies import Input, Output, State

from pathlib import Path
import pandas as pd
import pickle

# =============================================================================
# Main
# =============================================================================

# Relative paths respect to current file
THIS_FILE_PATH = str(Path(__file__).parent.absolute())+"/"
filename_to_load = THIS_FILE_PATH + "trained_model_dataset.pickle"

# Variables to create the data structure from the web interface
dataset_colnames = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
sample = None   # DataFrame with the data that the user has input in the webpage
selected_index = 0

# Load trained model
loaded_model = None
with open(filename_to_load, "rb") as readFile:
    loaded_model = pickle.load(readFile)

# Styling for HTML website
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create web server
#app = dash.Dash("intelli_cardio", external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)

# In the additional file `helper_dash_example` is hidden all webpage' structure
app.layout = helper_dash_example.app_html_layout

# =============================================================================
# Callbacks to setup the interaction between webpage and controls
# The next syntax is specific from the dash library, documentation can be found
# on https://dash.plotly.com/
# =============================================================================

# Dropdown list
@app.callback(
    [
    Output(component_id='value-loaded-age', component_property='children'),
    Output('value-loaded-sex', 'children'),
    Output('value-loaded-cp', 'children'),
    Output('value-loaded-trestbps', 'children'),
    Output('value-loaded-chol', 'children'),
    Output('value-loaded-fbs', 'children'),
    Output('value-loaded-restecg', 'children'),
    Output('value-loaded-thalach', 'children'),
    Output('value-loaded-exang', 'children'),
    Output('value-loaded-oldpeak', 'children'),
    Output('value-loaded-slope', 'children'),
    Output('value-loaded-ca', 'children'),
    Output('value-loaded-thal', 'children'),
    ],
    [Input('demo-dropdown', 'value')])
def select_dropdown_value(index):
    sample = helper_dash_example.select_dropdown_value(index)
    global selected_index
    selected_index = index
    return sample["age"], sample["sex"], sample["cp"], sample["trestbps"], sample["chol"], sample["fbs"], sample["restecg"], sample["thalach"], sample["exang"], sample["oldpeak"], sample["slope"], float(sample["ca"]), float(sample["thal"])

# Sliders
# Generic function to return the string from a change in the web app
def update_value(value):
    return str(value)

@app.callback(
    Output(component_id='value-slider-age', component_property='children'),
    [Input(component_id='slider-age', component_property='value')]
)
def update_age(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-sex', component_property='children'),
    [Input(component_id='slider-sex', component_property='value')]
)
def update_sex(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-cp', component_property='children'),
    [Input(component_id='slider-cp', component_property='value')]
)
def update_cp(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-trestbps', component_property='children'),
    [Input(component_id='slider-trestbps', component_property='value')]
)
def update_trestbps(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-chol', component_property='children'),
    [Input(component_id='slider-chol', component_property='value')]
)
def update_chol(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-fbs', component_property='children'),
    [Input(component_id='slider-fbs', component_property='value')]
)
def update_fbs(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-restecg', component_property='children'),
    [Input(component_id='slider-restecg', component_property='value')]
)
def update_restecg(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-thalach', component_property='children'),
    [Input(component_id='slider-thalach', component_property='value')]
)
def update_thalach(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-exang', component_property='children'),
    [Input(component_id='slider-exang', component_property='value')]
)
def update_exang(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-oldpeak', component_property='children'),
    [Input(component_id='slider-oldpeak', component_property='value')]
)
def update_oldpeak(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-slope', component_property='children'),
    [Input(component_id='slider-slope', component_property='value')]
)
def update_slope(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-ca', component_property='children'),
    [Input(component_id='slider-ca', component_property='value')]
)
def update_ca(value):
    return update_value(value)

@app.callback(
    Output(component_id='value-slider-thal', component_property='children'),
    [Input(component_id='slider-thal', component_property='value')]
)
def update_thal(value):
    return update_value(value)


# Classification Button
@app.callback(
    Output(component_id='classification-result', component_property='children'),
    [Input(component_id='submit', component_property='n_clicks')],
    [
    State('slider-age', 'value'),
    State('slider-sex', 'value'),
    State('slider-cp', 'value'),
    State('slider-trestbps', 'value'),
    State('slider-chol', 'value'),
    State('slider-fbs', 'value'),
    State('slider-restecg', 'value'),
    State('slider-thalach', 'value'),
    State('slider-exang', 'value'),
    State('slider-oldpeak', 'value'),
    State('slider-slope', 'value'),
    State('slider-ca', 'value'),
    State('slider-thal', 'value'),
    ]
)
def execute_classification(n_clicks, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
#def execute_classification_loaded(n_clicks):
    """
    Main method. Loads the trained model, applies the input data and returns a class
    """

    if(n_clicks == None): # When the application open
        return "Press below to execute the classification"
    else:
        # The sliders' values are already parsed to numeric values
        # Here we create a DataFrame with the input data
        data_from_user = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        global sample
        sample = pd.DataFrame(data=[data_from_user], columns=dataset_colnames)
        print(sample)

        sample = helper_dash_example.normalize_data(sample)

        # Execute the prediction using the loaded trained model.
        prediction = loaded_model.predict(sample)

        # Return final message
        prediction_labels = ["No Heart Disease", "Heart Disease Detected"]
        print("The predicted class of the input data is: ["+ str(int(prediction[0])) +":" + prediction_labels[int(prediction[0])] + "]")
        return "Results: "+ prediction_labels[int(prediction[0])] + ""


# Classification Button Loaded
@app.callback(
    Output(component_id='classification-result-loaded', component_property='children'),
    [Input(component_id='submit-loaded', component_property='n_clicks')],
    [
    State('value-loaded-age', 'children'),
    State('value-loaded-sex', 'children'),
    State('value-loaded-cp', 'children'),
    State('value-loaded-trestbps', 'children'),
    State('value-loaded-chol', 'children'),
    State('value-loaded-fbs', 'children'),
    State('value-loaded-restecg', 'children'),
    State('value-loaded-thalach', 'children'),
    State('value-loaded-exang', 'children'),
    State('value-loaded-oldpeak', 'children'),
    State('value-loaded-slope', 'children'),
    State('value-loaded-ca', 'children'),
    State('value-loaded-thal', 'children'),
    ]
)
def execute_classification(n_clicks, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """
    Main method. Loads the trained model, applies the input data and returns a class
    """

    if(n_clicks == None): # When the application open
        return "Press below to execute the classification"
    else:
        # The sliders' values are already parsed to numeric values
        # Here we create a DataFrame with the input data
        data_from_user = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        global sample
        sample = pd.DataFrame(data=[data_from_user], columns=dataset_colnames)
        print(sample)

        sample = helper_dash_example.normalize_data(sample)

        # Execute the prediction using the loaded trained model.
        prediction = loaded_model.predict(sample)

        # Return final message
        prediction_labels = ["No Heart Disease", "Heart Disease Detected"]
        print("The predicted class of the input data is: ["+ str(int(prediction[0])) +":" + prediction_labels[int(prediction[0])] + "]")
        return "Results: "+ prediction_labels[int(prediction[0])] + ""



# Run the web server when this script is executed in Python
if __name__ == "__main__":
    app.run_server(debug=True)