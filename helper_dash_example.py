# This code was based on the resources provided in the Data Mining for Computer and Systems Sciences (DAMI) course 2020 and originally created by luis-eduardo@dsv.su.se
#
# =============================================================================
# Imports
# =============================================================================

import dash
import dash_core_components as dcc
import dash_html_components as html

from pathlib import Path
import pandas as pd


# =============================================================================
# Functions
# =============================================================================

def select_dropdown_value(index):
    return sample_data.iloc[index]

def normalize_data(sample):
    sample['age'] = (sample['age'] - training_data['age'].min()) / (training_data['age'].max() - training_data['age'].min())
    sample['sex'] = (sample['sex'] - training_data['sex'].min()) / (training_data['sex'].max() - training_data['sex'].min())
    sample['cp'] = (sample['cp'] - training_data['cp'].min()) / (training_data['cp'].max() - training_data['cp'].min())
    sample['trestbps'] = (sample['trestbps'] - training_data['trestbps'].min()) / (training_data['trestbps'].max() - training_data['trestbps'].min())
    sample['chol'] = (sample['chol'] - training_data['chol'].min()) / (training_data['chol'].max() - training_data['chol'].min())
    sample['fbs'] = (sample['fbs'] - training_data['fbs'].min()) / (training_data['fbs'].max() - training_data['fbs'].min())
    sample['restecg'] = (sample['restecg'] - training_data['restecg'].min()) / (training_data['restecg'].max() - training_data['restecg'].min())
    sample['thalach'] = (sample['thalach'] - training_data['thalach'].min()) / (training_data['thalach'].max() - training_data['thalach'].min())
    sample['exang'] = (sample['exang'] - training_data['exang'].min()) / (training_data['exang'].max() - training_data['exang'].min())
    sample['oldpeak'] = (sample['oldpeak'] - training_data['oldpeak'].min()) / (training_data['oldpeak'].max() - training_data['oldpeak'].min())
    sample['slope'] = (sample['slope'] - training_data['slope'].min()) / (training_data['slope'].max() - training_data['slope'].min())
    sample['ca'] = (sample['ca'] - training_data['ca'].min()) / (training_data['ca'].max() - training_data['ca'].min())
    sample['thal'] = (sample['thal'] - training_data['thal'].min()) / (training_data['thal'].max() - training_data['thal'].min())

    return sample

# =============================================================================
# Main
# =============================================================================


#############
"""
Load and simple processing of the original dataset for visualization
purposes in the web application.
"""

# Relative paths respect to current file
# DO NOT MODIFY: Relative path prefix to be able to find the dataset
THIS_FILE_PATH = str(Path(__file__).parent.absolute())+"/"
#FOLDER_PATH = THIS_FILE_PATH + "../../datasets/"
FOLDER_PATH = THIS_FILE_PATH + "/"

# Load original dataset file
dataset_filename = THIS_FILE_PATH + "to_predict.csv"
sample_data = pd.read_csv(dataset_filename, sep=",")

training_data = pd.read_csv(FOLDER_PATH + 'processed.cleveland.csv', sep = '\t')

training_data = training_data[(training_data["ca"]!="?")&(training_data["thal"]!="?")&(training_data["slope"]!="?")]

training_data['thal'] = training_data['thal'].astype(float)
training_data['ca'] = training_data['ca'].astype(float)

# Structure to map df column names to meaningful labels
colnames = sample_data.columns
#colnames = colnames.drop('target').values
colnames = colnames.drop('num').values
#column_labels = ["Area", "Perimeter", "Compactness", "Length of Kernel",
#                "Width of Kernel", "Asymmetry Coeff.", "Length Kernel Groove"]
column_labels = ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
                "Serum Cholesterol", "Fasting Blood Sugar", "Resting Electrocardiographic Results",
                "Maximum Hart Rate Achieved", "Exercise Induced Angina", "ST depression induced by exercise reative to rest",
                "Slope of the peak exercise ST segment", "Number of Major Vessels", "Thalassemia"]

#############
"""
Structure of the HTML webpage using Dash library
"""
app_html_layout = html.Div([

    html.Center(html.H1("Intelli-Cardio - Heart Disease Predictor")),

    #html.Div("This app classifies patient presence of Heart Disease from thirteem real-value attributes extracted from Electronic Health Records"),

    #html.Div(['More information about dataset:',
    #    html.A('https://archive.ics.uci.edu/ml/datasets/Heart+Disease')
    #]),

    html.H3('Custom parameters'),

    # Create the table to put input values
    html.Table([ html.Tbody([
        # Age
        html.Tr([
            html.Td( html.B('Age:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-age',
                    min=30,
                    max=80,
                    step=1,
                    value=60,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-age',children=''), style={'width':'10%'} ),
            ]),
        # Sex
        html.Tr([
            html.Td( html.B('Sex:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            dcc.Dropdown(
                id='slider-sex',
                options=[
                    {'label': 'Female (0)', 'value': 0},
                    {'label': 'Male (1)', 'value': 1},
                ],
                value=0
            ),
            #html.Td( dcc.Slider(id='slider-sex',
            #        min=0,
            #        max=1,
            #        step=1,
            #        value=1,
            #    ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-sex',children=''), style={'width':'20%'} ),
            ]),
        # Chest Pain Type
        html.Tr([
            html.Td( html.B('Chest Pain Type:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            dcc.Dropdown(
                id='slider-cp',
                options=[
                    {'label': 'typical angina (1)', 'value': 1},
                    {'label': 'atypical angina (2)', 'value': 2},
                    {'label': 'non-anginal pain (3)', 'value': 3},
                    {'label': 'asymptomatic (4)', 'value': 4},
                ],
                value=1
            ),
            #html.Td( dcc.Slider(id='slider-cp',
            #        min=1,
            #        max=4,
            #        step=1,
            #        value=4,
            #    ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-cp',children=''), style={'width':'20%'} ),
            ]),
        # Resting Blood Pressure trestbps
        html.Tr([
            html.Td( html.B('Resting Blood Pressure:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-trestbps',
                    min=80,
                    max=200,
                    step=1,
                    value=110,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-trestbps',children=''), style={'width':'20%'} ),
            ]),
        # Serum Cholesterol chol
        html.Tr([
            html.Td( html.B('Serum Cholesterol:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-chol',
                    min=120,
                    max=570,
                    step=1,
                    value=130,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-chol',children=''), style={'width':'20%'} ),
            ]),
        # Fasting Blood Sugar fbs
        html.Tr([
            html.Td( html.B('Fasting Blood Sugar (fasting blood sugar > 120 mg/dl):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            dcc.Dropdown(
                id='slider-fbs',
                options=[
                    {'label': 'false (0)', 'value': 0},
                    {'label': 'true (1)', 'value': 1},
                ],
                value=0
            ),
            #html.Td( dcc.Slider(id='slider-fbs',
            #        min=0,
            #        max=1,
            #        step=1,
            #        value=1,
            #    ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-fbs',children=''), style={'width':'20%'} ),
            ]),

        # Resting Electrocardiographic Results restecg
        html.Tr([
            html.Td( html.B('Resting Electrocardiographic Results (restecg):', style={'font-size':'9pt'}), style={'width':'25%'} ),
            dcc.Dropdown(
                id='slider-restecg',
                options=[
                    {'label': 'normal (0)', 'value': 0},
                    {'label': 'having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) (1)', 'value': 1},
                    {'label': 'showing probable or definite left ventricular hypertrophy by Estes\' criteria (2)', 'value': 2},
                ],
                value=0
            ),
            #html.Td( dcc.Slider(id='slider-restecg', #restecg
            #        min=0,
            #        max=2,
            #        step=1,
            #        value=2,
            #    ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-restecg',children=''), style={'width':'20%'} ),
            ]),

        # Maximum Hart Rate Achieved thalach
        html.Tr([
            html.Td( html.B('Maximum Hart Rate Achieved:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-thalach',
                    min=70,
                    max=205,
                    step=1,
                    value=170,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-thalach',children=''), style={'width':'20%'} ),
            ]),

        # Exercise Induced Angina exang
        html.Tr([
            html.Td( html.B('Exercise Induced Angina:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            dcc.Dropdown(
                id='slider-exang',
                options=[
                    {'label': 'No (0)', 'value': 0},
                    {'label': 'Yes (1)', 'value': 1},
                ],
                value=0
            ),
            #html.Td( dcc.Slider(id='slider-exang',
            #        min=0,
            #        max=1,
            #        step=1,
            #        value=1,
            #    ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-exang',children=''), style={'width':'20%'} ),
            ]),

        # ST depression induced by exercise reative to rest oldpeak
        html.Tr([
            html.Td( html.B('ST depression induced by exercise reative to rest:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-oldpeak',
                    min=0,
                    max=6.5,
                    step=0.1,
                    value=1,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-oldpeak',children=''), style={'width':'20%'} ),
            ]),

        # Slope of the peak exercise ST segment slope
        html.Tr([
            html.Td( html.B('Slope of the peak exercise ST segment:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            dcc.Dropdown(
                id='slider-slope',
                options=[
                    {'label': 'upsloping (1)', 'value': 1},
                    {'label': 'flat (2)', 'value': 2},
                    {'label': 'downsloping (3)', 'value': 3},
                ],
                value=1
            ),
            #html.Td( dcc.Slider(id='slider-slope',
            #        min=1,
            #        max=3,
            #        step=1,
            #        value=1,
            #    ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-slope',children=''), style={'width':'20%'} ),
            ]),

        # Number of Major Vessels ca
        html.Tr([
            html.Td( html.B('Number of Major Vessels:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( dcc.Slider(id='slider-ca',
                    min=0,
                    max=3,
                    step=1,
                    value=2,
                ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-ca',children=''), style={'width':'20%'} ),
            ]),

        # thal thal
        html.Tr([
            html.Td( html.B('Thalassemia:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            dcc.Dropdown(
                id='slider-thal',
                options=[
                    {'label': 'Normal (3)', 'value': 3},
                    {'label': 'Fixed defect (6)', 'value': 6},
                    {'label': 'Reversable defect (7)', 'value': 7}
                ],
                value=3
            ),
            # html.Td( dcc.Slider(id='slider-thal',
            #         min=3,
            #         max=7,
            #         step=1,
            #         value=6,
            #     ), style={'width':'55%'} ),
            html.Td( html.P(id='value-slider-thal',children=''), style={'width':'20%'} ),
            ]),
        ]),
    ], style={'width':'100%', 'padding':'0', 'margin':'0'}),

    html.Center(
        html.Div([
            html.Br(),
            html.H4(html.B('Classification result', id='classification-result', style={'color':'#983e0f'})),
            html.Button('Execute Classification', id='submit', style={'margin':'0 auto', 'width':'30%'}),
        ])
    ),

    html.Br(),

    html.Center(html.B('Possible classes: [0:Negative to Heart Disease], [1:Positive to Heart Disease]', style={'color':'#33C3F0'})),

    html.Hr(),


    html.H3('Select Patient'),

    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'Patient ' + str(i +1), 'value': i}
            for i in range(len(sample_data.index))
        ],
        value=1
    ),
    html.Div(id='dd-output-container'),

    # Create the table to put input values
    html.Table([ html.Tbody([
        # Age
        html.Tr([
            html.Td( html.B('Age:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-age',children=''), style={'width':'10%'} ),
            ]),
        # Sex
        html.Tr([
            html.Td( html.B('Sex:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-sex',children=''), style={'width':'20%'} ),
            ]),
        # Chest Pain Type
        html.Tr([
            html.Td( html.B('Chest Pain Type:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-cp',children=''), style={'width':'20%'} ),
            ]),
        # Resting Blood Pressure trestbps
        html.Tr([
            html.Td( html.B('Resting Blood Pressure:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-trestbps',children=''), style={'width':'20%'} ),
            ]),
        # Serum Cholesterol chol
        html.Tr([
            html.Td( html.B('Serum Cholesterol:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-chol',children=''), style={'width':'20%'} ),
            ]),
        # Fasting Blood Sugar fbs
        html.Tr([
            html.Td( html.B('Fasting Blood Sugar:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-fbs',children=''), style={'width':'20%'} ),
            ]),

        # Resting Electrocardiographic Results restecg
        html.Tr([
            html.Td( html.B('Resting Electrocardiographic Results:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-restecg',children=''), style={'width':'20%'} ),
            ]),

        # Maximum Hart Rate Achieved thalach
        html.Tr([
            html.Td( html.B('Maximum Hart Rate Achieved:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-thalach',children=''), style={'width':'20%'} ),
            ]),

        # Exercise Induced Angina exang
        html.Tr([
            html.Td( html.B('Exercise Induced Angina:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-exang',children=''), style={'width':'20%'} ),
            ]),

        # ST depression induced by exercise reative to rest oldpeak
        html.Tr([
            html.Td( html.B('ST depression induced by exercise reative to rest:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-oldpeak',children=''), style={'width':'20%'} ),
            ]),

        # Slope of the peak exercise ST segment slope
        html.Tr([
            html.Td( html.B('Slope of the peak exercise ST segment:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-slope',children=''), style={'width':'20%'} ),
            ]),

        # Number of Major Vessels ca
        html.Tr([
            html.Td( html.B('Number of Major Vessels:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-ca',children=''), style={'width':'20%'} ),
            ]),

        # thal thal
        html.Tr([
            html.Td( html.B('Thalassemia:', style={'font-size':'9pt'}), style={'width':'25%'} ),
            html.Td( html.P(id='value-loaded-thal',children=''), style={'width':'20%'} ),
            ]),
        ]),
    ], style={'width':'100%', 'padding':'0', 'margin':'0'}),

    html.Center(
        html.Div([
            html.Br(),
            html.H4(html.B('Classification result', id='classification-result-loaded', style={'color':'#983e0f'})),
            html.Button('Execute Classification', id='submit-loaded', style={'margin':'0 auto', 'width':'30%'}),
        ])
    ),

    html.Br(),

    html.Center(html.B('Possible classes: [0:Negative to Heart Disease], [1:Positive to Heart Disease]', style={'color':'#33C3F0'})),

    html.Hr(),

    ], style={'width': '100%'})