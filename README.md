# Disaster Response Machine Learning Pipeline Project

This repo contains the implementation for ETL and ML pipeline where the input messages are classified into different categories to identify the disaster responses. There is also a GUI/Website written in Flask Web App.

### Dependencies

* python 3.11
* numpy

* sqlalchemy
* re

* pickle
* nltk

* scikit-learn

### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database (replace text in [...] by the variable name of your choice):

     ```
     python process_data.py disaster_messages.csv disaster_categories.csv [database_name].db
     ```
   - To run ML pipeline that trains classifier and save the model (replace text in [...] by the variable name of your choice):

     ```
     python train_classifier.py ../data/[database_name].db [model_name].pkl
     ```
2. To run the web app, go to `app` directory: `cd app`
3. Run your web app: `python run.py`
4. Open a web browser of your choice, and enter the localhost:port address that was selected in the `run.py`
5. To classify messages, enter a message in the textbox and click Classify Message

### Files

disaster_response
├── README.md
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── models
│   ├── classifier.pkl
│   └── train_classifier.py
└── notebooks
    ├── ETL Pipeline Preparation.ipynb
    └── ML Pipeline Preparation.ipynb
