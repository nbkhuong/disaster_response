# Disaster Response Machine Learning Pipeline Project

This project leverages data engineering and machine learning techniques to build a system that classifies disaster messages for efficient emergency response. Using real-world data from Appen, a machine learning pipeline is developed to categorize messages into various disaster-related categories. The project includes a web application where emergency workers can input new messages and receive classification results. It also incorporates visualizations to provide insights into the data. The project demonstrates proficiency in creating data pipelines, building machine learning models, and developing user-friendly web applications.

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
```
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
```
