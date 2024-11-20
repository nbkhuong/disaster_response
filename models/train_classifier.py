import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.gaussian_process import GaussianProcessClassifier

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)
    df.dropna(inplace=True)

    X = df['message']
    Y = df.loc[:, 'related':]

    return X, Y, Y.columns


def tokenize(text):
    text_ = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    words = word_tokenize(text_)

    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return words


def build_model():
    clf = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', clf),
    ])

    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__max_depth': [10, 30],
        'clf__estimator__min_samples_leaf': [2, 5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    predicted = model.predict(X_test)
    results = pd.DataFrame(columns=['category', 'f1_score', 'precision', 'recall'])

    for idx, category in enumerate(category_names):
        precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test[category], predicted[:, idx], average='weighted')
        results.loc[idx] = [category, f1_score, precision, recall]
    
    print('Aggregated f1_score:', results['f1_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())

    return results


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()