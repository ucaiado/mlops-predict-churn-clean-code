'''
Implement functions to predict the company's churn

Author: ucaiado
Date: January 4th, 2022
'''

# import libraries
import numpy as np
import pandas as pd
import yaml


'''
Begin setup structures and global variables
'''

# load confs
CONFS = yaml.safe_load(open('confs/churn_library.yml', 'r'))


'''
End setup structures and global variables
'''


def import_data(s_pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    Parameters
    ----------
    s_pth : str
        A path to the csv

    Returns
    -------
    df_data :  pandas dataframe
        Data Matrix
    '''
    df_data = pd.read_csv(s_pth)
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df_data


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    pass


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass
