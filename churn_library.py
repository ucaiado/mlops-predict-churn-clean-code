'''
Implement functions to predict the company's churn

Author: ucaiado
Date: January 4th, 2022
'''

# import libraries
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# setup environment and global variables
with open('confs/churn_library.yml', 'r', encoding="utf-8") as conf_file:
    CONFS = yaml.safe_load(conf_file)

sns.set()


# define help functions
def _save_figure(fig: plt.Figure, s_path2save: str, s_name: str) -> None:
    plt.tight_layout()
    fig.savefig(
        f'{s_path2save}{s_name}.png',
        format='png',
        bbox_inches='tight')


# implement mains functions

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


def perform_eda(df_data: pd.DataFrame) -> None:
    '''
    perform eda on df_data and save figures to images folder Parameters

    ----------
    df_data :  pandas dataframe
        Data Matrix

    Returns
    -------
    None
    '''
    # create histogramns
    s_path2save = CONFS.get('image_folder')
    for s_col in CONFS.get('histogram_plots'):
        fig = plt.figure(figsize=(20, 10))
        df_data[s_col].hist()
        _save_figure(fig, s_path2save, s_col)

    # create value count plot from Marital_Status col
    s_col = 'Marital_Status'
    fig = plt.figure(figsize=(20, 10))
    df_data[s_col].value_counts('normalize').plot(
        kind='bar', figsize=(20, 10))
    _save_figure(fig, s_path2save, s_col)

    # create distribution plot from Total_Trans_Ct col
    s_col = 'Total_Trans_Ct'
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(df_data[s_col], kde=True)
    _save_figure(fig, s_path2save, s_col)

    # create a heatmap from correlations between all columns
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    _save_figure(fig, s_path2save, 'HeatMap')


def encoder_helper(df_data, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df_data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df_data: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df_data, response):
    '''
    input:
              df_data: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

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
    produces classification report for training and testing results and stores
    report as image in images folder
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
