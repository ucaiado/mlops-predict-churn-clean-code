'''
Implement functions to predict the company's churn

Author: ucaiado
Date: January 4th, 2022
'''

# import libraries
from dataclasses import dataclass
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt


# setup environment and global variables
with open('confs/churn_library.yml', 'r', encoding="utf-8") as conf_file:
    CONFS = yaml.safe_load(conf_file)

sns.set()


# define help structures
def _save_figure(fig: plt.Figure, s_path2save: str, s_name: str) -> None:
    plt.tight_layout()
    fig.savefig(
        f'{s_path2save}{s_name}.png',
        format='png',
        bbox_inches='tight')


@dataclass
class Features:
    '''Class for keeping data splitted to traing models'''
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class ModelPredictions:
    '''Class for keeping data related to predictions on Features'''
    model_name: str
    y_train: pd.Series
    y_test: pd.Series


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
    raw_data :  pandas dataframe
        Data Matrix
    '''
    raw_data = pd.read_csv(s_pth)
    raw_data['Churn'] = raw_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return raw_data


def perform_eda(raw_data: pd.DataFrame) -> None:
    '''
    perform eda on raw_data and save figures to images folder

    Parameters
    ----------
    raw_data :  pandas dataframe
        Data Matrix

    Returns
    -------
    None
    '''
    # create histogramns
    s_path2save = CONFS.get('eda_folder')
    for s_col in CONFS.get('histogram_plots'):
        fig = plt.figure(figsize=(20, 10))
        raw_data[s_col].hist()
        _save_figure(fig, s_path2save, s_col)

    # create value count plot from Marital_Status col
    s_col = 'Marital_Status'
    fig = plt.figure(figsize=(20, 10))
    raw_data[s_col].value_counts('normalize').plot(
        kind='bar', figsize=(20, 10))
    _save_figure(fig, s_path2save, s_col)

    # create distribution plot from Total_Trans_Ct col
    s_col = 'Total_Trans_Ct'
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(raw_data[s_col], kde=True)
    _save_figure(fig, s_path2save, s_col)

    # create a heatmap from correlations between all columns
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(raw_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    _save_figure(fig, s_path2save, 'HeatMap')


def encoder_helper(
    raw_data: pd.DataFrame,
    category_lst: list,
    response: str = None
) -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    Parameters
    ----------
    raw_data: pandas dataframe
    category_lst: list
        List of columns that contain categorical features
    response: str (optional)
        Response name [could be used for naming variables or index y column]

    Returns
    -------
    enconded_data: pandas dataframe
        Initial data with new columns
    '''
    enconded_data = raw_data.copy()
    for s_category in category_lst:
        df_category_groups = raw_data.groupby(s_category).mean()['Churn']
        s_new_col = s_category
        if not isinstance(response, type(None)):
            s_new_col = f'{s_category}_{response}'
        enconded_data[s_new_col] = raw_data[s_category].map(df_category_groups)

    return enconded_data


def perform_feature_engineering(enconded_data: pd.DataFrame) -> Features:
    '''
    Parameters
    ----------
    enconded_data: pandas dataframe

    Returns
    -------
    obj_features: Features
        Struct holding X training, X testing, y training, y testing data
    '''
    # split data
    l_keep_cols = CONFS.get('keep_cols')
    x_data = enconded_data[l_keep_cols]
    y_data = enconded_data['Churn']
    t_rtn = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    # instantiate Features object to return
    obj_features = Features(
        x_train=t_rtn[0],
        x_test=t_rtn[1],
        y_train=t_rtn[2],
        y_test=t_rtn[3],
    )

    return obj_features


def classification_report_image(
    obj_features: Features,
    obj_preds: ModelPredictions,
    output_pth: str
) -> None:
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

    # create figure
    fig = plt.figure(figsize=(5, 5))
    plt.rc('figure')
    s_model = obj_preds.model_name

    # fill in the figure with classification_report from train set
    plt.text(0.01, 0.6, str(f'{s_model} Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(
        obj_preds.y_train,
        obj_preds.y_train)),
        {'fontsize': 10}, fontproperties='monospace')

    # fill in the figure with classification_report from test set
    plt.text(0.01, 1.25, str(f'{s_model} Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(
        obj_features.y_test,
        obj_preds.y_test)),
        {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')

    # store figure
    _save_figure(fig, output_pth, f'{s_model}_ClassificationReport')


def feature_importance_plot(
        model: object,
        obj_features: Features,
        output_pth: str) -> None:
    '''
    Creates and stores the feature importances in output_pth

    Parameters
    ----------
    model: Scikit-learn Model object
        model object containing feature_importances_
    obj_features: Features
        Struct holding X training, X testing, y training, y testing data
    output_pth: str
        path to store the figure

    Returns
    -------
    None
    '''
    #  Sort feature importances in descending order
    na_importances = model.feature_importances_
    na_indices = np.argsort(na_importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    l_names = [obj_features.x_train.columns[i] for i in na_indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(obj_features.x_train.shape[1]), na_importances[na_indices])
    plt.xticks(range(obj_features.x_train.shape[1]), l_names, rotation=90)

    # store figure
    _save_figure(fig, output_pth, 'FeatureImportances')


def train_models(obj_features: Features) -> dict:
    '''
    train, store model results: images + scores, and store models

    Parameters
    ----------
    obj_features: Features
        Struct holding X training, X testing, y training, y testing data

    Returns
    -------
    d_models: dict
        Dictionary with the models created and indexed by name
    '''

    # initialize models
    d_rfc_confs = CONFS['models'].get('RandomForestClassifier')
    d_lrc_confs = CONFS['models'].get('LogisticRegression')

    rfc = RandomForestClassifier(random_state=d_rfc_confs.get('random_state'))
    lrc = LogisticRegression(solver=d_lrc_confs.get('solver'))

    # perform grid search on Random Forest
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=d_rfc_confs.get('param_grid'),
        cv=d_rfc_confs.get('grid_cv'))
    cv_rfc.fit(obj_features.x_train, obj_features.y_train)

    # fit logistic regression
    lrc.fit(obj_features.x_train, obj_features.y_train)

    # store models
    d_models = {model.__class__.__name__: model for model in
                [cv_rfc.best_estimator_, lrc]}
    for s_model in d_models:
        s_path = CONFS['models']['RandomForestClassifier'].get('path')
        joblib.dump(d_models[s_model], s_path)

    # analize models created
    s_plot_path = d_rfc_confs.get('plot_path')
    feature_importance_plot(
        model=d_models['RandomForestClassifier'],
        obj_features=obj_features,
        output_pth=s_plot_path)

    for s_model in d_models:
        model = d_models[s_model]
        obj_tests = ModelPredictions(
            model_name=s_model,
            y_train=model.predict(obj_features.x_train),
            y_test=model.predict(obj_features.x_test)
        )
        classification_report_image(
            obj_features=obj_features,
            obj_preds=obj_tests,
            output_pth=s_plot_path)

    return d_models


def main():
    '''
    Load data from data_path in conf file and train and evaluate some models
    '''
    # import data
    raw_data = import_data(CONFS.get('data_path'))

    # perform EDA
    perform_eda(raw_data)

    # enconde columns
    encoded_data = encoder_helper(
        raw_data=raw_data,
        category_lst=CONFS.get('encoding_columns'),
        response='Churn')

    # create features
    obj_features = perform_feature_engineering(enconded_data=encoded_data)

    # train and evaluate models
    _ = train_models(obj_features=obj_features)


if __name__ == '__main__':
    main()
