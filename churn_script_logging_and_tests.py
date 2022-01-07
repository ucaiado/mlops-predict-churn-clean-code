'''
Implement tests to clasees and functions implemented in churn_library

Author: ucaiado
Date: January 4th, 2022
'''
import os
import sys
import logging
import yaml
import pathlib
import pandas as pd
import churn_library as cls



'''
Begin setup structures and global variables
'''

# load confs
CONFS = yaml.safe_load(open('confs/churn_library.yml', 'r'))


# set logging to write a file and to stodout
logging.basicConfig(
    filename=CONFS.get('log_path'),
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))



'''
End setup structures and global variables
'''


def test_import(import_data: object) -> pd.DataFrame:
    '''
    test data import
    '''
    try:
        df = import_data(CONFS.get('data_path'))
        logging.info("Testing import_data: function SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: Checking Data SUCCESS")
    except AssertionError as err:
        s_msg = (
            "Testing import_data: Checking Data ERROR"
            "\n\t!! Testing import_data: The file doesn't appear "
            "to have rows and columns")
        logging.error(s_msg)
        raise err
    return df


def test_eda(perform_eda: object, df_data: pd.DataFrame) -> None:
    '''
    test perform eda function
    '''
    # try to run eda function
    try:
        perform_eda(df_data)
        logging.info("Testing test_eda: function SUCCESS")
    except Exception as err:
        logging.error("Testing test_eda: function ERROR")
        raise err

    # check the outputs
    s_path2save = CONFS.get('eda_folder')
    for s_col in CONFS.get('histogram_plots') + CONFS.get('other_plots'):
        try:
            assert pathlib.Path(f'{s_path2save}{s_col}.png').is_file()
        except AssertionError as err:
            s_msg = (f"Testing test_eda: Checking plot files ERROR"
                     f"\n\t!! File {s_col}.png does not exist")
            logging.error(s_msg)
            raise err
    logging.info("Testing test_eda: Checking plot files SUCCESS")


def test_encoder_helper(
    encoder_helper: object, df_data: pd.DataFrame) -> pd.DataFrame:
    '''
    test encoder helper
    '''
    # try to run encoder_helper function
    try:
        df_data = encoder_helper(
            raw_data=df_data,
            category_lst=CONFS.get('encoding_columns'),
            response='Churn')
        logging.info("Testing encoder_helper: function SUCCESS")
    except AssertionError as err:
        s_msg = "Testing encoder_helper: function ERROR"
        logging.error(s_msg)
        raise err

    # check the outputs
    l_expected_cols = [f'{x}_Churn' for x in CONFS.get('encoding_columns')]
    l_cols_found = set(df_data.columns).intersection(set(l_expected_cols))
    try:
        assert len(l_expected_cols) == len(l_cols_found)
        logging.info("Testing encoder_helper: Checking Data SUCCESS")
    except AssertionError as err:
        s_msg = (
            "Testing encoder_helper: Checking Data ERROR"
            f"\n\t!! Testing encoder_helper: Was expected "
            f"new {len(l_expected_cols)} columns. "
            f"Just found {len(l_cols_found)} columns.")
        logging.error(s_msg)
        raise err
    return df_data


def test_perform_feature_engineering(
    perform_feature_engineering: object,
    df_data: pd.DataFrame) -> cls.Features:
    '''
    test perform_feature_engineering
    '''
    # try to run encoder_helper function
    try:
        obj_features = perform_feature_engineering(enconded_data=df_data)
        logging.info("Testing perform_feature_engineering: function SUCCESS")
    except AssertionError as err:
        s_msg = "Testing perform_feature_engineering: function ERROR"
        logging.error(s_msg)
        raise err

    # check the columns from x_train
    l_expected_cols = CONFS.get('keep_cols')
    l_cols_found = set(obj_features.x_train.columns).intersection(
        set(l_expected_cols))
    try:
        assert len(l_expected_cols) == len(l_cols_found)
        logging.info(
            "Testing perform_feature_engineering: Checking columns SUCCESS")
    except AssertionError as err:
        s_msg = (
            "Testing perform_feature_engineering: Checking columns ERROR"
            f"\n\t!! Testing perform_feature_engineering: Was expected "
            f"new {len(l_expected_cols)} columns. "
            f"Just found {len(l_cols_found)} columns.")
        logging.error(s_msg)
        raise err

    # check data difference length
    f_diff_size = obj_features.x_test.shape[0]
    f_diff_size = f_diff_size / (f_diff_size + obj_features.x_train.shape[0])
    try:
        assert abs(f_diff_size - 0.3) <= 0.01
        logging.info(
            "Testing perform_feature_engineering: Checking split size SUCCESS")
    except AssertionError as err:
        s_msg = (
            "Testing perform_feature_engineering: Checking split size ERROR"
            f"\n\t!! Testing perform_feature_engineering: Was expected "
            f"30% in test size. It is {f_diff_size*100:.0f}%.")
        logging.error(s_msg)
        raise err

    return obj_features


def test_train_models(
    train_models: object, obj_features: cls.Features) -> None:
    '''
    test train_models
    '''
    # try to run train_models function
    try:
        d_models = train_models(obj_features=obj_features)
        logging.info("Testing train_models: function SUCCESS")
    except AssertionError as err:
        s_msg = "Testing train_models: function ERROR"
        logging.error(s_msg)
        raise err

    # check the stored models
    try:
        for s_model in d_models:
            s_model_path = CONFS['models'][s_model].get('path')
            assert pathlib.Path(s_model_path).is_file()
        logging.info(
            "Testing train_models: Checking model pkl files SUCCESS")
    except AssertionError as err:
        s_msg = ("Testing train_models: Checking model pkl files ERROR")
        logging.error(s_msg)
        raise err

    # check the Feature Importance plot
    try:
        s_model = 'RandomForestClassifier'
        s_plot_name = 'FeatureImportances'
        s_plot_path = (f"{CONFS['models'][s_model].get('plot_path')}"
                       f"{s_plot_name}.png")
        assert pathlib.Path(s_plot_path).is_file()
        logging.info(
            "Testing train_models: Checking Feature Importance plot SUCCESS")
    except AssertionError as err:
        s_msg = ("Testing train_models: Checking Feature Importance plot ERROR")
        logging.error(s_msg)
        raise err


if __name__ == "__main__":
    df_data = test_import(cls.import_data)
    test_eda(cls.perform_eda, df_data)
    df_data = test_encoder_helper(cls.encoder_helper, df_data)
    obj_features = test_perform_feature_engineering(
        cls.perform_feature_engineering, df_data)
    test_train_models(cls.train_models, obj_features)









