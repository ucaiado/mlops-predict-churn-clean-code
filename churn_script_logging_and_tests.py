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
    s_path2save = CONFS.get('image_folder')
    for s_col in CONFS.get('histogram_plots') + CONFS.get('other_plots'):
        try:
            assert pathlib.Path(f'{s_path2save}{s_col}.png').is_file()
        except AssertionError as err:
            s_msg = (f"Testing test_eda: Checking plot files ERROR"
                     f"\n\t!! File {s_col}.png does not exist")
            logging.error(s_msg)
            raise err
    logging.info("Testing test_eda: Checking plot files SUCCESS")


def test_encoder_helper(encoder_helper: object, df_data: pd.DataFrame) -> None:
    '''
    test encoder helper
    '''
    # try to run encoder_helper function
    try:
        df_data = encoder_helper(
            df_data=df_data,
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
            f"new {len(l_expected_cols)} columns."
            f"Just found {len(l_cols_found)} columns.")
        logging.error(s_msg)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    pass


def test_train_models(train_models):
    '''
    test train_models
    '''
    pass


if __name__ == "__main__":
    df_data = test_import(cls.import_data)
    test_eda(cls.perform_eda, df_data)
    test_encoder_helper(cls.encoder_helper, df_data)








