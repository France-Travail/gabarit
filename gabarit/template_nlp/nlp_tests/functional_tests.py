#!/usr/bin/env python3
# Starts all functional tests
# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Libs unittest
import unittest
from unittest.mock import Mock
from unittest.mock import patch

# utils libs
import os
import sys
import shutil
import subprocess
import pandas as pd
import importlib.util
from pathlib import Path
from datetime import datetime

from test_template_nlp import utils
from test_template_nlp.models_training import (model_tfidf_svm, model_tfidf_gbt, model_tfidf_lgbm, model_tfidf_sgdc,
                                               model_tfidf_dense, model_embedding_lstm, model_embedding_lstm_attention,
                                               model_embedding_lstm_structured_attention, model_embedding_lstm_gru_gpu,
                                               model_embedding_cnn, model_pytorch_transformers,
                                               utils_models)


class Case1_e2e_pipeline(unittest.TestCase):
    '''Class to test the project end to end'''

    def test01_CreateSamples(self):
        '''Test of the file 0_create_samples.py'''
        print("Test of the file 0_create_samples.py")

        # "Basic" case
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_create_samples.py -f mono_class_mono_label.csv -n 15"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_15_samples.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_15_samples.csv", sep=';', encoding='utf-8')
        self.assertEqual(df.shape[0], 15)

        # Double files
        fonctionnement_double = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_create_samples.py -f mono_class_mono_label.csv multi_class_mono_label.csv -n 2000"
        self.assertEqual(subprocess.run(fonctionnement_double, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_2000_samples.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'multi_class_mono_label_2000_samples.csv')))
        df1 = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_2000_samples.csv", sep=';', encoding='utf-8')
        df2 = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/multi_class_mono_label_2000_samples.csv", sep=';', encoding='utf-8')
        self.assertEqual(df1.shape[0], 200)
        self.assertEqual(df2.shape[0], 200) # 200 row max

    def test02_GetEmbeddingDict(self):
        '''Test of the file 0_get_embedding_dict.py'''
        print("Test of the file 0_get_embedding_dict.py")

        # "Basic" case
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_get_embedding_dict.py -f custom.300.vec"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'custom.300.pkl')))

    def test03_MergeFiles(self):
        '''Test of the file 0_merge_files.py'''
        print("Test of the file 0_merge_files.py")

        # "Basic" case
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_merge_files.py -f mono_class_mono_label.csv multi_class_mono_label.csv -c x_col y_col -o merged_file.csv"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'merged_file.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/merged_file.csv", sep=';', encoding='utf-8')
        self.assertGreater(df.shape[0], 200) # We check that there are more than 200 elements (ie. the size of one of the two files)

    def test04_SplitTrainValidTest(self):
        '''Test of the file 0_split_train_valid_test.py'''
        print("Test of the file 0_split_train_valid_test.py")

        # "Basic" case
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_train.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_valid.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_test.csv')))
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 120)
        self.assertEqual(df_valid.shape[0], 40)
        self.assertEqual(df_test.shape[0], 40)

        # Test of perc_x arguments
        test_perc = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.3 --perc_valid 0.6 --perc_test 0.1 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_perc, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 60)
        self.assertEqual(df_valid.shape[0], 120)
        self.assertEqual(df_test.shape[0], 20)

        # Test split_type stratified
        test_stratified = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type stratified --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_stratified, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        # Check number of elements
        self.assertGreater(df_train.shape[0], 114)
        self.assertLess(df_train.shape[0], 126)
        self.assertGreater(df_valid.shape[0], 34)
        self.assertLess(df_valid.shape[0], 46)
        self.assertGreater(df_test.shape[0], 34)
        self.assertLess(df_test.shape[0], 46)
        # Check stratified
        self.assertGreater(sum(df_train.y_col == "oui")/df_train.shape[0], 0.47)
        self.assertLess(sum(df_train.y_col == "oui")/df_train.shape[0], 0.53)
        self.assertGreater(sum(df_valid.y_col == "oui")/df_valid.shape[0], 0.47)
        self.assertLess(sum(df_valid.y_col == "oui")/df_valid.shape[0], 0.53)
        self.assertGreater(sum(df_test.y_col == "oui")/df_test.shape[0], 0.47)
        self.assertLess(sum(df_test.y_col == "oui")/df_test.shape[0], 0.53)

        # Test split_type hierarchical
        test_hierarchical = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type hierarchical --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_hierarchical, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        # Check number of elements
        self.assertGreater(df_train.shape[0], 114)
        self.assertLess(df_train.shape[0], 126)
        self.assertGreater(df_valid.shape[0], 34)
        self.assertLess(df_valid.shape[0], 46)
        self.assertGreater(df_test.shape[0], 34)
        self.assertLess(df_test.shape[0], 46)
        # Check hierarchical
        self.assertFalse(any([_ in df_valid.x_col.values for _ in df_train.x_col.values]))
        self.assertFalse(any([_ in df_test.x_col.values for _ in df_train.x_col.values]))
        self.assertFalse(any([_ in df_valid.x_col.values for _ in df_test.x_col.values]))

    def test05_PreProcessData(self):
        '''Test of the file 1_preprocess_data.py'''
        print("Test of the file 1_preprocess_data.py")

        # "Basic" case
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py -f mono_class_mono_label_train.csv mono_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        # Check if exists
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_train_preprocess_P1.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_valid_preprocess_P1.csv')))
        df_train = pd.read_csv(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_train_preprocess_P1.csv'), sep=';', encoding='utf-8', skiprows=1)
        df_valid = pd.read_csv(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_valid_preprocess_P1.csv'), sep=';', encoding='utf-8', skiprows=1)
        # Check col preprocessed_text
        self.assertTrue('preprocessed_text' in df_train.columns)
        self.assertTrue('preprocessed_text' in df_valid.columns)
        # Check preprocess (at least lower)
        self.assertEqual(list(df_train.preprocessed_text.str.lower().values), list(df_train.preprocessed_text.values))
        self.assertEqual(list(df_valid.preprocessed_text.str.lower().values), list(df_valid.preprocessed_text.values))

    def test06_TrainingE2E(self):
        '''Test of the file 2_training.py'''
        print("Test of the file 2_training.py")

        # "Basic" case
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/2_training.py -f mono_class_mono_label_train_preprocess_P1.csv -x preprocessed_text -y y_col --filename_valid mono_class_mono_label_valid_preprocess_P1.csv"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_nlp-models', 'model_tfidf_svm') # tfidf svm by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertGreater(len(listdir), 0)

    def test07_PredictE2E(self):
        '''Test of the file 3_predict.py'''
        print("Test of the file 3_predict.py")

        # "Basic" case
        save_model_dir = os.path.join(full_path_lib, 'test_template_nlp-models', 'model_tfidf_svm') # tfidf svm by default
        listdir = os.listdir(os.path.join(save_model_dir))
        model_name = listdir[0]
        fonctionnement_basique = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/3_predict.py -f mono_class_mono_label_test.csv -x x_col -m {model_name}"
        self.assertEqual(subprocess.run(fonctionnement_basique, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_nlp-data', 'predictions', 'mono_class_mono_label_test')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

        # With y_col
        fonctionnement_with_y = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/3_predict.py -f mono_class_mono_label_test.csv -x x_col -y y_col -m {model_name}"
        self.assertEqual(subprocess.run(fonctionnement_with_y, shell=True).returncode, 0)
        # Check predictions
        listdir = sorted(os.listdir(os.path.join(save_predictions_dir)))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[-1], 'predictions_with_y_true.csv'))) # last folder

def test_model_mono_class_mono_label(test_class, test_model):
    '''Generic fonction to test a given model for mono-class/mono-label'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Try some functions
    # predict
    preds = test_model.predict(['cdi à temps complet', 'vous disposez du permis'])
    test_class.assertEqual(list(preds), ['non', 'oui'])
    # predict_proba
    index_non = test_model.list_classes.index('non')
    index_oui = test_model.list_classes.index('oui')
    probas = test_model.predict_proba(['cdi à temps complet', 'vous disposez du permis'])
    test_class.assertGreater(probas[0][index_non], 0.5)
    test_class.assertLess(probas[0][index_oui], 0.5)
    test_class.assertGreater(probas[1][index_oui], 0.5)
    test_class.assertLess(probas[1][index_non], 0.5)
    # predict w/ return_proba=True
    probas2 = test_model.predict(['cdi à temps complet', 'vous disposez du permis'], return_proba=True)
    test_class.assertGreater(probas2[0][index_non], 0.5)
    test_class.assertLess(probas2[0][index_oui], 0.5)
    test_class.assertGreater(probas2[1][index_oui], 0.5)
    test_class.assertLess(probas2[1][index_non], 0.5)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(['cdi à temps complet', 'vous disposez du permis'])
    test_class.assertEqual(list(pred_proba[0]), ['non', 'oui'])
    test_class.assertGreater(pred_proba[1][0][index_non], 0.5)
    test_class.assertLess(pred_proba[1][0][index_oui], 0.5)
    test_class.assertGreater(pred_proba[1][1][index_oui], 0.5)
    test_class.assertLess(pred_proba[1][1][index_non], 0.5)
    # get_predict_position
    # position start at 1
    test_class.assertEqual(list(test_model.get_predict_position(['cdi à temps complet', 'vous disposez du permis', 'titi'], ['oui', 'oui', 'toto'])), [2, 1, -1])
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['non', 'oui'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5) # Only 2 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=2)
    test_class.assertEqual([list(_) for _ in top_n], [['non', 'oui'], ['oui', 'non']])
    test_class.assertEqual([list(_) for _ in top_n_proba], [[probas[0][index_non], probas[0][index_oui]], [probas[1][index_oui], probas[1][index_non]]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['non', 'oui'])


class Case2_MonoClassMonoLabel(unittest.TestCase):
    '''Class to test the mono-class / mono-label case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for mono-class / mono-label case")

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py -f mono_class_mono_label_train.csv mono_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_TfidfSvm(self):
        '''Test of the model TF-IDF/SVM'''
        print('            ------------------ >     Test of the model TF-IDF/SVM     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_svm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       svc_params={'C': 1.0, 'max_iter': 10000},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_svm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         svc_params={'C': 1.0, 'max_iter': 10000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_svm_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              svc_params={'C': 1.0, 'max_iter': 10000},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSvm failed')

    def test03_Model_TfidfGbt(self):
        '''Test of the model TF-IDF/GBT'''
        print('            ------------------ >     Test of the model TF-IDF/GBT     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_gbt_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_gbt_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_gbt_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                            tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                            gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
            #                                            multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                            multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfGbt failed')

    def test04_Model_TfidfLgbm(self):
        '''Test of the model TF-IDF/LGBM'''
        print('            ------------------ >     Test of the model TF-IDF/LGBM     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_lgbm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_lgbm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model_2 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_lgbm_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # # We overfit on purpose !
            # test_model_3 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #          filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfLgbm failed')


    def test05_Model_TfidfDense(self):
        '''Test of the model TF-IDF/Dense'''
        print('            ------------------ >     Test of the model TF-IDF/Dense     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'tfidf_dense_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_dense.ModelTfidfDense(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           batch_size=16, epochs=20, patience=20,
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_TfidfDense failed')

    def test06_Model_EmbeddingLstm(self):
        '''Test of the model Embedding/LSTM'''
        print('            ------------------ >     Test of the model Embedding/LSTM     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                   filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstm failed')

    def test07_Model_EmbeddingLstmAttention(self):
        '''Test of the model Embedding/LSTM/Attention'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=16, epochs=40, patience=20,
                                                                                    max_sequence_length=60, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmAttention failed')

    def test08_Model_EmbeddingLstmGruGpu(self):
        '''Test of the model Embedding/LSTM/GRU'''
        print('            ------------------ >     Test of the model Embedding/LSTM/GRU     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_gru_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_gru_gpu.ModelEmbeddingLstmGruGpu(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                               batch_size=16, epochs=20, patience=20,
                                                                               max_sequence_length=60, max_words=100000,
                                                                               embedding_name="custom.300.pkl",
                                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmGruGpu failed')

    def test09_Model_EmbeddingCnn(self):
        '''Test of the model Embedding/CNN'''
        print('            ------------------ >     Test of the model Embedding/CNN     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_cnn_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_cnn.ModelEmbeddingCnn(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                               batch_size=16, epochs=20, patience=20,
                                                               max_sequence_length=60, max_words=100000,
                                                               embedding_name="custom.300.pkl",
                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingCnn failed')

    def test10_Model_Keras_continue_training(self):
        '''Test continuing a training for a keras model'''
        print("            ------------------ >     Test continuing a training for a keras model     /   Mono-class & Mono-label")

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Run a first training
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertEqual(model_dir, test_model.model_dir)

            # Retrieve model & run a second training
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_Keras_continue_training failed')

    def test11_Model_TfidfSgdc(self):
        '''Test of the model TF-IDF/SGDClassifier'''
        print('            ------------------ >     Test of the model TF-IDF/SGDClassifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_sgdc_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         sgdc_params={'loss': 'hinge', 'max_iter': 1000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_sgdc_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           sgdc_params={'loss': 'log', 'max_iter': 1000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_sgdc_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                sgdc_params={'loss': 'log', 'max_iter': 1000},
            #                                                multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSgdc failed')

    def test12_Model_PytorchTransformer(self):
        '''Test of the model Pytorch Transformer'''
        transformers_path = utils.get_transformers_path()
        transformer_path = os.path.join(transformers_path, 'flaubert', 'flaubert_small_cased')
        if not os.path.exists(transformer_path):
            print("WARNING : Can't test the Pytorch Transformer model -> can't find transformer")
            print("How to use : download flaubert_small_cased in the folder of the module to test")
            print("We ignore this test.")
            return None
        print('            ------------------ >     Test of the model Pytorch Transformer     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'pytorch_transformer_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_pytorch_transformers.ModelPyTorchTransformers(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                             batch_size=16, epochs=20, patience=20,
                                                                             max_sequence_length=60,
                                                                             transformer_name='flaubert/flaubert_small_cased',
                                                                             tokenizer_special_tokens=tuple(),
                                                                             padding="max_length", truncation=True,
                                                                             multi_label=False, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_PytorchTransformer failed')

    def test013_Model_EmbeddingLstmStructuredAttention(self):
        '''Test of the model Embedding/LSTM/Attention + explainable'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention + explainable     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                                         batch_size=16, epochs=40, patience=20,
                                                                                                         max_sequence_length=60, max_words=100000,
                                                                                                         embedding_name="custom.300.pkl",
                                                                                                         multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('ModelEmbeddingLstmStructuredAttention failed')

def test_model_mono_class_multi_label(test_class, test_model):
    '''Generic fonction to test a given model for mono-class/multi-labels'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Try some functions
    index_col_1 = test_model.list_classes.index('y_col_1')
    index_col_2 = test_model.list_classes.index('y_col_2')
    pred_none = [0, 0]
    pred_col_1 = [0, 0]
    pred_col_1[index_col_1] = 1
    pred_col_2 = [0, 0]
    pred_col_2[index_col_2] = 1
    pred_all = [1, 1]
    # predict
    preds = test_model.predict(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual([list(_) for _ in preds], [pred_none, pred_col_1, pred_col_2, pred_all])
    # predict_proba
    probas = test_model.predict_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertLess(probas[0][index_col_1], 0.5)
    test_class.assertLess(probas[0][index_col_2], 0.5)
    test_class.assertGreater(probas[1][index_col_1], 0.5)
    test_class.assertLess(probas[1][index_col_2], 0.5)
    test_class.assertLess(probas[2][index_col_1], 0.5)
    test_class.assertGreater(probas[2][index_col_2], 0.5)
    test_class.assertGreater(probas[3][index_col_1], 0.5)
    test_class.assertGreater(probas[3][index_col_2], 0.5)
    # predict w/ return_proba=True
    probas2 = test_model.predict(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'], return_proba=True)
    test_class.assertLess(probas2[0][index_col_1], 0.5)
    test_class.assertLess(probas2[0][index_col_2], 0.5)
    test_class.assertGreater(probas2[1][index_col_1], 0.5)
    test_class.assertLess(probas2[1][index_col_2], 0.5)
    test_class.assertLess(probas2[2][index_col_1], 0.5)
    test_class.assertGreater(probas2[2][index_col_2], 0.5)
    test_class.assertGreater(probas2[3][index_col_1], 0.5)
    test_class.assertGreater(probas2[3][index_col_2], 0.5)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual([list(_) for _ in pred_proba[0]], [pred_none, pred_col_1, pred_col_2, pred_all])
    test_class.assertLess(pred_proba[1][0][index_col_1], 0.5)
    test_class.assertLess(pred_proba[1][0][index_col_2], 0.5)
    test_class.assertGreater(pred_proba[1][1][index_col_1], 0.5)
    test_class.assertLess(pred_proba[1][1][index_col_2], 0.5)
    test_class.assertLess(pred_proba[1][2][index_col_1], 0.5)
    test_class.assertGreater(pred_proba[1][2][index_col_2], 0.5)
    test_class.assertGreater(pred_proba[1][3][index_col_1], 0.5)
    test_class.assertGreater(pred_proba[1][3][index_col_2], 0.5)
    # get_predict_position
    # position start at 1
    with test_class.assertRaises(ValueError):
        test_model.get_predict_position(['toto', 'tata', 'toto', 'titi'], ['toto', 'tata', 'toto', 'titi'])  # Does not work with multi-labels
    # get_classes_from_proba
    test_class.assertEqual([list(_) for _ in test_model.get_classes_from_proba(probas)], [pred_none, pred_col_1, pred_col_2, pred_all])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=2) # Does not work with multi-labels
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), [(), ('y_col_1',), ('y_col_2',), ('y_col_1', 'y_col_2')])


class Case3_MonoClassMultiLabel(unittest.TestCase):
    '''Class to test the mono-class / multi-labels case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the mono-class / multi-labels case")

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_multi_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py -f mono_class_multi_label_train.csv mono_class_multi_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_TfidfSvm(self):
        '''Test of the model TF-IDF/SVM'''
        print('            ------------------ >     Test of the model TF-IDF/SVM     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_svm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       svc_params={'C': 1.0, 'max_iter': 10000},
                                                       multi_label=True, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_svm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         svc_params={'C': 1.0, 'max_iter': 10000},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_svm_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              svc_params={'C': 1.0, 'max_iter': 10000},
            #                                              multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #           filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSvm failed')

    def test03_Model_TfidfGbt(self):
        '''Test of the model TF-IDF/GBT'''
        print('            ------------------ >     Test of the model TF-IDF/GBT     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_gbt_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=True, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_gbt_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=True, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_gbt_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                            tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                            gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
            #                                            multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                            multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #           filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfGbt failed')

    def test04_Model_TfidfLgbm(self):
        '''Test of the model TF-IDF/LGBM'''
        print('            ------------------ >     Test of the model TF-IDF/LGBM     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_lgbm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                     filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_lgbm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model_2 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                     filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_lgbm_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # # We overfit on purpose !
            # test_model_3 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
            #                                              multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #          filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfLgbm failed')

    def test05_Model_TfidfDense(self):
        '''Test of the model TF-IDF/Dense'''
        print('            ------------------ >     Test of the model TF-IDF/Dense     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'tfidf_dense_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_dense.ModelTfidfDense(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           batch_size=16, epochs=20, patience=20,
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                     filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_TfidfDense failed')

    def test06_Model_EmbeddingLstm(self):
        '''Test of the model Embedding/LSTM'''
        print('            ------------------ >     Test of the model Embedding/LSTM     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstm failed')

    # TODO : Make sure that this test passes everytime by making it more stable
    @unittest.skip("The test of the LSTM model with attention + Mono-class & Multi-labels is unstable, for now, we skip it !")
    def test07_Model_EmbeddingLstmAttention(self):
        '''Test of the model Embedding/LSTM/Attention'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=16, epochs=40, patience=40,
                                                                                    max_sequence_length=60, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmAttention failed')

    def test08_Model_EmbeddingLstmGruGpu(self):
        '''Test of the model Embedding/LSTM/GRU'''
        print('            ------------------ >     Test of the model Embedding/LSTM/GRU     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_gru_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_gru_gpu.ModelEmbeddingLstmGruGpu(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                               batch_size=16, epochs=20, patience=20,
                                                                               max_sequence_length=60, max_words=100000,
                                                                               embedding_name="custom.300.pkl",
                                                                               multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmGruGpu failed')

    def test09_Model_EmbeddingCnn(self):
        '''Test of the model Embedding/CNN'''
        print('            ------------------ >     Test of the model Embedding/CNN     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_cnn_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_cnn.ModelEmbeddingCnn(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                               batch_size=16, epochs=20, patience=20,
                                                               max_sequence_length=60, max_words=100000,
                                                               embedding_name="custom.300.pkl",
                                                               multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingCnn failed')

    def test10_Model_Keras_continue_training(self):
        '''Test continuing a training for a keras model'''
        print("            ------------------ >     Test continuing a training for a keras model     /   Mono-class & multi-labels")

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=16, epochs=40, patience=40,
                                                                                    max_sequence_length=60, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=True, model_name=model_name, model_dir=model_dir)
            # Run a first training
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            self.assertEqual(model_dir, test_model.model_dir)

            # Retrieve model & run a second training
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_Keras_continue_training failed')

    def test11_Model_TfidfSgdc(self):
        '''Test of the model TF-IDF/SGDClassifier'''
        print('            ------------------ >     Test of the model TF-IDF/SGDClassifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_sgdc_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         sgdc_params={'loss': 'hinge', 'max_iter': 1000},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_sgdc_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           sgdc_params={'loss': 'log', 'max_iter': 1000},
                                                           multi_label=True, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_sgdc_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                sgdc_params={'loss': 'log', 'max_iter': 1000},
            #                                                multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #           filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSgdc failed')

    def test12_Model_PytorchTransformer(self):
        '''Test of the model Pytorch Transformer'''
        transformers_path = utils.get_transformers_path()
        transformer_path = os.path.join(transformers_path, 'flaubert', 'flaubert_small_cased')
        if not os.path.exists(transformer_path):
            print("WARNING : Can't test the Pytorch Transformer model -> can't find transformer")
            print("How to use : download flaubert_small_cased in the folder of the module to test")
            print("We ignore this test.")
            return None
        print('            ------------------ >     Test of the model Pytorch Transformer     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'pytorch_transformer_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_pytorch_transformers.ModelPyTorchTransformers(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                             batch_size=16, epochs=20, patience=20,
                                                                             max_sequence_length=60,
                                                                             transformer_name='flaubert/flaubert_small_cased',
                                                                             tokenizer_special_tokens=tuple(),
                                                                             padding="max_length", truncation=True,
                                                                             multi_label=True, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_PytorchTransformer failed')

    def test13_Model_EmbeddingLstmStructuredAttention(self):
        '''Test of the model Embedding/LSTM/Attention + explainable'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention + explainable     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                                         batch_size=16, epochs=40, patience=40,
                                                                                                         max_sequence_length=60, max_words=100000,
                                                                                                         embedding_name="custom.300.pkl",
                                                                                                         multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmStructuredAttention failed')

def test_model_multi_class_mono_label(test_class, test_model):
    '''Generic fonction to test a given model for multi-classes/mono-label'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Try some functions
    index_none = test_model.list_classes.index('none')
    index_a = test_model.list_classes.index('a')
    index_b = test_model.list_classes.index('b')
    index_both = test_model.list_classes.index('both')
    pred_none = [0, 0, 0, 0]
    pred_none[index_none] = 1
    pred_a = [0, 0, 0, 0]
    pred_a[index_a] = 1
    pred_b = [0, 0, 0, 0]
    pred_b[index_b] = 1
    pred_both = [0, 0, 0, 0]
    pred_both[index_both] = 1
    # predict
    preds = test_model.predict(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual(list(preds), ['none', 'a', 'b', 'both'])
    # predict_proba
    probas = test_model.predict_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual(round(probas.sum(), 3), 4.) # We round for deep learning models
    test_class.assertGreater(probas[0][index_none], 1/4)
    test_class.assertLess(probas[0][index_a], probas[0][index_none])
    test_class.assertLess(probas[0][index_b], probas[0][index_none])
    test_class.assertLess(probas[0][index_both], probas[0][index_none])
    test_class.assertLess(probas[1][index_none], probas[1][index_a])
    test_class.assertGreater(probas[1][index_a], 1/4)
    test_class.assertLess(probas[1][index_b], probas[1][index_a])
    test_class.assertLess(probas[1][index_both], probas[1][index_a])
    test_class.assertLess(probas[2][index_none], probas[2][index_b])
    test_class.assertLess(probas[2][index_a], probas[2][index_b])
    test_class.assertGreater(probas[2][index_b], 1/4)
    test_class.assertLess(probas[2][index_both], probas[2][index_b])
    test_class.assertLess(probas[3][index_none], probas[3][index_both])
    test_class.assertLess(probas[3][index_a], probas[3][index_both])
    test_class.assertLess(probas[3][index_b], probas[3][index_both])
    test_class.assertGreater(probas[3][index_both], 1/4)
    # predict w/ return_proba=True
    probas2 = test_model.predict(['cdi à temps complet', 'vous disposez du permis',
                                  'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'],
                                  return_proba=True)
    test_class.assertEqual(round(probas2.sum(), 3), 4.) # We round for deep learning models
    test_class.assertGreater(probas2[0][index_none], 1/4)
    test_class.assertLess(probas2[0][index_a], probas2[0][index_none])
    test_class.assertLess(probas2[0][index_b], probas2[0][index_none])
    test_class.assertLess(probas2[0][index_both], probas2[0][index_none])
    test_class.assertLess(probas2[1][index_none], probas2[1][index_a])
    test_class.assertGreater(probas2[1][index_a], 1/4)
    test_class.assertLess(probas2[1][index_b], probas2[1][index_a])
    test_class.assertLess(probas2[1][index_both], probas2[1][index_a])
    test_class.assertLess(probas2[2][index_none], probas2[2][index_b])
    test_class.assertLess(probas2[2][index_a], probas2[2][index_b])
    test_class.assertGreater(probas2[2][index_b], 1/4)
    test_class.assertLess(probas2[2][index_both], probas2[2][index_b])
    test_class.assertLess(probas2[3][index_none], probas2[3][index_both])
    test_class.assertLess(probas2[3][index_a], probas2[3][index_both])
    test_class.assertLess(probas2[3][index_b], probas2[3][index_both])
    test_class.assertGreater(probas2[3][index_both], 1/4)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual(list(pred_proba[0]), ['none', 'a', 'b', 'both'])
    test_class.assertEqual(round(pred_proba[1].sum(), 3), 4.) # We round for deep learning models
    test_class.assertGreater(pred_proba[1][0][index_none], 1/4)
    test_class.assertLess(pred_proba[1][0][index_a], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][0][index_b], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][0][index_both], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][1][index_none], pred_proba[1][1][index_a])
    test_class.assertGreater(pred_proba[1][1][index_a], 1/4)
    test_class.assertLess(pred_proba[1][1][index_b], pred_proba[1][1][index_a])
    test_class.assertLess(pred_proba[1][1][index_both], pred_proba[1][1][index_a])
    test_class.assertLess(pred_proba[1][2][index_none], pred_proba[1][2][index_b])
    test_class.assertLess(pred_proba[1][2][index_a], pred_proba[1][2][index_b])
    test_class.assertGreater(pred_proba[1][2][index_b], 1/4)
    test_class.assertLess(pred_proba[1][2][index_both], pred_proba[1][2][index_b])
    test_class.assertLess(pred_proba[1][3][index_none], pred_proba[1][3][index_both])
    test_class.assertLess(pred_proba[1][3][index_a], pred_proba[1][3][index_both])
    test_class.assertLess(pred_proba[1][3][index_b], pred_proba[1][3][index_both])
    test_class.assertGreater(pred_proba[1][3][index_both], 1/4)
    # get_predict_position
    # position start at 1
    predict_pos = test_model.get_predict_position(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire', 'titi'], ['none', 'a', 'a', 'both', 'toto'])
    test_class.assertEqual(list(predict_pos[[0, 1, 3, 4]]), [1, 1, 1, -1])
    test_class.assertGreater(predict_pos[2], 1)
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['none', 'a', 'b', 'both'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5) # Only 4 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=4)
    test_class.assertEqual([_[0] for _ in top_n], ['none', 'a', 'b', 'both'])
    test_class.assertEqual(sorted(top_n[0]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[1]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[2]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[3]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual([_[0] for _ in top_n_proba], [probas[0][index_none], probas[1][index_a], probas[2][index_b], probas[3][index_both]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['none', 'a', 'b', 'both'])


class Case4_MultiClassMonoLabel(unittest.TestCase):
    '''Class to test the multi-classes / mono-label case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the multi-classes / mono-label case")

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f multi_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py -f multi_class_mono_label_train.csv multi_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_TfidfSvm(self):
        '''Test of the model TF-IDF/SVM'''
        print('            ------------------ >     Test of the model TF-IDF/SVM     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_svm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       svc_params={'C': 1.0, 'max_iter': 10000},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_svm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         svc_params={'C': 1.0, 'max_iter': 10000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_svm_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              svc_params={'C': 1.0, 'max_iter': 10000},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSvm failed')

    def test03_Model_TfidfGbt(self):
        '''Test of the model TF-IDF/GBT'''
        print('            ------------------ >     Test of the model TF-IDF/GBT     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_gbt_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_gbt_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_gbt_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfGbt failed')

    def test04_Model_TfidfLgbm(self):
        '''Test of the model TF-IDF/LGBM'''
        print('            ------------------ >     Test of the model TF-IDF/LGBM     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_lgbm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_lgbm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model_2 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_lgbm_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # # We overfit on purpose !
            # test_model_3 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
            #                                                multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #          filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfLgbm failed')

    def test05_Model_TfidfDense(self):
        '''Test of the model TF-IDF/Dense'''
        print('            ------------------ >     Test of the model TF-IDF/Dense     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'tfidf_dense_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_dense.ModelTfidfDense(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           batch_size=16, epochs=20, patience=20,
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_TfidfDense failed')

    def test06_Model_EmbeddingLstm(self):
        '''Test of the model Embedding/LSTM'''
        print('            ------------------ >     Test of the model Embedding/LSTM     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                   filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstm failed')

    def test07_Model_EmbeddingLstmAttention(self):
        '''Test of the model Embedding/LSTM/Attention'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=64, epochs=40, patience=20,
                                                                                    max_sequence_length=10, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmAttention failed')

    def test08_Model_EmbeddingLstmGruGpu(self):
        '''Test of the model Embedding/LSTM/GRU'''
        print('            ------------------ >     Test of the model Embedding/LSTM/GRU     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_gru_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_gru_gpu.ModelEmbeddingLstmGruGpu(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                               batch_size=16, epochs=20, patience=20,
                                                                               max_sequence_length=60, max_words=100000,
                                                                               embedding_name="custom.300.pkl",
                                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmGruGpu failed')

    def test09_Model_EmbeddingCnn(self):
        '''Test of the model Embedding/CNN'''
        print('            ------------------ >     Test of the model Embedding/CNN     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_cnn_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_cnn.ModelEmbeddingCnn(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                               batch_size=16, epochs=20, patience=20,
                                                               max_sequence_length=60, max_words=100000,
                                                               embedding_name="custom.300.pkl",
                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingCnn failed')

    def test10_Model_Keras_continue_training(self):
        '''Test continuing a training for a keras model'''
        print("            ------------------ >     Test continuing a training for a keras model     /   Multi-class & Mono-label")

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Run a first training
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertEqual(model_dir, test_model.model_dir)

            # Retrieve model & run a second training
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_Keras_continue_training failed')

    def test11_Model_TfidfSgdc(self):
        '''Test of the model TF-IDF/SGDClassifier'''
        print('            ------------------ >     Test of the model TF-IDF/SGDClassifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_sgdc_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         sgdc_params={'loss': 'hinge', 'max_iter': 1000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_sgdc_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           sgdc_params={'loss': 'log', 'max_iter': 1000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_sgdc_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                sgdc_params={'loss': 'log', 'max_iter': 1000},
            #                                                multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSgdc failed')

    def test12_Model_PytorchTransformer(self):
        '''Test of the model Pytorch Transformer'''
        transformers_path = utils.get_transformers_path()
        transformer_path = os.path.join(transformers_path, 'flaubert', 'flaubert_small_cased')
        if not os.path.exists(transformer_path):
            print("WARNING : Can't test the Pytorch Transformer model -> can't find transformer")
            print("How to use : download flaubert_small_cased in the folder of the module to test")
            print("We ignore this test.")
            return None
        print('            ------------------ >     Test of the model Pytorch Transformer     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'pytorch_transformer_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_pytorch_transformers.ModelPyTorchTransformers(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                             batch_size=16, epochs=20, patience=20,
                                                                             max_sequence_length=60,
                                                                             transformer_name='flaubert/flaubert_small_cased',
                                                                             tokenizer_special_tokens=tuple(),
                                                                             padding="max_length", truncation=True,
                                                                             multi_label=False, model_name=model_name, model_dir=model_dir)

            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_PytorchTransformer failed')

    def test13_Model_EmbeddingLstmStructuredAttention(self):
        '''Test of the model Embedding/LSTM/Attention + explainable'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention + explainable     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                                         batch_size=64, epochs=40, patience=20,
                                                                                                         max_sequence_length=10, max_words=100000,
                                                                                                         embedding_name="custom.300.pkl",
                                                                                                         multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmStructuredAttention failed')


if __name__ == '__main__':
    # Change directory to script directory parent
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    parentname = str(Path(dname).parent)
    os.chdir(parentname)
    # Manage venv
    full_path_lib = os.path.abspath(os.path.join(os.getcwd(), 'test_template_nlp'))
    if os.name == 'nt':
        is_windows = True
        # Windows: activate the virtual environment & continue with the other processes
        activate_venv = f"cd {full_path_lib}/venv_test_template_nlp/Scripts & activate & "
    else:
        is_windows = False
        # UNIX : We can't use "source" so we directly call python/pip from the bin of the virtual environment
        activate_venv = f"{full_path_lib}/venv_test_template_nlp/bin/"
    # Start tests
    unittest.main()
