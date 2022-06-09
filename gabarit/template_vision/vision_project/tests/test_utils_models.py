#!/usr/bin/env python3
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

# Utils libs
import os
import shutil
import dill as pickle
import numpy as np
import pandas as pd
from PIL import Image
from {{package_name}} import utils
from {{package_name}}.models_training import utils_models
from {{package_name}}.preprocessing import preprocess
from {{package_name}}.models_training.classifiers.model_cnn_classifier import ModelCnnClassifier

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class UtilsModelsTests(unittest.TestCase):
    '''Main class to test all functions in utils_models.py'''

    # We avoid tqdm prints
    pd.Series.progress_apply = pd.Series.apply


    def setUp(self):
        '''SetUp fonction'''
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_normal_split(self):
        '''Test of the method {{package_name}}.models_training.utils_models.normal_split'''
        # Valids to test
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']})
        test_size = 0.2
        train, test = utils_models.normal_split(input_test, test_size=test_size)
        self.assertEqual(train.shape[0], input_test.shape[0] * (1 - test_size))
        self.assertEqual(test.shape[0], input_test.shape[0] * test_size)

        # Check inputs
        with self.assertRaises(ValueError):
            utils_models.normal_split(input_test, test_size=1.2)
        with self.assertRaises(ValueError):
            utils_models.normal_split(input_test, test_size=-0.2)


    def test02_stratified_split(self):
        '''Test of the method {{package_name}}.models_training.utils_models.stratified_split'''
        # Valids to test
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                                   'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})
        test_size = 0.5
        col = 'col2'
        train, test = utils_models.stratified_split(input_test, col, test_size=test_size)
        self.assertEqual(train.shape[0], (input_test.shape[0] - 1) * (1 - test_size))
        self.assertEqual(test.shape[0], (input_test.shape[0] - 1) * test_size)
        self.assertEqual(train[train[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0] * (1 - test_size))
        self.assertEqual(test[test[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0] * test_size)
        self.assertEqual(train[train[col] == 2].shape[0], 0)
        self.assertEqual(test[test[col] == 2].shape[0], 0)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            utils_models.stratified_split(input_test, col, test_size=1.2)
        with self.assertRaises(ValueError):
            utils_models.stratified_split(input_test, col, test_size=-0.2)


    def test03_remove_small_classes(self):
        '''Test of the method {{package_name}}.models_training.utils_models.remove_small_classes'''
        # Valids to test
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                                   'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})
        test_size = 0.2
        col = 'col2'

        result_df = utils_models.remove_small_classes(input_test, col, min_rows=2)
        self.assertEqual(result_df[result_df[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0])
        self.assertEqual(result_df[result_df[col] == 1].shape[0], input_test[input_test[col] == 1].shape[0])
        self.assertEqual(result_df[result_df[col] == 2].shape[0], 0)

        result_df = utils_models.remove_small_classes(input_test, col, min_rows=5)
        self.assertEqual(result_df[result_df[col] == 0].shape[0], 0)
        self.assertEqual(result_df[result_df[col] == 1].shape[0], input_test[input_test[col] == 1].shape[0])
        self.assertEqual(result_df[result_df[col] == 2].shape[0], 0)

        # Check the input(s) type(s)
        with self.assertRaises(ValueError):
            utils_models.remove_small_classes(input_test, col, min_rows=0)


    def test04_display_train_test_shape(self):
        '''Test of the method {{package_name}}.models_training.utils_models.display_train_test_shape'''
        # Valids to test
        df = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                           'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})

        # Nominal case
        utils_models.display_train_test_shape(df, df)
        utils_models.display_train_test_shape(df, df, df_shape=10)


    def test05_load_model(self):
        '''Test of the method {{package_name}}.models_training.utils_models.load_model'''

        model_dir = os.path.join(utils.get_models_path(), 'test_model_123456789')
        remove_dir(model_dir)

        # Data for training
        # Set vars
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        model_name = 'test_model_name'
        batch_size = 8
        epochs = 3
        patience = 5


        ####################################################

        # Tests sur un model cnn
        model = ModelCnnClassifier(model_dir=model_dir, model_name=model_name, batch_size=batch_size,
                                   epochs=epochs, patience=patience)
        model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model_123456789')
        # We perform some tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['batch_size'], batch_size)
        self.assertEqual(new_config['epochs'], epochs)
        self.assertEqual(new_config['patience'], patience)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(df_train_mono)), list(model.predict(df_train_mono)))

        # Same thing but with a path
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # We perform some tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['batch_size'], batch_size)
        self.assertEqual(new_config['epochs'], epochs)
        self.assertEqual(new_config['patience'], patience)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(df_train_mono)), list(model.predict(df_train_mono)))
        remove_dir(model_dir)

        ####################################################

        # Check errors
        with self.assertRaises(FileNotFoundError):
            utils_models.load_model(model_dir='tototo')
        with self.assertRaises(FileNotFoundError):
            utils_models.load_model(model_dir='./tototo', is_path=True)


    def test06_predict(self):
        '''Test of the method {{package_name}}.models_training.utils_models.predict'''

        model_dir = os.path.join(utils.get_models_path(), 'test_model_123456789')
        remove_dir(model_dir)

        # Data for training
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        file_paths = [os.path.join(data_path, _) for _ in filenames]
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        model_name = 'test_model_123456789'
        batch_size = 8
        epochs = 3
        patience = 5
        np_images_rgba = np.array([np.array(Image.open(_)) for _ in file_paths[1:]])  # Problème format première image
        np_images_rgb = np.array([np.array(Image.open(_).convert('RGB')) for _ in file_paths[1:]])  # Problème format première image


        ################
        # Classification - mono-class

        # Creation fake model
        model = ModelCnnClassifier(model_dir=model_dir, model_name=model_name, batch_size=batch_size,
                                   epochs=epochs, patience=patience, color_mode='rgb')
        model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()
        y_classes = list(model.inverse_transform(model.predict(df_train_mono)))
        probas = model.predict_proba(df_train_mono)

        # Nominal case
        model, model_conf = utils_models.load_model(model_dir=model_name)
        self.assertEqual(utils_models.predict(df_train_mono, model, model_conf), y_classes)  # DataFrame
        self.assertEqual(utils_models.predict(file_paths, model, model_conf), y_classes)  # Liste fichiers
        self.assertEqual(utils_models.predict(file_paths[0], model, model_conf), y_classes[0])  # Chemin fichier
        self.assertEqual(utils_models.predict(np_images_rgb, model, model_conf), y_classes[1:]) # np.ndarray 'RGB'
        self.assertEqual(utils_models.predict(np_images_rgb[0], model, model_conf), y_classes[1]) # np.ndarray 'RGB' - 1 seule image
        np.testing.assert_almost_equal(utils_models.predict(df_train_mono, model, model_conf, return_proba=True), probas, 3)  # DataFrame
        np.testing.assert_almost_equal(utils_models.predict(file_paths, model, model_conf, return_proba=True), probas, 3)  # Liste fichiers
        np.testing.assert_almost_equal(utils_models.predict(file_paths[0], model, model_conf, return_proba=True), probas[0], 3)  # Chemin fichier
        np.testing.assert_almost_equal(utils_models.predict(np_images_rgb, model, model_conf, return_proba=True), probas[1:], 3) # np.ndarray 'RGB'
        np.testing.assert_almost_equal(utils_models.predict(np_images_rgb[0], model, model_conf, return_proba=True), probas[1], 3) # np.ndarray 'RGB' - 1 seule image
        remove_dir(model_dir)


        ################
        # Classification - multi-class - color mode rgba & depth = 4

        # Creation fake model
        model = ModelCnnClassifier(model_dir=model_dir, model_name=model_name, batch_size=batch_size,
                                   epochs=epochs, patience=patience, color_mode='rgba', depth=4)
        model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        model.save()
        y_classes = list(model.inverse_transform(model.predict(df_train_multi)))
        probas = model.predict_proba(df_train_multi)

        # Nominal case
        model, model_conf = utils_models.load_model(model_dir=model_name)
        self.assertEqual(utils_models.predict(df_train_multi, model, model_conf), y_classes)  # DataFrame
        self.assertEqual(utils_models.predict(file_paths, model, model_conf), y_classes)  # Liste fichiers
        self.assertEqual(utils_models.predict(file_paths[0], model, model_conf), y_classes[0])  # Chemin fichier
        self.assertEqual(utils_models.predict(np_images_rgba, model, model_conf), y_classes[1:]) # np.ndarray 'RGBA'
        self.assertEqual(utils_models.predict(np_images_rgba[0], model, model_conf), y_classes[1]) # np.ndarray 'RGB' - 1 seule image
        np.testing.assert_almost_equal(utils_models.predict(df_train_multi, model, model_conf, return_proba=True), probas, 3)  # DataFrame
        np.testing.assert_almost_equal(utils_models.predict(file_paths, model, model_conf, return_proba=True), probas, 3)  # Liste fichiers
        np.testing.assert_almost_equal(utils_models.predict(file_paths[0], model, model_conf, return_proba=True), probas[0], 3)  # Chemin fichier
        np.testing.assert_almost_equal(utils_models.predict(np_images_rgba, model, model_conf, return_proba=True), probas[1:], 3) # np.ndarray 'RGBA'
        np.testing.assert_almost_equal(utils_models.predict(np_images_rgba[0], model, model_conf, return_proba=True), probas[1], 3) # np.ndarray 'RGB' - 1 seule image
        remove_dir(model_dir)


        ################
        # Manage errors
        model = ModelCnnClassifier(model_dir=model_dir, model_name=model_name, batch_size=batch_size,
                                   epochs=epochs, patience=patience, color_mode='rgb')
        model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()
        with self.assertRaises(FileNotFoundError):
            utils_models.predict('bad_file_path', model, model_conf)
        with self.assertRaises(FileNotFoundError):
            utils_models.predict(['bad_file_path'], model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict(np.array([[[[10, 2], [5, 6]], [[1, 0], [4, 6]]]]), model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict(np.array([[[10, 2], [5, 6]], [[1, 0], [4, 6]]]), model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict(np.array([[0, 1]]), model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict(pd.DataFrame({'toto': ['titi', 'tata']}), model, model_conf)
        with self.assertRaises(FileNotFoundError):
            utils_models.predict(pd.DataFrame({'file_path': ['bad_file_path']}), model, model_conf)
        remove_dir(model_dir)


    def test07_predict_with_proba(self):
        '''Test of the method {{package_name}}.models_training.utils_models.predict_with_proba'''
        model_dir = os.path.join(utils.get_models_path(), 'test_model_123456789')
        remove_dir(model_dir)

        # Data for training
        data_path = os.path.join(os.getcwd(), 'test_data', '{{package_name}}-data', 'test_data_1')
        filenames = ['Birman_1.jpg', 'Birman_2.jpg', 'Birman_4.jpg', 'Bombay_1.png', 'Bombay_10.jpg',
                     'Bombay_3.jpg', 'shiba_inu_15.jpg', 'shiba_inu_30.jpg', 'shiba_inu_31.jpg',
                     'shiba_inu_34.jpg', 'subfolder/Birman_36.jpg', 'subfolder/Bombay_19.jpg']
        file_paths = [os.path.join(data_path, _) for _ in filenames]
        df_train_mono = pd.DataFrame({
            'file_class': ['cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'cat'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        df_train_multi = pd.DataFrame({
            'file_class': ['birman', 'birman', 'birman', 'bombay', 'bombay', 'bombay', 'shiba', 'shiba', 'shiba', 'shiba', 'birman', 'bombay'],
            'file_path': [os.path.join(data_path, _) for _ in filenames],
        })
        model_name = 'test_model_123456789'
        batch_size = 8
        epochs = 3
        patience = 5
        np_images_rgba = np.array([np.array(Image.open(_)) for _ in file_paths[1:]])  # Problème format première image
        np_images_rgb = np.array([np.array(Image.open(_).convert('RGB')) for _ in file_paths[1:]])  # Problème format première image


        ################
        # Classification - mono-class

        # Creation fake model
        model = ModelCnnClassifier(model_dir=model_dir, model_name=model_name, batch_size=batch_size,
                                   epochs=epochs, patience=patience, color_mode='rgb')
        model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()
        y_classes = list(model.inverse_transform(model.predict(df_train_mono)))
        probas = model.predict_proba(df_train_mono)
        max_probas = list(probas.max(axis=1))

        # Nominal case
        model, model_conf = utils_models.load_model(model_dir=model_name)
        self.assertEqual(utils_models.predict_with_proba(df_train_mono, model, model_conf)[0], y_classes)  # DataFrame
        self.assertEqual(utils_models.predict_with_proba(file_paths, model, model_conf)[0], y_classes)  # Liste fichiers
        self.assertEqual(utils_models.predict_with_proba(file_paths[0], model, model_conf)[0], y_classes[0])  # Chemin fichier
        self.assertEqual(utils_models.predict_with_proba(np_images_rgb, model, model_conf)[0], y_classes[1:]) # np.ndarray 'RGB'
        self.assertEqual(utils_models.predict_with_proba(np_images_rgb[0], model, model_conf)[0], y_classes[1]) # np.ndarray 'RGB' - 1 seule image
        np.testing.assert_almost_equal(utils_models.predict_with_proba(df_train_mono, model, model_conf)[1], max_probas, 3)  # DataFrame
        np.testing.assert_almost_equal(utils_models.predict_with_proba(file_paths, model, model_conf)[1], max_probas, 3)  # Liste fichiers
        np.testing.assert_almost_equal(utils_models.predict_with_proba(file_paths[0], model, model_conf)[1], max_probas[0], 3)  # Chemin fichier
        np.testing.assert_almost_equal(utils_models.predict_with_proba(np_images_rgb, model, model_conf)[1], max_probas[1:], 3) # np.ndarray 'RGB'
        np.testing.assert_almost_equal(utils_models.predict_with_proba(np_images_rgb[0], model, model_conf)[1], max_probas[1], 3) # np.ndarray 'RGB' - 1 seule image
        remove_dir(model_dir)


        ################
        # Classification - multi-class - color mode rgba & depth = 4

        # Creation fake model
        model = ModelCnnClassifier(model_dir=model_dir, model_name=model_name, batch_size=batch_size,
                                   epochs=epochs, patience=patience, color_mode='rgba', depth=4)
        model.fit(df_train_multi, df_valid=df_train_multi, with_shuffle=True)
        model.save()
        y_classes = list(model.inverse_transform(model.predict(df_train_multi)))
        probas = model.predict_proba(df_train_multi)
        max_probas = list(probas.max(axis=1))

        # Nominal case
        model, model_conf = utils_models.load_model(model_dir=model_name)
        self.assertEqual(utils_models.predict_with_proba(df_train_multi, model, model_conf)[0], y_classes)  # DataFrame
        self.assertEqual(utils_models.predict_with_proba(file_paths, model, model_conf)[0], y_classes)  # Liste fichiers
        self.assertEqual(utils_models.predict_with_proba(file_paths[0], model, model_conf)[0], y_classes[0])  # Chemin fichier
        self.assertEqual(utils_models.predict_with_proba(np_images_rgba, model, model_conf)[0], y_classes[1:]) # np.ndarray 'RGBA'
        self.assertEqual(utils_models.predict_with_proba(np_images_rgba[0], model, model_conf)[0], y_classes[1]) # np.ndarray 'RGB' - 1 seule image
        np.testing.assert_almost_equal(utils_models.predict_with_proba(df_train_multi, model, model_conf)[1], max_probas, 3)  # DataFrame
        np.testing.assert_almost_equal(utils_models.predict_with_proba(file_paths, model, model_conf)[1], max_probas, 3)  # Liste fichiers
        np.testing.assert_almost_equal(utils_models.predict_with_proba(file_paths[0], model, model_conf)[1], max_probas[0], 3)  # Chemin fichier
        np.testing.assert_almost_equal(utils_models.predict_with_proba(np_images_rgba, model, model_conf)[1], max_probas[1:], 3) # np.ndarray 'RGBA'
        np.testing.assert_almost_equal(utils_models.predict_with_proba(np_images_rgba[0], model, model_conf)[1], max_probas[1], 3) # np.ndarray 'RGB' - 1 seule image
        remove_dir(model_dir)


        ################
        # Manage errors
        model = ModelCnnClassifier(model_dir=model_dir, model_name=model_name, batch_size=batch_size,
                                   epochs=epochs, patience=patience, color_mode='rgb')
        model.fit(df_train_mono, df_valid=df_train_mono, with_shuffle=True)
        model.save()
        with self.assertRaises(FileNotFoundError):
            utils_models.predict_with_proba('bad_file_path', model, model_conf)
        with self.assertRaises(FileNotFoundError):
            utils_models.predict_with_proba(['bad_file_path'], model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict_with_proba(np.array([[[[10, 2], [5, 6]], [[1, 0], [4, 6]]]]), model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict_with_proba(np.array([[[10, 2], [5, 6]], [[1, 0], [4, 6]]]), model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict_with_proba(np.array([[0, 1]]), model, model_conf)
        with self.assertRaises(ValueError):
            utils_models.predict_with_proba(pd.DataFrame({'toto': ['titi', 'tata']}), model, model_conf)
        with self.assertRaises(FileNotFoundError):
            utils_models.predict_with_proba(pd.DataFrame({'file_path': ['bad_file_path']}), model, model_conf)
        remove_dir(model_dir)


# Perform tests
if __name__ == '__main__':
    # Start tests
    unittest.main()
