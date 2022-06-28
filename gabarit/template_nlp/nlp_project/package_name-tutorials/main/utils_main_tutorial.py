#!/usr/bin/env python3

## Useful file for the NLP tutorial
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

import os
import sys
import logging
import numpy as np
import pandas as pd

from {{package_name}} import utils


## Manage logger
# Get logger (def level: INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Get console handler
# On log tout ce qui est possible ici (i.e >= level du logger)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
# Manage formatter
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
# Add handler to the logger
logger.addHandler(ch)


def change_dir():
    '''Gets us to the root of this script'''
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


def test_exercice_1():
    '''Tests the first exercise'''
    change_dir()
    data_path = utils.get_data_path()
    train_path = os.path.join(data_path, 'dataset_jvc_train.csv')
    valid_path = os.path.join(data_path, 'dataset_jvc_valid.csv')
    test_path = os.path.join(data_path, 'dataset_jvc_test.csv')
    if not os.path.exists(train_path):
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        logger.error(f"Can't find file {train_path}")
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
    elif not os.path.exists(valid_path):
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        logger.error(f"Can't find file {valid_path}")
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
    elif not os.path.exists(test_path):
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        logger.error(f"Can't find file {test_path}")
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
    # Check the seed
    else:
        df_train = pd.read_csv(train_path, sep=';', encoding='utf-8')
        expected_game = 'Trials Evolution : Gold Edition sur PC'
        game = df_train['game'].iloc[5]
        if expected_game != game:
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
            logger.error("The files have been found but with wrong content.")
            logger.error("Did you put the random seed equal to 42?")
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        else:
            print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
            print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
    return None


def get_exercice_1_solution():
    '''Gets the solution for exercise 1'''
    change_dir()
    data_path = utils.get_data_path()
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.split(dir_path)[0]
    scripts_path = os.path.join(dir_path, '{{package_name}}-scripts', 'utils')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
    print('')
    print('Activate your virtual environment')
    print('')
    print(f"cd {scripts_path}")
    print('')
    print(f"python 0_split_train_valid_test.py -f dataset_jvc.csv --split_type random --seed 42")
    print('')
    print(f"If you already generated some files and need to overwrite them, you need to add the --overwrite argument:")
    print(f"python 0_split_train_valid_test.py -f dataset_jvc.csv --split_type random --seed 42 --overwrite")
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")


def test_exercice_2():
    '''Tests the second exercise'''
    change_dir()
    project_path = os.path.join(os.getcwd(), 'my_awesome_folder')
    data_path = utils.get_data_path()
    train_sample_path = os.path.join(data_path, 'dataset_jvc_train_10_samples.csv')
    if not os.path.exists(train_sample_path):
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        logger.error(f"Can't find file {train_sample_path}")
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
    # Check that there are 10 lines
    else:
        df_train_sample = pd.read_csv(train_sample_path, sep=';', encoding='utf-8')
        expected_n_lines = 10
        n_lines = df_train_sample.shape[0]
        if expected_n_lines != n_lines:
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
            logger.error(f"The file was found but with {n_lines} lines instead of {expected_n_lines} !")
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        else:
            print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
            print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
    return None


def get_exercice_2_solution():
    '''Gets the solution for exercise 2'''
    change_dir()
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.split(dir_path)[0]
    scripts_path = os.path.join(dir_path, '{{package_name}}-scripts', 'utils')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
    print('')
    print('Activate your virtual environment')
    print('')
    print(f"cd {scripts_path}")
    print('')
    print(f"python 0_create_samples.py -f dataset_jvc_train.csv -n 10")
    print('')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")


def test_exercice_3():
    '''Tests exercise 3'''
    change_dir()
    data_path = utils.get_data_path()
    train_preprocess_P1_path = os.path.join(data_path, 'dataset_jvc_train_preprocess_P1.csv')
    if not os.path.exists(train_preprocess_P1_path):
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        logger.error(f"Can't find file {train_preprocess_P1_path}")
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
    # Check one line
    else:
        df_train_preprocess_P1 = pd.read_csv(train_preprocess_P1_path, sep=';', encoding='utf-8', skiprows=1)
        expected_preprocess = 'trials evolution gold edition est un jeu mêlant courses et plates formes sur pc le joueur y pilote une moto dans des niveaux semés d embûches et d obstacles pour les franchir il doit jouer en permanence sur l équilibre de sa monture un éditeur de niveaux permet de créer des pistes et de les partager en ligne cette version pc comprend en outre le jeu trials hd'
        preprocess = df_train_preprocess_P1['preprocessed_text'].iloc[5]
        if expected_preprocess != preprocess:
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
            logger.error("The preprocessed file exists but with the wrong data.")
            logger.error("Did you apply preprocess_P1 to dataset_jvc_train.csv ?")
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        else:
            print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
            print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
    return None


def get_exercice_3_solution():
    '''Gets the solution for exercise 3'''
    change_dir()
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.split(dir_path)[0]
    scripts_path = os.path.join(dir_path, '{{package_name}}-scripts')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
    print('')
    print('Activate your virtual environment')
    print('')
    print(f"cd {scripts_path}")
    print('')
    print(f"python 1_preprocess_data.py -f dataset_jvc_train.csv --input_col description")
    print('')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")


def test_exercice_4():
    '''Tests exercise 4'''
    change_dir()
    data_path = utils.get_data_path()
    train_preprocess_P2_path = os.path.join(data_path, 'dataset_jvc_train_preprocess_P2.csv')
    valid_preprocess_P2_path = os.path.join(data_path, 'dataset_jvc_valid_preprocess_P2.csv')
    if not os.path.exists(train_preprocess_P2_path):
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        logger.error(f"Can't find file {train_preprocess_P2_path}")
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
    elif not os.path.exists(valid_preprocess_P2_path):
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        logger.error(f"Can't find file {valid_preprocess_P2_path}")
        print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
    # Check one line on the train
    else:
        df_train_preprocess_P2 = pd.read_csv(train_preprocess_P2_path, sep=';', encoding='utf-8', skiprows=1)
        expected_preprocess = 'trials evolution gold edition jeu mêlant courses plates formes pc joueur pilote moto niveaux semés embûches obstacles franchir jouer permanence équilibre monture éditeur niveaux créer pistes partager ligne version pc comprend jeu trials hd'
        preprocess = df_train_preprocess_P2['preprocessed_text'].iloc[5]
        if expected_preprocess != preprocess:
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
            logger.error("The preprocessed file exists but with the wrong data.")
            logger.error("Did you apply preprocess_P2 to dataset_jvc_train.csv ?")
            print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
        # Check one line on the valid
        else:
            df_valid_preprocess_P2 = pd.read_csv(valid_preprocess_P2_path, sep=';', encoding='utf-8', skiprows=1)
            expected_preprocess = 'jeu plates formes poétique xbox one ori and the blind forest hommage gameplay jeux aventure rpg metroidvania joueur explorer niveaux trouver pouvoirs pouvoir accéder nouvelles zones'
            preprocess = df_valid_preprocess_P2['preprocessed_text'].iloc[17]
            if expected_preprocess != preprocess:
                print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
                logger.error("The preprocessed file exists but with the wrong data.")
                logger.error("Did you apply preprocess_P2 to dataset_jvc_valid.csv ?")
                print(f"\033[1m\033[91m🤬 TEST FAILED 🤬\033[0m")
            else:
                print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
                print(f"\033[1m\033[92m😀 TEST SUCCEEDED 😀\033[0m")
    return None


def get_exercice_4_solution():
    '''Gets the solution for exercise 4'''
    change_dir()
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.split(dir_path)[0]
    scripts_path = os.path.join(dir_path, '{{package_name}}-scripts')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
    print('')
    print('In the file preprocess.py :')
    print('')
    print('   1. Add the function preprocess_sentence_P2 :')
    print('''
@wnf_utils.data_agnostic
@wnf_utils.regroup_data_series
def preprocess_sentence_P2(docs):
    if type(docs) != pd.Series: raise TypeError("L'objet docs doit être du type pd.Series.")
    pipeline = ['remove_non_string', 'get_true_spaces', 'remove_punct', 'to_lower',
                'remove_stopwords', 'trim_string', 'remove_leading_and_ending_spaces']
    return api.preprocess_pipeline(docs, pipeline=pipeline, chunksize=100000)
    ''')
    print('')
    print('   2. Add an entry in the dictionary preprocessors_dict :')
    print('''
            preprocessors_dict = {
                'no_preprocess': lambda x: x,
                'preprocess_P1': preprocess_sentence_P1, # Example of a preprocessing
                'preprocess_P2': preprocess_sentence_P2, # New preprocessing
            }
    ''')
    print('')
    print('Activate your virtual environment')
    print('')
    print(f"cd {scripts_path}")
    print('')
    print(f"python 1_preprocess_data.py -f dataset_jvc_train.csv dataset_jvc_valid.csv --input_col description --preprocessing preprocess_P2")
    print('')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")


def get_exercice_5_solution():
    '''Gets the solution for exercise 5'''
    change_dir()
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.split(dir_path)[0]
    scripts_path = os.path.join(dir_path, '{{package_name}}-scripts')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
    print('')
    print(f"Activate your virtual environment")
    print('')
    print(f"cd {scripts_path}")
    print('')
    print(f"python 2_training.py --filename dataset_jvc_train_preprocess_P2.csv --x_col preprocessed_text --y_col RPG --filename_valid dataset_jvc_valid_preprocess_P2.csv")
    print('')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")


def get_exercice_6_solution():
    '''Gets the solution for exercise 6'''
    change_dir()
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.split(dir_path)[0]
    scripts_path = os.path.join(dir_path, '{{package_name}}-scripts')
    scripts_path_utils = os.path.join(scripts_path, 'utils')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
    print('')
    print('# Comment the line with ModelTfidfSvm in 2_training.py.')
    print('')
    print('# Uncomment the line with ModelEmbeddingLstm in 2_training.py.')
    print('')
    print(f"Activate your virtual environment")
    print('')
    print(f"cd {scripts_path_utils}")
    print('')
    print(f"python 0_get_embedding_dict.py -f cc.fr.300.vec")
    print('')
    print(f"cd {scripts_path}")
    print('')
    print(f'python 2_training.py --filename dataset_jvc_train.csv --x_col description --y_col "Action" "Aventure" "RPG" "Plate-Forme" "FPS" "Course" "Strategie" "Sport" "Reflexion" "Combat" --filename_valid dataset_jvc_valid.csv')
    print('')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")


def get_exercice_7_solution():
    '''Gets the solution for exercise 8'''
    change_dir()
    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dir_path = os.path.split(dir_path)[0]
    scripts_path = os.path.join(dir_path, '{{package_name}}-scripts')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
    print('')
    print(f"cd {scripts_path}")
    print('')
    print(f"Activate your virtual environment")
    print('')
    print('python 3_predict.py --filename dataset_jvc_test.csv --x_col description --y_col "Action" "Aventure" "RPG" "Plate-Forme" "FPS" "Course" "Strategie" "Sport" "Reflexion" "Combat" --model_dir model_embedding_lstm_{YYYY_MM_DD-hh_mm_ss}')
    print('')
    print(f"\033[1m\033[94m💡 SOLUTION 💡\033[0m")
