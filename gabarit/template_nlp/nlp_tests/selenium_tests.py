#!/usr/bin/env python3
# Simple selenium tests to check that a demonstrator is running and working
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


import sys
import time
import argparse

# Libs unittest
import unittest
from unittest.mock import Mock
from unittest.mock import patch

# Libs selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
except ImportError as e:
    print("Selenium needs to be installed to run these tests")
    print("Tested with selenium==4.2.0")
    sys.exit("Can't import selenium")


class DemonstratorTests(unittest.TestCase):
    '''Main class to check that a demonstrator is running and working

    /!\ The demonstrator must be started BEFORE running this script /!\
    '''
    demonstrator_url = None
    trained_models = False
    driver = None

    def setUp(self):
        '''Setup fonction -> start a driver'''
        self.driver = webdriver.Chrome()
        self.driver.get(self.demonstrator_url)
        time.sleep(10)  # Wait 10s

    def tearDown(self):
        '''Cleaning fonction -> close the driver'''
        self.driver.close()
        self.driver = None

    def test01_demonstrator_up(self):
        '''Checks that the demonstrator is UP and running'''
        self.assertTrue('streamlit' in self.driver.title.lower())  # TODO: this might change in the future ? Only check if not empty ?

    def test02_sidebar_exists(self):
        '''Checks that the sidebar exists'''
        sidebars = self.driver.find_elements(By.XPATH, "//*[@data-testid='stSidebar']")
        self.assertTrue(len(sidebars) >= 1)

    def test03_sidebar_trained_models(self):
        '''Checks selectable options if models have been trained'''
        if not self.trained_models:
            unittest.SkipTest('No model have been trained.')
        else:
            sidebar = self.driver.find_element(By.XPATH, "//*[@data-testid='stSidebar']")
            sidebar_selectbox = sidebar.find_element(By.XPATH, ".//*[@class='row-widget stSelectbox']")
            text = sidebar_selectbox.find_element(By.XPATH, "./div/div/div/div[@aria-selected='true']").text
            self.assertFalse(text.startswith('No options to select'))

    def test04_title_exists(self):
        '''Checks that the title exists'''
        titles = self.driver.find_elements(By.XPATH, "//*/h1")
        titles_id = [title.get_property('id') for title in titles]
        main_titles = [title_id for title_id in titles_id if title_id.startswith('demonstrator')]
        self.assertTrue(len(main_titles) >= 1)

    def test05_no_exceptions(self):
        '''Checks that there is no streamlit exceptions'''
        exceptions = self.driver.find_elements(By.XPATH, "//*[@class='stException']")
        self.assertTrue(len(exceptions) == 0)


if __name__ == '__main__':
    # Retrieve params
    # Based on https://stackoverflow.com/questions/1029891/python-unittest-is-there-a-way-to-pass-command-line-options-to-the-app
    # and https://stackoverflow.com/questions/11380413/python-unittest-passing-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', default='http://localhost:8501', help="Demonstrator's URL to be tested")
    parser.add_argument('--with_trained_models', dest='trained_models', action='store_true', help="If models have been trained and must be present in Streamlit.")
    parser.set_defaults(trained_models=False)
    parser.add_argument('unittest_args', nargs='*', help="Optional unitest args")
    args = parser.parse_args()
    DemonstratorTests.demonstrator_url = args.url
    DemonstratorTests.trained_models = args.trained_models
    sys.argv[1:] = args.unittest_args

    # Start tests
    unittest.main()
