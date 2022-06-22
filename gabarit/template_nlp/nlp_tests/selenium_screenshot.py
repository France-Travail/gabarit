#!/usr/bin/env python3
# Use selenium to take a screenshot of an application
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


import argparse

# Libs selenium
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
except ImportError as e:
    print("Selenium needs to be installed to run these tests")
    print("Tested with selenium==4.2.0")
    sys.exit("Can't import selenium")


def main(output_path, url: str = 'http://localhost:8501'):
    ''''''
    driver = webdriver.Chrome()
    driver.get(url)
    driver.implicitly_wait(10)  # Wait 10 seconds for the page to load
    driver.get_screenshot_as_file(output_path)   # Take a screenshot & save it
    driver.close()


if __name__ == '__main__':
    # Retrieve params
    # Based on https://stackoverflow.com/questions/1029891/python-unittest-is-there-a-way-to-pass-command-line-options-to-the-app
    # and https://stackoverflow.com/questions/11380413/python-unittest-passing-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help="Desired output path")
    parser.add_argument('-u', '--url', default='http://localhost:8501', help="Demonstrator's URL to be tested")
    args = parser.parse_args()
    # Take Screenshot
    main(output_path=args.output, url=args.url)
