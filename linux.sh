#!/bin/sh

# create the virtual environment
python3 -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# ask if run
read -p "Do you want to run the Python script (Y | N)? " run

# Check if the user entered 'Y' or 'y' to run the script
if [ "$run" = "Y" ] || [ "$run" = "y" ]; then
  python3 project.py
fi

# inform and ask if cleanup
echo "The project created a venv"
read -p "Do you wish to clean the created files (Y|N)? " clean

if [ "$clean" = "Y" ] || [ "$clean" = "y" ]; then
  rm -rf app.log
  echo "Succesfully removed the additional files created"
fi

echo "All done."