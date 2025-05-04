#!/bin/sh

# ————————————————————————————————
#  Qwirkle Project Launcher for Linux
# ————————————————————————————————


python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# ask if run
read -p "Do you want to run the Python script (Y | N)? " run

if [ "$run" = "Y" ] || [ "$run" = "y" ]; then
  python3 project.py
else
  echo "Skipping execution."
fi

echo "The project created a venv"
read -p "Do you wish to clean the .venv folder (Y|N)? " clean

if [ "$clean" = "Y" ] || [ "$clean" = "y" ]; then
  rm -rf .venv
  echo "Succesfully removed the .venv folder created"
fi

echo "All done."