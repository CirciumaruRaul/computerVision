#!/usr/bin/env zsh

# ————————————————————————————————
#  Qwirkle Project Launcher for macOS
# ————————————————————————————————

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

read -q "?Do you want to run project.py now? [Y/n] " run
echo
if [[ "$run" =~ ^[Yy]$ || -z "$run" ]]; then
  echo "Running project.py…"
  python3 project.py
else
  echo "Skipping execution."
fi

echo "Note: The project created a venv"
read -q "Do you wish to remove the virtualenv and logs? [y/N] " clean
echo
if [[ "$clean" =~ ^[Yy]$ ]]; then
  echo "Removing .venv files…"
  rm -rf .venv app.log
  echo "Cleanup complete."
else
  echo "Leaving .venv in place."
fi

echo "All done."
