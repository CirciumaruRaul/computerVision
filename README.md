# How to run the project

## Use the following file in accordance with the OS your using. Make sure the file has the right permisions to be executed

- Windows: Double click `windows.bat` file
- Linux: Execute `./linux.sh` file
- MacOS: Double click `macos.command` file
### Note: you will be prompted to select if you want to run the .py file as well, please make sure you have the .py file and the script file in the same directory as the data

### The above script will do the following:
 - Create a venv
 - Upgrade pip and install dependencies
 - Prompt you if you wish to run the project (Y|N)
 - Automatically delete the .venv in the end leaving only the created .txt files

If you wish to run it manually, the required dependencies are in requirements.txt file. By running `pip install -r requirements.txt` you should have all dependencies needed.

## What if it does not have the permission:

 - Linux & MacOS you can use `chmod +x linux.sh` or with if it says denied or not permited`sudo chmod +x linux.sh`.

 - Windows try using by `right click + Run as Administrator`. 
If not make sure to have Read&Execute by `right click + Show More + Properties -> Security -> Edit(add them if missing)`.



## Program Output

The .txt files outputed should be in `evaluation/submission_files/Circiumaru_Raul_407/` if you need it somewhere else change `OUTPUT_DIR` to the desired relative path to the project.