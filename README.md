# Overview of the project

## Scope 
Automated tracking score of qwirkle game using images taken from different angles.

## General Arhitecture of the implementation
- Board Extraction using countours and selecting the biggest 4 that appear or simply the biggest one if there are less then 4 contours found if no contour found default to taking the biggest rectangle that it can find using minAreaRect opencv function
-  Divede the extracted board into patches as the image is now just the board regardeless of the angle that the picture was taken
-  There are 4 quadrants and each quadrant has 2 possible configs, determine what config is used by detecting if a starting tile is placed on a specific square
-  Init general matrix for the entire board
-  If there is a tile detect, using the resnet18 finetuned with my shapes decide what shape each new tile is and compute its color using a range of possibilities for each color
- Output is a txt file named "i_j.txt" (i = number of game, j = the move correspondent) per each image except the first image that dictates the configuration for the starting board

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
