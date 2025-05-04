import cv2
import numpy as np
import os
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

'''
--------------------------------------
--------------------------------------
---------- Patch definition ----------
--------------------------------------
'''
class Patch:
    def __init__(self, image, line_idx, column_idx):
        self.image = image
        self.line_idx = line_idx
        self.column_idx = column_idx
        self.scored = 0
        self.has_tile = 0 
        self.color = 0
        self.shape = 0


# ---------- Creation of the model ----------
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
# ])

# dataset = datasets.ImageFolder('train/shapes', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# model = models.resnet18(pretrained=True)
# num_classes = len(dataset.classes)
# model.fc = nn.Linear(model.fc.in_features, num_classes)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# for epoch in range(25):  # adjust as needed
#     model.train()
#     running_loss = 0.0
#     for images, labels in dataloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

# SAVE_PATH = "shape_classifier.pth"
# torch.save(model.state_dict(), SAVE_PATH)
# print(f"Saved model parameters to {SAVE_PATH}")
# ---------- End Creation ----------
'''
--------------------------------
--------------------------------
---------- Load model ----------
--------------------------------
'''
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
SAVE_PATH = "shape_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 9)

state_dict = torch.load(SAVE_PATH, map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

print("Model parameters loaded and ready to use.")

'''
------------------------------------
------------------------------------
---------- Label decoding ----------
-------------------------------------
'''
def decode_label(model_output_class):
    if 1 <= model_output_class <= 6:
        return model_output_class
    else:
        return 0

def predict_tile_shape(np_image, model, transform):
    if np_image.dtype != np.uint8:
        np_image = (np_image * 255).astype(np.uint8)  
    if np_image.ndim == 2: 
        np_image = np.expand_dims(np_image, axis=-1)
    if np_image.shape[2] == 1:  
        np_image = np.repeat(np_image, 3, axis=2)
    pil_image = Image.fromarray(np_image)

    image = transform(pil_image).unsqueeze(0).to(device)
    # model.eval()  # uncomment only if you trained first
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, 1).item()
    return decode_label(predicted_class)

'''
------------------------------------------------
------------------------------------------------
---------- Printing/Showing Functions ----------
------------------------------------------------
'''
def show_image(img, name='image', scale=0.3):
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_aligned_board(patches, show_presence=False):
    n_rows = len(patches)
    n_cols = len(patches[0])

    # Column headers A, B, C, …
    header = "     " + " ".join(f"{chr(65 + j):>3}" for j in range(n_cols))
    print(header)
    print("   " + "---" * (n_cols + 6) )

    # Each row
    for i in range(n_rows):
        row_label = f"{i:>2} |"  
        row_vals = []
        for p in patches[i]:
            if show_presence == False:
                string = str(p.shape) + str(p.color)
                row_vals.append(string)
            else:
                row_vals.append(p.scored)
        row_str = " ".join(f"{v:>3}" for v in row_vals)
        print(f"{row_label} {row_str}")


'''
-------------------------------------------------
-------------------------------------------------
---------- Getting data from directory ----------
-------------------------------------------------
'''
def get_games(baseDir, nrOfGames):
  files = os.listdir(baseDir) 
  imageNames = list(filter(lambda x: x.endswith(".jpg"), files))
  groundTruths = list(filter(lambda x: x.endswith(".txt"), files))

  listOfGames = []
  listOfGroundTruths = []
  for i in range(1, nrOfGames + 1):
    print(f"Getting files for game: {i}.")
    string = str(i) + "_"
    
    game = sorted(list(filter(lambda x: x.startswith(string), imageNames)), key=lambda x: int(x.split('_')[1].split('.')[0]))
    game = [baseDir + '/' + g for g in game]
    listOfGames.append(game)
    
    groundTruth = sorted(list(filter(lambda x: x.startswith(string), groundTruths)), key=lambda x: int(x.split('_')[1].split('.')[0]))
    listOfGroundTruths.append(groundTruth)

  return listOfGames, listOfGroundTruths

'''
------------------------------------------------
------------------------------------------------
----------- Tile Color and Detection -----------
------------------------------------------------
'''
COLOR_RANGES_HSV = {
    "R": [((0,150,50), (10,255,255)),
          ((170,150,50), (180,255,255))],
    "O": [((11,150,100), (25,255,255))],
    "Y": [((26,150,100), (34,255,255))],
    "G": [((35,100,50), (85,255,255))],
    "B": [((86,100,50), (135,255,255))],
    "W": [((0,0,200), (180,50,255))],
}

def detect_tile_color(tile_img, shape_mask=None):
    hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
    best_color, best_count = None, 0
    
    for color, ranges in COLOR_RANGES_HSV.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (low, high) in ranges:
            low  = np.array(low, dtype=np.uint8)
            high = np.array(high,dtype=np.uint8)
            mask |= cv2.inRange(hsv, low, high)
        
        if shape_mask is not None:
            mask &= shape_mask
        
        cnt = cv2.countNonZero(mask)
        if cnt > best_count:
            best_count, best_color = cnt, color
    
    return best_color

def tile_detect(cell_img, threshold=60, min_foreground_ratio=0.28):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    foreground_ratio = np.sum(binary == 255) / binary.size
    return foreground_ratio > min_foreground_ratio

'''
------------------------------------------------
------------------------------------------------
--------------- Board Extraction ---------------
------------------------------------------------
'''
# 
def preprocess_image(image, version):
  to_return = None
  match version:
    case "v1":
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      gray = cv2.bilateralFilter(gray, d=13, sigmaColor=17, sigmaSpace=17)

      edges = cv2.Canny(gray, 1, 10, apertureSize=3)
      to_return = cv2.dilate(edges, None, iterations=1)
    case "v2":
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      blurred = cv2.GaussianBlur(gray, (3, 3), 150, 250)
      to_return = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 15, 10)
  return to_return

def connect_lines_after_preprocessing(thresh):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
  connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  return connected

def find_contours(c, version):
  contours, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    print("No contours found.")
    return None
  match version:
    case "v1":
      return sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    case "v2":
      return max(contours, key=cv2.contourArea)

def find_rectangle_points(contours):
  for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
      _, _, w, h = cv2.boundingRect(approx)
      aspect_ratio = w / float(h)
      if 0.75 < aspect_ratio < 1.25:  
        return approx
  return None

def find_approx_points(contour):
  flag = False
  peri = cv2.arcLength(contour, True)
  approx = None
  ll = []
  for eps_ratio in [0.02, 0.03, 0.04, 0.05, 0.06, 0.01, 0.018]:
    approx = cv2.approxPolyDP(contour, eps_ratio * peri, True)
    ll.append(len(approx))
    if len(approx) == 4:
      break

  if len(approx) != 4:
    print(f"Expected 4 corners, but found {ll}")
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    flag = True
    return box, flag
  return approx, flag

def sort_points(points):
  points = points.reshape(4, 2)
  points = sorted(points, key=lambda x: (x[1], x[0]))
  top = sorted(points[:2], key=lambda x: x[0])
  bottom = sorted(points[2:], key=lambda x: x[0])
  return np.array([top[0], top[1], bottom[1], bottom[0]], dtype='float32')

def get_cut_out_board(src_points, image):
  (top_left, top_right, bottom_right, bottom_left) = src_points
  widthA = np.linalg.norm(bottom_right - bottom_left)    
  widthB = np.linalg.norm(top_right - top_left)

  heightA = np.linalg.norm(top_right - bottom_right)
  heightB = np.linalg.norm(top_left - bottom_left)
  maxWidth = int(max(widthA, widthB))
  maxHeight = int(max(heightA, heightB))

  dst_pts = np.array([
      [0, 0],                       # origin topLeft
      [maxWidth-1, 0],              # topRight
      [maxWidth-1, maxHeight-1],    # bottomRight
      [0, maxHeight-1]],            # bottomLeft
      dtype='float32')

  M = cv2.getPerspectiveTransform(src_points, dst_pts)
  return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def extract_board_v1(image, game='game', show_details=False):
    VERSION = "v1"
    edges = preprocess_image(image, VERSION)
    if show_details:
        show_image(edges)

    contours = find_contours(edges, VERSION)
    if show_details:
        details = image.copy()
        for contour in contours:
            cv2.drawContours(details, [contour], -1, (0, 0, 255), 10)  
        show_image(details)

    board_contour = find_rectangle_points(contours)
    if show_details and board_contour is not None:
      details = image.copy()
      cv2.drawContours(details, [board_contour], -1, (0, 0, 255), 10)
      show_image(details, "Detected Board Outline")
    if board_contour is None:
        return extract_board_v2(image=image, game=game, show_details=show_details)
    
    src_points = sort_points(board_contour)
    warped = get_cut_out_board(src_points, image)
    if show_details:
      show_image(warped, game)
    
    return warped

def extract_board_v2(image, game='game', show_details=False):
  VERSION = "v2"
  threshold = preprocess_image(image, VERSION)
  if show_details:
      show_image(threshold, "Threshold")

  connected = connect_lines_after_preprocessing(threshold)
  if show_details:
      show_image(connected, "Connected Lines")

  largest_contour = find_contours(connected, VERSION) 
  flag = False
  approx_points, flag = find_approx_points(largest_contour)

  src_points = sort_points(approx_points)
  warped = get_cut_out_board(src_points, image)
  if flag or show_details:
    show_image(warped, game)
  return warped

'''
------------------------------------------------
------------------------------------------------
--------------- Creating Patches ---------------
------------------------------------------------
'''
def get_patches(board, grid_size=16):
    cols = list(string.ascii_uppercase[:grid_size])
    height, width, _ = board.shape
    patch_height = height // grid_size
    patch_width = width // grid_size

    patches = [[None for _ in range(grid_size)] for _ in range(grid_size)]

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = max(i * patch_height, 0)
            y2 = min((i + 1) * patch_height, height)
            x1 = max(j * patch_width, 0)
            x2 = min((j + 1) * patch_width, width)
            
            patch = board[y1:y2, x1:x2]
            patches[i][j] = Patch(patch, line_idx=i+1, column_idx=cols[j])

    return patches
'''
------------------------------------------------
------------------------------------------------
----------------- Init Labels ------------------
------------------------------------------------
'''
def init_matrix(patches):
  for i in range(16):
    for patch in patches[i]:
      if tile_detect(patch.image):
        patch.has_tile = 1
        patch.scored = 1
        patch.shape = predict_tile_shape(patch.image, model, transform)
        if patch.shape != 0:
          patch.color = detect_tile_color(patch.image)
  return patches

'''
------------------------------------------------
------------------------------------------------
------------- Quadrants Functions --------------
------------------------------------------------
'''
def split_into_quadrants(board):
    arr = np.asarray(board)
    return {
        "top_left":     arr[:8,   :8],
        "top_right":    arr[:8,   8:],
        "bottom_left":  arr[8: ,  :8],
        "bottom_right": arr[8: ,  8:],
    }

def merge_quadrants(tl, tr, bl, br):
    # sanity check
    if any(arr.shape != tl.shape for arr in (tr, bl, br)):
        raise ValueError("All quadrants must have the same shape")
    # top half: [tl tr], bottom half: [bl br]
    top    = np.hstack((tl, tr))
    bottom = np.hstack((bl, br))
    return np.vstack((top, bottom))

'''
------------------------------------------------
------------------------------------------------
---------- Init configuration matrix -----------
------------------------------------------------
'''
CONFIG_1 = [
 [0, 0, 0, 0, 0, 0, 0, 0],
 [0, -2, 0, 0, 0, -1, 0, 0],
 [0, 0, 0, 0, -1, 0, -1, 0],
 [0, 0, 0, -1, 0, -1, 0, 0],
 [0, 0, -1, 0, -1, 0, 0, 0],
 [0, -1, 0, -1, 0, 0, 0, 0],
 [0, 0, -1, 0, 0, 0, -2, 0],
 [0, 0, 0, 0, 0, 0, 0, 0],
]

CONFIG_2 = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, -2, 0],
    [0, -1, 0, -1, 0, 0, 0, 0],
    [0, 0, -1, 0, -1, 0, 0, 0],
    [0, 0, 0, -1, 0, -1, 0, 0],
    [0, 0, 0, 0, -1, 0, -1, 0],
    [0, -2, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
WAS_RUN = 0
def map_quadrant(q):
  if q[1][1].has_tile:
    q = apply_config(q, CONFIG_2)
  else:
    q = apply_config(q, CONFIG_1)
  return q

def apply_config(q, config):
  for i in range(8):
    for j in range(8):
      q[i][j].has_tile += config[i][j]
  return q

def get_init_matrix_with_scores(patches):
  quadrants = split_into_quadrants(patches)
  new_q = []
  for q in quadrants.values():
    new_q.append(map_quadrant(q))
  return merge_quadrants(new_q[0], new_q[1], new_q[2], new_q[3])


'''
------------------------------------------------
------------------------------------------------
--------- Scoring and updating matrix ----------
------------------------------------------------
'''
def diff_patches_and_score(prev, current) -> (str, int): # type: ignore 
    diffs = []
    moves = []
    specials = 0
    R = len(prev)
    C = len(prev[0]) if R else 0

    for i in range(R):
        for j in range(C):
            before = prev[i][j]
            after  = current[i][j]
            if before.has_tile != after.has_tile and after.has_tile != 0 and before.scored == 0:
                moves.append((i, j))
                if before.has_tile < 0:
                    if abs(before.has_tile) == 2:
                        specials += 2
                    else:
                        specials += 1
                    before.has_tile = abs(before.has_tile) + after.has_tile
                else:
                    before.has_tile = after.has_tile
                before.shape = after.shape
                before.color = after.color
                before.scored = 1
                diffs.append(f"{after.line_idx}{after.column_idx} {after.shape}{after.color}")

    # No new tiles => zero score
    if not moves:
        return '', 0

    score = 0
    if len(moves) >= 2:
        if moves[0][0] == moves[1][0]:
            # columns
            lenght = 0
            score = moves[len(moves)-1][1] - moves[0][1] + 1
            for i,j in moves:
                while i-1 > -1 and prev[i][j].has_tile > 0: # to the top
                    lenght +=1  
                    i -= 1
                while i+1 < 16 and prev[i][j].has_tile > 0: # to the bottom
                    lenght +=1
                    i +=1 
            if lenght == 6: # Qwirkle bonus
                score = score + 2 * lenght
            score += lenght
        else:
            score = moves[len(moves)-1][0] - moves[0][0] + 1
            lenght = 0
            for i,j in moves:
                while j-1 > -1 and prev[i][j].has_tile > 0: # left
                    lenght +=1  
                    j -= 1

                while j+1 < 16 and prev[i][j].has_tile > 0: # right
                    lenght +=1
                    j +=1
            if lenght == 6:
                score = score + 2 * lenght
            score += lenght 
    else:
        i,j = moves[0]
        lenght = 0
        if i > -1:
            if prev[i-1][j].has_tile > 0:
                while i-1 > -1 and prev[i][j].has_tile > 0: # to the left
                    lenght +=1  
                    i -= 1
                while i+1 < 16 and prev[i][j].has_tile > 0: # to the right
                    lenght +=1
                    i +=1 
            else:
                while j-1 > -1 and prev[i][j].has_tile > 0: # left
                    lenght +=1  
                    j -= 1

                while j+1 < 16 and prev[i][j].has_tile > 0: # right
                    lenght +=1
                    j +=1
        if lenght == 6:
                score = score + 2 * lenght
        score += lenght
    if len(diffs) == 1:
        to_return = diffs[0] + '\n'
    elif len(diffs) == 0:  
        to_return = ""
    else:
        to_return = "\n".join(diffs) + ("\n" if diffs else "")
    return to_return, score+specials

'''
--------------------------------------------------------------------
--------------------------------------------------------------------
--------- What is done for init image with no moves played ---------
--------------------------------------------------------------------
'''
def flow_init(img, SHOW_DETAILS=False):
  init_img = cv2.imread(img)
  if SHOW_DETAILS:
    show_image(init_img, name=img)
  board = extract_board_v1(init_img, game=img, show_details=SHOW_DETAILS)
  if board is None:
    print("problem!")
    return None
  patches = get_patches(board)
  if len(patches) == len(patches[0]) and len(patches) != 0:
    matrix = init_matrix(patches=patches)
    matrix = get_init_matrix_with_scores(matrix)
  return matrix

'''
--------------------------------------------------
--------------------------------------------------
--------- Write extraction to directiory ---------
--------------------------------------------------
'''
def write_to_dir(target_dir, filename, data):
    full_path = os.path.join(target_dir, filename)

    # Ensure all parent directories exist
    parent_dir = os.path.dirname(full_path)
    os.makedirs(parent_dir, exist_ok=True)

    # Now safe to open and write
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(data)


'''
--------------------------------------------------
--------------------------------------------------
------------- Main loop for running --------------
--------------------------------------------------
'''

NUMBER_OF_GAMES = 5
IMAGES_PER_GAME = 20
DIRECTORY_OF_IMAGES = 'train'           # Relative to current path
games, txt = get_games(DIRECTORY_OF_IMAGES, NUMBER_OF_GAMES)

init_img = None
SHOW_DETAILS = False
prev = None
current = None

for i in range(NUMBER_OF_GAMES):
  for j in range(IMAGES_PER_GAME + 1): # 1 from the reference image aka the 1_00
    img = games[i][j]
    if img.endswith('_00.jpg'):
      prev = flow_init(img)
    else:
      image = cv2.imread(img)
      board = extract_board_v1(image, game=img, show_details=SHOW_DETAILS)
      if board is None:
        print("Could not find a valid Board!")
        break
      patches = get_patches(board)
      current = init_matrix(patches)
      diffs, score = diff_patches_and_score(prev=prev, current=current)
      filename = img.replace(DIRECTORY_OF_IMAGES + "/", '')
      filename = filename.replace('.jpg', '.txt')
      data = diffs + str(score)
      write_to_dir(target_dir='evaluation/submission_files/Circiumaru_Raul_407/', filename=filename, data=data)
