import cv2 as cv # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import logging as log
import os
import project1_types as types

# Basic logging configs PS: it just looks nice in the console :)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[log.FileHandler('app.log', mode='w'), # low-key good for debugging and pretty in general
                          log.StreamHandler()])

def show_image(img, name):
  cv.namedWindow(name, cv.WINDOW_NORMAL)
  cv.resizeWindow(name, 800, 600)
  cv.imshow(name, img)
  cv.waitKey(0) 
  cv.destroyAllWindows() 

def get_games(baseDir, nrOfGames):
  '''
    Returns a list of lists where each list represents a game
  '''
  files = os.listdir(baseDir) 
  imageNames = list(filter(lambda x: x.endswith(".jpg"), files))
  groundTruths = list(filter(lambda x: x.endswith(".txt"), files))
  listOfGames = []
  listOfGroundTruths = []
  for i in range(1, nrOfGames + 1):
    # current game
    log.info(f"Getting files for game: {i}.")
    string = str(i) + "_"
    
    game = sorted(list(filter(lambda x: x.startswith(string), imageNames)), key=lambda x: int(x.split('_')[1].split('.')[0]))
    listOfGames.append(game)
    
    groundTruth = sorted(list(filter(lambda x: x.startswith(string), groundTruths)), key=lambda x: int(x.split('_')[1].split('.')[0]))
    listOfGroundTruths.append(groundTruth)

  return listOfGames, listOfGroundTruths

images, texts = get_games('train', 5)

def draw_lines(image, lines: [types.Line], timeout: int = 0, color: tuple = (0, 0, 255), # type: ignore
               return_drawing: bool = False, window_name: str = 'window'):
    """
    Plots the lines into the image.
    """
    drawing = image.copy()
    if drawing.ndim == 2:
        drawing = cv.cvtColor(drawing, cv.COLOR_GRAY2BGR)
    for line in lines: 
        cv.line(drawing, line.point_1.get_point_as_tuple(), line.point_2.get_point_as_tuple(), color, 2, cv.LINE_AA)
        
    if return_drawing:
        return drawing
    else:
        show_image(drawing, window_name=window_name, timeout=timeout)

def remove_close_lines(lines: [types.Line], threshold: int, is_vertical: bool): # type: ignore
    """
    It removes the closest lines.
    """
    different_lines = [] 
    if is_vertical:
        lines.sort(key=lambda line: line.point_1.x)
    else:
        lines.sort(key=lambda line: line.point_1.y)
    
    #  add the first line
    different_lines.append(lines[0])
    if is_vertical:
        for line_idx in range(1, len(lines)):
            if lines[line_idx].point_1.x - different_lines[-1].point_1.x > threshold:
                different_lines.append(lines[line_idx])
    else:
        for line_idx in range(1, len(lines)): 
            if lines[line_idx].point_1.y - different_lines[-1].point_1.y > threshold:
                different_lines.append(lines[line_idx])
    return different_lines

def get_patches(lines: [types.Line], columns: [types.Line], image, show_patches: bool = False) -> [types.Patch]: # type: ignore
    """
    It cuts out each box from the table defined by the lines and columns.
    """
    
    def crop_patch(image_, x_min, y_min, x_max, y_max):
        """
        Crops the bounding box represented by the coordinates.
        """
        return image_[y_min: y_max, x_min: x_max].copy()
    
    def draw_patch(image_, patch: types.Patch, color: tuple = (255, 0, 255)): 
        """
        Draw the bounding box corresponding to the patch on the image.
        """
        cv.rectangle(image_, (patch.x_min, patch.y_min), (patch.x_max, patch.y_max), color=color, thickness=5)
    
    assert image.ndim == 2
    if show_patches: 
        image_color = np.dstack((image, image, image))
  
    lines.sort(key=lambda line: line.point_1.y)
    columns.sort(key=lambda column: column.point_1.x)
    patches = []
    step = 5
    for line_idx in range(len(lines) - 1):
        for col_idx in range(len(columns) - 1):
            current_line = lines[line_idx]
            next_line = lines[line_idx + 1] 
            
            y_min = current_line.point_1.y + step
            y_max = next_line.point_1.y - step
            
            current_col = columns[col_idx]
            next_col = columns[col_idx + 1]
            x_min = current_col.point_1.x + step 
            x_max = next_col.point_1.x - step
            
            patch = types.Patch(image_patch=crop_patch(image,  x_min, y_min, x_max, y_max),
                          x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                          line_idx=line_idx, column_idx=col_idx)
            
            if show_patches:
                draw_patch(image_color, patch)
            
            patches.append(patch)
            
    if show_patches:
        show_image(image_color, window_name='patches', timeout=0)
    return patches


def match_features(features_source, features_dest) -> [[cv.DMatch]]:
    """
    Match features from the source image with the features from the destination image.
    :return: [[DMatch]] - The rusult of the matching. For each set of features from the source image,
    it returns the first 'K' matchings from the destination images.
    """
    feature_matcher = cv.DescriptorMatcher_create("FlannBased")
    matches = feature_matcher.knnMatch(features_source, features_dest, k=2)   
    return matches


def get_key_points_and_features(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    sift = cv.SIFT_create() 
    keypoints = sift.detect(gray_image, None)
    keypoints, features = sift.compute(gray_image, keypoints) 
        
    return keypoints, features

def generate_homography(all_matches:  [cv.DMatch], keypoints_source: [cv.KeyPoint], keypoints_dest : [cv.KeyPoint],
                        ratio: float = 0.75, ransac_rep: int = 4.0):
    if not all_matches:
        return None
    
    matches = [] 
    for match in all_matches:  
        if len(match) == 2 and (match[0].distance / match[1].distance) < ratio:
            matches.append(match[0])
     
    points_source = np.float32([keypoints_source[m.queryIdx].pt for m in matches])
    points_dest = np.float32([keypoints_dest[m.trainIdx].pt for m in matches])

    if len(points_source) > 4:
        H, status = cv.findHomography(points_source, points_dest, cv.RANSAC, ransac_rep)
        return H
    else:
        return None

refFileName = "board+tiles/tiles_3.jpg"
log.info("Read Ground Truth")
refImage = cv.imread(refFileName)
refImage = cv.cvtColor(refImage, cv.COLOR_BGR2RGB)

refFileName = "board+tiles/tiles_4.jpg"
log.info("Read Image")
currentImage = cv.imread(refFileName)
currentImage = cv.cvtColor(currentImage, cv.COLOR_BGR2RGB)


keypoints_source, featuresSrc = get_key_points_and_features(refImage)
keypoints_dest, featuresDest = get_key_points_and_features(currentImage)
all_matches = match_features(featuresSrc, featuresDest)
H = generate_homography(all_matches, keypoints_source, keypoints_dest, 0.75, 4.0)

result = cv.warpPerspective(refImage, H, 
        (currentImage.shape[1] + refImage.shape[1], currentImage.shape[0]))
show_image(result, 'r')
# Show side-by-side
combined = np.hstack((refImage, result))
show_image(combined, 'c')
# show_image(img3, "img3")

