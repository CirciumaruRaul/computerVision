import cv2 as cv
import numpy as np
import project1_types as t

def show_image(img, name):
  cv.namedWindow(name, cv.WINDOW_NORMAL)
  cv.resizeWindow(name, 800, 600)
  cv.imshow(name, img)
  cv.waitKey(0) 
  cv.destroyAllWindows() 



img = cv.imread('train/5_00.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Step 3: Edge detection
blurred = cv.GaussianBlur(gray, (5, 5), 0)
# show_image(blurred, 'blur')
edges = cv.Canny(blurred, 200, 450, apertureSize=5)
show_image(edges, 'edges')
# Step 4: Line detection using Hough Transform
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=200, minLineLength=800, maxLineGap=1000)

# Step 5: Draw the lines
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    if abs(angle) < 10 or abs(angle - 90) < 10 or abs(angle + 90) < 10:
        # It's a mostly horizontal or vertical line
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Show the result
show_image(img, 'img')



def detect_board_and_warp(image_path, output_size=(800, 800)):
    # Load and resize image (optional)
    img = cv2.imread(image_path)
    # img = cv2.resize(img, (1000, 1000))  # scale for consistency
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #plot grayscaled image
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.show()
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=150, minLineLength=300, maxLineGap=20)
    
    debug_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_lines, (x1, y1), (x2, y2), (255, 0, 0), 5)
            
    # Approximate outer rectangle from Hough lines (very naive)
    # We assume that the outermost lines (bounding the board) are the longest horizontal/vertical
    verticals = []
    horizontals = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 20:  # vertical line
            verticals.append(line[0])
        elif abs(y1 - y2) < 20:  # horizontal line
            horizontals.append(line[0])
            
    top = min(horizontals, key=lambda l: l[1])
    bottom = max(horizontals, key=lambda l: l[1])
    left = min(verticals, key=lambda l: l[0])
    right = max(verticals, key=lambda l: l[0])
    
    src_pts = np.array([
        [left[0], top[1]],
        [right[2], top[1]],
        [right[2], bottom[3]],
        [left[0], bottom[3]]
    ], dtype='float32')

    dst_pts = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype='float32')

    # Perspective warp
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, output_size)

    return img, debug_lines, warped