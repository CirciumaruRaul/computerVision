import cv2 as cv
import numpy as np
import project1_types as t


def show_image(image, window_name='image', timeout=0):
    """
    :param timeout. How many seconds to wait untill it close the window.
    """
    cv.imshow(window_name, cv.resize(image, None, fx=0.6, fy=0.6))
    cv.waitKey(timeout)
    cv.destroyAllWindows()

def draw_lines(image, lines: [t.Line], timeout: int = 0, color: tuple = (0, 0, 255),
               return_drawing: bool = False, window_name: str = 'window'):
    """
    Plots the lines into the image.
    :param image.
    :param lines.
    :param timeout. How many seconds to wait untill it close the window.
    :param color. The color used to draw the lines
    :param return_drawing. Use it if you want the drawing to be return instead of displayed.
    :return None if return_drawing is False, otherwise returns the drawing.
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

def get_horizontal_and_vertical_lines_hough(edges, threshold: int = 160) -> tuple:
    """
    Returns the horizontal and vertical lines found by Hough transform.
    :param edges = the edges of the image.
    :threshold = it specifies how many votes need a line to be considered a line.
    :return (horizontal_lines: List(Line), vertical_lines: List(Line))
    """
    
    lines = cv.HoughLines(edges,1,np.pi/180,threshold)
    assert lines is not None
    
    vertical_lines: [t.Line] = []
    horizontal_lines: [t.Line] = [] 
    
    
    
    
    for i in range(0, len(lines)):
        # TODO: compute the line coordinate
        rho=lines[i][0][0]
        theta=lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
       
        pt1 = (x1,y1) # x, y
        pt2 = (x2,y2) # x, y
        
        if  1.4 <= theta <= 1.65:
            line = t.Line(t.Point(x=pt1[0], y=pt2[1]), t.Point(x=pt2[0], y=pt2[1])) 
            horizontal_lines.append(line)
        else:
            line = t.Line(t.Point(x=pt2[0], y=pt1[1]), t.Point(x=pt2[0], y=pt2[1])) 
            vertical_lines.append(line) 
             
    return horizontal_lines, vertical_lines


def get_patches(lines: [t.Line], columns: [t.Line], image) -> [t.Patch]:
    """
    It cuts out each box from the table defined by the lines and columns.
    """
    
    def crop_patch(image_, x_min, y_min, x_max, y_max):
        """
        Crops the bounding box represented by the coordinates.
        """
        return image_[y_min: y_max, x_min: x_max].copy()
    
    def draw_patch(image_, patch: t.Patch, color: tuple = (255, 0, 255)):
        """
        Draw the bounding box corresponding to the patch on the image.
        """
        cv.rectangle(image_, (patch.x_min, patch.y_min), (patch.x_max, patch.y_max), color=color, thickness=5)
    
    assert image.ndim == 2
  
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
            
            patch = t.Patch(image_patch=crop_patch(image,  x_min, y_min, x_max, y_max),
                          x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                          line_idx=line_idx, column_idx=col_idx)
            
            patches.append(patch)
            
    return patches

def remove_close_lines(lines: [t.Line], threshold: int, is_vertical: bool):
    """
    It removes the closest lines.
    :param lines.
    :param threshold. It specify when the lines are too close to each other.
    :param is_vertical. Set it to True or False.
    :return : The different lines.
    """
    
    different_lines = [] 
    if is_vertical:
        lines.sort(key=lambda line: line.point_1.x)
    else:
        lines.sort(key=lambda line: line.point_1.y)
    
    if len(lines) != 0:
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

def find_patches_hough(image, show_intermediate_results=False) -> [t.Patch]:
    gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_image,100,150)
    if show_intermediate_results: 
        show_image(edges, window_name='edges', timeout=0)
        
    horizontal_lines, vertical_lines = get_horizontal_and_vertical_lines_hough(edges,160)
    
    distinct_horizontal_lines = remove_close_lines(horizontal_lines,threshold=30,is_vertical=False)
    distinct_vertical_lines = remove_close_lines(vertical_lines,threshold=30,is_vertical=True)
    
    # take the last 5 verical lines and the last 16 horizontal lines
    distinct_horizontal_lines = distinct_horizontal_lines[-16:]
    distinct_vertical_lines = distinct_vertical_lines[-5:]
    
    patches = get_patches(distinct_horizontal_lines, distinct_vertical_lines, gray_image)
     
    return patches


def get_horizontal_and_vertical_lines_hough(edges, threshold: int = 160) -> tuple:
    """
    Returns the horizontal and vertical lines found by Hough transform.
    :param edges = the edges of the image.
    :threshold = it specifies how many votes need a line to be considered a line.
    :return (horizontal_lines: List(Line), vertical_lines: List(Line))
    """
    
    lines = cv.HoughLines(edges, 1, np.pi/180, threshold)
    assert lines is not None
    
    vertical_lines: [t.Line] = []
    horizontal_lines: [t.Line] = [] 
    
    for i in range(0, len(lines)):
        rho=lines[i][0][0]
        theta=lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
       
        pt1 = (x1,y1) # x, y
        pt2 = (x2,y2) # x, y
        
        if  1.4 <= theta <= 1.65:
            line = t.Line(t.Point(x=pt1[0], y=pt2[1]), t.Point(x=pt2[0], y=pt2[1])) 
            horizontal_lines.append(line)
        else:
            line = t.Line(t.Point(x=pt2[0], y=pt1[1]), t.Point(x=pt2[0], y=pt2[1])) 
            vertical_lines.append(line) 
             
    return horizontal_lines, vertical_lines

def classify_patches_with_magic_classifier(patches: [t.Patch]) -> None:
    # TODO
    magic_classifier = t.MagicClassifier()
    for patch in patches:
        patch.set_x(magic_classifier.classify(patch))

original_image = cv.imread('train/4_13.jpg')
original_image = cv.resize(original_image, None, fx=0.3, fy=0.3)
patches_left_image = find_patches_hough(original_image, 
                                        show_intermediate_results=True)
# patches_right_image = find_patches_hough(original_image, is_left=False, 
#                                         show_intermediate_results=True)
classify_patches_with_magic_classifier(patches_left_image) 
# classify_patches_with_magic_classifier(patches_right_image) 