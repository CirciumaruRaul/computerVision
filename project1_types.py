import numpy as np
class Line:
    """
    Store a line based on the two points.
    """
    def __init__(self, point_1, point_2):
        self.point_1 = point_1
        self.point_2 = point_2
        

class Point:
    def __init__(self, x, y):
        self.x = np.int32(np.round(x))
        self.y = np.int32(np.round(y))
    
    def get_point_as_tuple(self):
        return (self.x, self.y)
    
        
class Patch:
    """
    Store information about each item (where the mark should be found) in the table. 
    """
    def __init__(self, image_patch, x_min, y_min, x_max, y_max, line_idx, column_idx):
        self.image_patch = image_patch
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.line_idx = line_idx
        self.column_idx = column_idx
        self.has_x: int = 0 # 0 meaning it does not contain an 'X', 1 meaning it contains an 'X'
    
    def set_x(self, has_x: int):
        assert has_x == 0 or has_x == 1 # convention 
        self.has_x = has_x


class MagicClassifier:
    """
    A very strong classifier that detects if the patch has an 'X' or not.
    """
    def __init__(self):
        self.threshold = 245
    
    def classify(self, patch: Patch) -> int:
        """
        Receive a Patch and return 1 if there is an 'X' in the pacth and 0 otherwise.
        """ 
        if patch.image_patch.mean() > self.threshold:
            return 0
        else: 
            return 1

if __name__ == "__main__":
    pass