import numpy as np
from numpy.typing import NDArray

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = COLORS = {
    # Original colors
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "yellow": np.array([255, 255, 0]),
    "purple": np.array([112, 39, 195]),
    "grey": np.array([100, 100, 100]),
    "orange": np.array([255, 165, 0]),
    "cyan": np.array([0, 255, 255]),
    "pink": np.array([255, 192, 203]),
    "brown": np.array([165, 42, 42]),
    
    # Additional colors
    "navy": np.array([0, 0, 128]),
    "teal": np.array([0, 128, 128]),
    "olive": np.array([128, 128, 0]),
    "maroon": np.array([128, 0, 0]),
    "lime": np.array([0, 255, 0]),
    "aqua": np.array([0, 255, 255]),
    "silver": np.array([192, 192, 192]),
    "magenta": np.array([255, 0, 255]),
    "coral": np.array([255, 127, 80]),
    "khaki": np.array([240, 230, 140]),
    "indigo": np.array([75, 0, 130]),
    "gold": np.array([255, 215, 0]),
    "violet": np.array([238, 130, 238]),
    "tan": np.array([210, 180, 140]),
    "salmon": np.array([250, 128, 114]),
    "plum": np.array([221, 160, 221]),
    "turquoise": np.array([64, 224, 208]),
    "chocolate": np.array([210, 105, 30]),
    "tomato": np.array([255, 99, 71]),
    "wheat": np.array([245, 222, 179]),
    
    # Even more colors (using darker/lighter variations)
    "darkred": np.array([139, 0, 0]),
    "darkgreen": np.array([0, 100, 0]),
    "darkblue": np.array([0, 0, 139]),
    "darkorange": np.array([255, 140, 0]),
    "darkpurple": np.array([72, 61, 139]),
    "darkgrey": np.array([169, 169, 169]),
    "lightred": np.array([255, 127, 127]),
    "lightgreen": np.array([144, 238, 144]),
    "lightblue": np.array([173, 216, 230]),
    "lightorange": np.array([255, 200, 150]),
    
    # Use RGB combinations for even more colors
    "rgb1": np.array([128, 64, 0]),
    "rgb2": np.array([192, 64, 0]),
    "rgb3": np.array([64, 192, 0]),
    "rgb4": np.array([0, 64, 192]),
    "rgb5": np.array([192, 0, 64]),
    "rgb6": np.array([64, 0, 192]),
    "rgb7": np.array([128, 128, 64]),
    "rgb8": np.array([64, 128, 128]),
    "rgb9": np.array([128, 64, 128]),
    "rgb10": np.array([200, 100, 50]),
}
CTF_COLORS: dict[str, NDArray[np.uint]] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
    "red_grey": np.array([170, 152, 169]),
    "blue_grey": np.array([140, 146, 172]),
}

MAZE_COLORS: dict[str, NDArray[np.uint]] = {
    "red": np.array([228, 3, 3]),
    "orange": np.array([255, 140, 0]),
    "yellow": np.array([255, 237, 0]),
    "green": np.array([0, 128, 38]),
    "blue": np.array([0, 77, 255]),
    "purple": np.array([117, 7, 135]),
    "brown": np.array([120, 79, 23]),
    "grey": np.array([100, 100, 100]),
    "light_red": np.array([255, 228, 225]),
    "light_blue": np.array([240, 248, 255]),
    "white": np.array([255, 250, 250]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX: "dict[str, int]" = {key: i for i, key in enumerate(COLORS.keys())}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

# Map of object types to short string
OBJECT_TO_STR = {
    "wall": "x",
    "floor": "F",
    "door": "D",
    "key": "K",
    "ball": "o",
    "box": "B",
    "goal": "G",
    "lava": "V",
    "agent": "a",
}

# Short string for opened door
OPENED_DOOR_IDS = "_"

# Map agent's direction to short string
AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
