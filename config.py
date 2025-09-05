# Configuration file for Point Collision Simulation
# All configurable parameters are centralized here for easy modification

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Point Collision Simulation"
FPS = 60
BACKGROUND_COLOR = (0, 0, 0)  # Black

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================
DEFAULT_NUM_POINTS = 5
DEFAULT_ENERGY_FACTOR = 1.0
BASE_SPEED = 100  # pixels per second
MAX_POINTS = 20
MIN_POINTS = 1

# Energy factor limits
MIN_ENERGY_FACTOR = 0.5
MAX_ENERGY_FACTOR = 2.0
ENERGY_FACTOR_STEP = 0.05

# =============================================================================
# POINT SETTINGS
# =============================================================================
POINT_RADIUS = 10
POINT_MASS = 1.0
MAX_POINT_SPEED = 2000  # Maximum allowed speed to prevent instability
MIN_DISTANCE_BETWEEN_POINTS = 50  # Minimum distance when generating points
MAX_PLACEMENT_ATTEMPTS = 1000

# Point color range (RGB values)
POINT_COLOR_MIN = 100
POINT_COLOR_MAX = 255

# =============================================================================
# LINE SYSTEM SETTINGS
# =============================================================================
LINES_PER_POINT = 3  # Number of lines connecting each point to the wall
LINE_ARC_DEGREES = 6  # Arc length in degrees between adjacent lines
LINE_WIDTH = 2  # Thickness of the lines drawn to the wall
LINE_ALPHA = 200  # Transparency of lines (0-255, 255 = opaque)

# Line collision detection
LINE_COLLISION_TOLERANCE = 5  # Distance tolerance for point-line intersection detection
LINE_COLLISION_SUBDIVISIONS = 10  # Number of subdivisions for path-based collision detection
MIN_LINE_SEGMENT_LENGTH = 2  # Minimum length for line segment collision detection

# =============================================================================
# PHYSICS SETTINGS
# =============================================================================
MAX_TIME_STEP = 0.016  # Maximum time step (16ms) to prevent instability
MAX_IMPULSE = 500  # Limit impulse magnitude to prevent extreme velocity changes
MAX_SEPARATION = 20  # Maximum separation distance when resolving overlaps
MAX_SPEED_INCREASE_PER_COLLISION = 1.5  # Maximum speed multiplier per collision

# Boundary settings
CIRCLE_MARGIN = 50  # Distance from window edge to circle boundary
BOUNDS_BUFFER = 100  # How far outside window bounds points can go before reset
RESET_RANDOMNESS = 50  # Random offset when resetting points to center

# =============================================================================
# RENDERING SETTINGS
# =============================================================================
CIRCLE_BOUNDARY_COLOR = (100, 100, 100)  # Gray
CIRCLE_BOUNDARY_WIDTH = 2

# Safe rendering limits
MAX_RENDER_RADIUS = 50
MIN_RENDER_RADIUS = 1

# =============================================================================
# UI SETTINGS
# =============================================================================
UI_FONT_SIZE = 36
UI_TEXT_COLOR = (255, 255, 255)  # White
UI_TITLE_COLOR = (200, 200, 200)  # Light gray
UI_SECONDARY_COLOR = (150, 150, 150)  # Medium gray

# UI positioning
UI_MARGIN = 10
UI_LINE_HEIGHT = 40
UI_STATS_SPACING = 40

# Controls positioning
CONTROLS_FROM_BOTTOM = 150
CONTROLS_LINE_HEIGHT = 25

# =============================================================================
# CONTROL SETTINGS
# =============================================================================
# Key bindings (using pygame constants)
import pygame

KEY_EXIT = pygame.K_ESCAPE
KEY_ENERGY_UP = pygame.K_UP
KEY_ENERGY_DOWN = pygame.K_DOWN
KEY_ADD_POINT_1 = pygame.K_PLUS
KEY_ADD_POINT_2 = pygame.K_EQUALS  # For keyboards where + requires shift
KEY_REMOVE_POINT = pygame.K_MINUS
KEY_RESET = pygame.K_r
KEY_TOGGLE_UI = pygame.K_h  # 'H' for hide/show

# Control descriptions for UI
CONTROLS = [
    "Controls (H to hide):",
    "UP/DOWN: Adjust energy factor",
    "+/-: Add/remove points", 
    "R: Reset simulation",
    "H: Toggle UI display",
    "ESC: Exit"
]

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================
# Safety values for invalid calculations
SAFE_FALLBACK_X = 400
SAFE_FALLBACK_Y = 300
SAFE_FALLBACK_SPEED = 0

# Numerical stability
MIN_DISTANCE_EPSILON = 0.1  # Minimum distance to prevent division by zero
MIN_SPEED_EPSILON = 0.001   # Minimum speed to consider for calculations
