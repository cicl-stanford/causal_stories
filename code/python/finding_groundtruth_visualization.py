import math
import random
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from conditions import red, black, grey, yellow, blue

# =========================
# CONFIGURATIONS
# =========================

# Simulation parameters
DEFAULT_SEED = 1
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
FPS = 50
TIME_STEP = 1 / 50.0
GRAVITY = (0, -981)
BLOCK_SIZE = 35
BLOCK_MASS = 1
RAMP_ANGLE = 29.05
VELOCITY_THRESHOLD = 0.01

# Positions
BLOCK_START_X = 645 
BLOCK_START_Y = 251
BLOCK_BACKWARD_START_X = 665 
RAMP_LEFT_X = 520
RAMP_RIGHT_X = 790
RAMP_POSITION_Y = 150
RAMP_HEIGHT = 150
GROUND_POSITION_X = 450
GROUND_POSITION_Y = 150

# Target positions
POSITION_1 = 223
POSITION_2 = 337
POSITION_3 = 973
POSITION_4 = 1087

# Fixed ground friction
GROUND_FRICTION = 0.5

OPTIMAL_RED_FRICTION = 0.2640
OPTIMAL_BLACK_FRICTION =  0.3929
OPTIMAL_GREY_FRICTION = 0.3015
OPTIMAL_GREY_RAMP_FRICTION = 0.2012
OPTIMAL_YELLOW_RAMP_FRICTION = 0.5698
OPTIMAL_BLUE_RAMP_FRICTION = 0.1023
OPTIMAL_FORCE = 557.4

# Final positions from optimization
FINAL_POSITIONS = {
    'red_up': 231.0,
    'red_down': 1089.0,
    'black_up': 327.1,
    'black_down': 972.8,
    'blue_up': 223.0,
    'blue_down': 1073.4,
    'yellow_up': 332.3,
    'yellow_down': 981.0
}

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
# Darker yellow for better contrast
YELLOW = (204, 204, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (150, 150, 150)

# Y-position offsets for different scenarios
Y_OFFSETS = {
    'red_up': 0,
    'red_down': 0,
    'black_up': 0,
    'black_down': 0,
    'blue_up': 60,
    'blue_down': 60,
    'yellow_up': 60,
    'yellow_down': 60
}

# =========================
# SETTING UP THE SIMULATION
# =========================

def set_reproducibility_seeds(seed=DEFAULT_SEED):
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        pygame.random.seed(seed)
    except AttributeError:
        pass  

class CustomDrawOptions(pymunk.pygame_util.DrawOptions):
    def __init__(self, surface):
        super().__init__(surface)
    def draw_polygon(self, verts, radius, outline_color, fill_color):
        flipped_verts = [self._flip_y(v) for v in verts]
        super().draw_polygon(flipped_verts, radius, outline_color, fill_color)
    def draw_dot(self, size, pos, color):
        flipped_pos = self._flip_y(pos)
        super().draw_dot(size, flipped_pos, color)
    def _flip_y(self, pos):
        return pymunk.Vec2d(pos.x, -pos.y + SCREEN_HEIGHT)

def create_block(color, friction):
    """Create a block with exact friction"""
    friction = max(0.01, friction)  # Ensure positive friction
    
    mass = BLOCK_MASS
    moment = pymunk.moment_for_box(mass, (BLOCK_SIZE, BLOCK_SIZE))
    body = pymunk.Body(mass, moment)
    body.angle = math.radians(180 - (180 - RAMP_ANGLE))
    shape = pymunk.Poly.create_box(body, (BLOCK_SIZE, BLOCK_SIZE))
    shape.friction = friction
    shape.color = color
    body.position = (BLOCK_START_X, BLOCK_START_Y)
    return body, shape

def create_ramp(color, friction, direction):
    """Create a ramp with exact friction"""
    friction = max(0.01, friction)  # Ensure positive friction
    
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    if direction == "forward":
        body.vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_LEFT_X, RAMP_HEIGHT)]
    else:
        body.vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_RIGHT_X, RAMP_HEIGHT)]
    body.position = (0, RAMP_POSITION_Y)
    shape = pymunk.Poly(body, body.vertices)
    shape.friction = friction
    shape.color = color
    return body, shape

def create_world(block, ramp, visualization=False):
    """Create the physics world"""
    space = pymunk.Space()
    space.gravity = GRAVITY
    forward_ramp_vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_LEFT_X, RAMP_HEIGHT)]
    backward_ramp_vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_RIGHT_X, RAMP_HEIGHT)]
    
    if ramp[0].vertices == forward_ramp_vertices:
        block[0].angle = math.radians(180 - RAMP_ANGLE)
    elif ramp[0].vertices == backward_ramp_vertices:
        block[0].angle = math.radians(360 - (180 - RAMP_ANGLE))
        block[0].position = (BLOCK_BACKWARD_START_X, BLOCK_START_Y)
    
    space.add(ramp[0], ramp[1])
    space.add(block[0], block[1])
    
    # Create ground
    ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_body.position = (GROUND_POSITION_X, GROUND_POSITION_Y)
    ground_shape = pymunk.Poly.create_box(ground_body, (5000, 10))
    ground_shape.friction = GROUND_FRICTION
    ground_shape.color = grey
    space.add(ground_body, ground_shape)
    
    return space

def run_simulation(space, dynamics, force_magnitude, visualization=False):
    """Run the simulation - CORRECT FORCE APPLICATION"""
    # Find the block in the space
    block = [body for body in space.bodies if body.body_type != pymunk.Body.STATIC][0]
    
    if dynamics == "up":
        # Apply force up the ramp
        ramp_angle = math.radians(RAMP_ANGLE)
        force_vector = pymunk.Vec2d(
            force_magnitude * math.cos(ramp_angle), 
            force_magnitude * math.sin(ramp_angle)
        )
        block.apply_impulse_at_local_point(force_vector)
    elif dynamics == "down":
        # NO FORCE - just let gravity work
        pass

    for frame in range(4000):  # MAX_SIMULATION_FRAMES
        space.step(TIME_STEP)
        if block is None:
            return None
        if block.velocity.length < VELOCITY_THRESHOLD:
            return block.position.x

        if visualization:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
            screen.fill(WHITE)
            space.debug_draw(draw_options)
            pygame.display.flip()
            clock.tick(FPS)

    return None

def draw_position_markers(screen, positions):
    """Draw markers for all final positions with different y-offsets"""
    font = pygame.font.Font(None, 15)
    
    # Define colors for each scenario
    scenario_colors = {
        'red_up': RED,
        'red_down': RED,
        'black_up': BLACK,
        'black_down': BLACK,
        'blue_up': BLUE,
        'blue_down': BLUE,
        'yellow_up': YELLOW,
        'yellow_down': YELLOW
    }
    
    # Base y position
    base_y = SCREEN_HEIGHT - BLOCK_START_Y
    
    # Draw position markers
    for scenario, pos in positions.items():
        if pos is not None and pos > 0:
            color = scenario_colors[scenario]
            y_pos = base_y + Y_OFFSETS[scenario]
            
            # Draw circle marker
            pygame.draw.circle(screen, color, (int(pos), int(y_pos)), 10)
            pygame.draw.circle(screen, WHITE, (int(pos), int(y_pos)), 6)
            
            # Draw label
            label = font.render(f"{scenario}: {pos:.1f}", True, color)
            screen.blit(label, (int(pos) - 40, int(y_pos) - 25))

def draw_target_markers(screen):
    """Draw target position markers with different y-offsets"""
    font = pygame.font.Font(None, 20)
    
    # Target positions
    targets = {
        'Position 1': (POSITION_1, GRAY),
        'Position 2': (POSITION_2, GRAY),
        'Position 3': (POSITION_3, GRAY),
        'Position 4': (POSITION_4, GRAY)
    }
    
    # Base y position
    base_y = SCREEN_HEIGHT - BLOCK_START_Y
    
    # Draw target markers for each row (only vertical lines, no triangles)
    for i, (label, (pos, color)) in enumerate(targets.items()):
        y_top = base_y - 50
        y_bottom = base_y + 150
        pygame.draw.line(screen, color, (int(pos), y_top), (int(pos), y_bottom), 2)
        # Optional: small label above the line
        

def draw_ramp_markers(screen):
    """Draw the ramp triangle like in the simulation (forward ramp)."""
    # Base y aligned with other markers
    base_y = SCREEN_HEIGHT - BLOCK_START_Y
    
    # Forward ramp vertices as in simulation coordinates, mapped to screen space
    # Physics vertices: (RAMP_LEFT_X,0), (RAMP_RIGHT_X,0), (RAMP_LEFT_X,RAMP_HEIGHT)
    # We render relative to base_y: up is negative screen y
    p1 = (int(RAMP_LEFT_X), int(base_y))
    p2 = (int(RAMP_RIGHT_X), int(base_y))
    p3 = (int(RAMP_LEFT_X), int(base_y - RAMP_HEIGHT))
    
    # Fill ramp with lighter gray and no outline
    pygame.draw.polygon(screen, (220, 220, 220), [p1, p2, p3])
    
    # Label left and right edge numbers
    font = pygame.font.Font(None, 20)
    left_label = font.render(f"{RAMP_LEFT_X}", True, BLACK)
    right_label = font.render(f"{RAMP_RIGHT_X}", True, BLACK)
    screen.blit(left_label, (int(RAMP_LEFT_X) - 14, int(base_y) + 8))
    screen.blit(right_label, (int(RAMP_RIGHT_X) - 14, int(base_y) + 8))

def draw_info_panel(screen, positions):
    """Draw information panel with results"""
    font_large = pygame.font.Font(None, 36)
    font_medium = pygame.font.Font(None, 28)
    font_small = pygame.font.Font(None, 24)
    
    # Title
    title = font_large.render("FINAL POSITIONS", True, BLACK)
    screen.blit(title, (10, 10))
    
    # Results
    results_text = font_medium.render("FINAL POSITIONS:", True, BLACK)
    screen.blit(results_text, (10, 50))
    
    result_lines = [
        f"Red up: {positions['red_up']:.1f} (target: {POSITION_1})",
        f"Red down: {positions['red_down']:.1f} (target: {POSITION_4})",
        f"Black up: {positions['black_up']:.1f} (target: {POSITION_2})",
        f"Black down: {positions['black_down']:.1f} (target: {POSITION_3})",
        f"Blue up: {positions['blue_up']:.1f} (target: {POSITION_1})",
        f"Blue down: {positions['blue_down']:.1f} (target: {POSITION_4})",
        f"Yellow up: {positions['yellow_up']:.1f} (target: {POSITION_2})",
        f"Yellow down: {positions['yellow_down']:.1f} (target: {POSITION_3})"
    ]
    
    for i, line in enumerate(result_lines):
        text = font_small.render(line, True, BLACK)
        screen.blit(text, (10, 80 + i * 25))
    
    # Legend (positions only)
    legend_text = font_medium.render("LEGEND:", True, BLACK)
    screen.blit(legend_text, (10, 300))
    
    legend_items = [
        "Circles: Final positions",
        "Vertical lines: Target positions",
    ]
    
    for i, item in enumerate(legend_items):
        text = font_small.render(item, True, BLACK)
        screen.blit(text, (10, 330 + i * 25))

def draw_grid_lines(screen):
    """Draw vertical grid lines to help visualize positions"""
    # Draw vertical lines at target positions
    target_positions = [POSITION_1, POSITION_2, POSITION_3, POSITION_4]
    base_y = SCREEN_HEIGHT - BLOCK_START_Y
    
    for pos in target_positions:
        # Draw vertical line
        pygame.draw.line(screen, (200, 200, 200), (int(pos), base_y - 50), (int(pos), base_y + 150), 1)
        
        # Draw position number
        font = pygame.font.Font(None, 16)
        text = font.render(f"{pos}", True, (100, 100, 100))
        screen.blit(text, (int(pos) - 10, base_y + 160))

def main():
    """Main visualization function"""
    global screen, clock, draw_options
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    draw_options = CustomDrawOptions(screen)
    
    print("=== FINAL POSITIONS VISUALIZATION ===")
    print("Showing optimized parameters and their final positions")
    print("Press ESC to quit")
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        screen.fill(WHITE)
        draw_grid_lines(screen)
        draw_target_markers(screen)
        draw_ramp_markers(screen)
        draw_info_panel(screen, FINAL_POSITIONS)
        draw_position_markers(screen, FINAL_POSITIONS)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    print("Visualization closed")

if __name__ == "__main__":
    main()
