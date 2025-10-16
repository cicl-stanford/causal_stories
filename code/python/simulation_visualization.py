from step1_physics_simulations import (
    create_block, 
    create_ramp, 
    create_world, 
    SCREEN_WIDTH, 
    SCREEN_HEIGHT, 
    FPS,  
    TIME_STEP, 
    MAX_SIMULATION_FRAMES , 
    RAMP_ANGLE, 
    VELOCITY_THRESHOLD
)
from conditions import (
    grey,
    white,
    black
)
import math
import pygame
import pymunk
import pymunk.pygame_util

# Default values for parameters
DEFAULT_BLOCK_FRICTION = 0.4
DEFAULT_RAMP_FRICTION = 0.4
DEFAULT_BLOCK_NOISE_SD = 0.05
DEFAULT_RAMP_NOISE_SD = 0.05
DEFAULT_FORCE_MAGNITUDE = 590

# Dynamics options
DYNAMICS_OPTIONS = [
    "forward_ramp_up", 
    "forward_ramp_down", 
    "backward_ramp_up", 
    "backward_ramp_down"
]

# Set up Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

class CustomDrawOptions(pymunk.pygame_util.DrawOptions):
    """Custom draw options that flip Y coordinates for proper pygame rendering."""
    
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

draw_options = CustomDrawOptions(screen)

def _apply_dynamics_force(block, dynamics, force_magnitude):
    """Apply force to block based on dynamics type."""
    if dynamics == "forward_ramp_up":
        ramp_angle = math.radians(RAMP_ANGLE)
        force_vector = pymunk.Vec2d(
            force_magnitude * math.cos(ramp_angle), 
            force_magnitude * math.sin(ramp_angle)
        )
        block.apply_impulse_at_local_point(force_vector)
    elif dynamics == "forward_ramp_down":
        print("No force applied - block will move due to gravity only")
    elif dynamics == "backward_ramp_up":
        ramp_angle = math.radians(180 - RAMP_ANGLE)
        force_vector = pymunk.Vec2d(
            force_magnitude * math.cos(ramp_angle), 
            force_magnitude * math.sin(ramp_angle)
        )
        block.apply_impulse_at_local_point(force_vector)
    elif dynamics == "backward_ramp_down":
        print("No force applied - block will move due to gravity only")
    else:
        ramp_angle = math.radians(RAMP_ANGLE)
        force_vector = pymunk.Vec2d(
            force_magnitude * math.cos(ramp_angle), 
            force_magnitude * math.sin(ramp_angle)
        )
        block.apply_impulse_at_local_point(force_vector)

def run_simulation(space, dynamics, force_magnitude, visualization=True):
    """Run the physics simulation with given dynamics and force."""

    block = [body for body in space.bodies if body.body_type != pymunk.Body.STATIC][0]
    
    # Show starting scene first
    if visualization:
        show_starting_scene(space, block, dynamics)
    
    # Don't apply force immediately - let user see starting position
    if visualization:
        print("Press SPACE to start simulation, or ESC to quit...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return None
            clock.tick(60)
    
    # Now apply force and run simulation
    _apply_dynamics_force(block, dynamics, force_magnitude)
    
    # Run simulation loop
    for frame in range(MAX_SIMULATION_FRAMES):
        space.step(TIME_STEP)
        
        # Check if block has stopped moving
        if block.velocity.length < VELOCITY_THRESHOLD:
            return block.position.x
            
        if visualization and handle_visualization(space, block, dynamics):
            return None  
    
    return None 

def show_starting_scene(space, block, dynamics):
    """Show the starting scene with starting positions."""
    from step1_physics_simulations import BLOCK_START_X, BLOCK_BACKWARD_START_X, RAMP_LEFT_X, RAMP_RIGHT_X, BLOCK_START_Y
    
    screen.fill(white)
    space.debug_draw(draw_options)
    
    # Determine starting and target positions
    if dynamics.startswith("forward"):
        start_x = BLOCK_START_X
        target_x = RAMP_LEFT_X
        direction = "left"
    else:
        start_x = BLOCK_BACKWARD_START_X
        target_x = RAMP_RIGHT_X
        direction = "right"
    
    # Add minimal text overlay
    font = pygame.font.Font(None, 36)
    instruction_text = font.render("Press SPACE to start, ESC to quit", True, black[:3])
    screen.blit(instruction_text, (10, 10))
    
    pygame.display.flip()

def handle_visualization(space, block, dynamics):
    """Handle pygame visualization and events."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True
    
    screen.fill(white)
    space.debug_draw(draw_options)
    
    # Show final position
    font = pygame.font.Font(None, 36)
    position_text = font.render(f"Position: {block.position.x:.2f}", True, black[:3])
    screen.blit(position_text, (10, 10))
    
    pygame.display.flip()
    clock.tick(FPS)
    
    return False

def get_user_input(prompt, default_value, validation_func=None):
    """get user input to set up the simulation"""
    user_input = input(f"{prompt} (default: {default_value}): ").strip()
    
    if user_input == "":
        print(f"Using default: {default_value}")
        return default_value
    
    if validation_func:
        validated_value = validation_func(user_input, default_value)
        if validated_value is not None:
            print(f"Set to: {validated_value}")
            return validated_value
        else:
            print(f"Invalid input. Using default: {default_value}")
            return default_value
    
    return user_input

def validate_float(value_str, default_value, min_value=0):
    """Validate float input with minimum value check."""
    try:
        value = float(value_str)
        if value < min_value:
            print(f"Value cannot be less than {min_value}.")
            return None
        return value
    except ValueError:
        return None

def run_trials(dynamics, BLOCK_FRICTION, RAMP_FRICTION, block_noise_sd, ramp_noise_sd, force_magnitude, n_trials, show_visualization=True):
    """Run multiple trials with the same parameters and record final positions."""
    results = []
    direction = "forward" if dynamics.startswith("forward") else "backward"
    
    for trial in range(n_trials):
        block = create_block(grey, BLOCK_FRICTION, block_noise_sd)
        ramp = create_ramp(grey, RAMP_FRICTION, ramp_noise_sd, direction)
        space = create_world(block, ramp, visualization=False)
        
        # Run simulation 
        show_viz = show_visualization
        final_x = run_simulation(space, dynamics, force_magnitude, visualization=show_viz)
        
        results.append(final_x)
        print(f"Trial {trial+1}: {final_x:.2f}")
        
    return results

def main():
    print("=== Causal Abstraction Physics Simulation ===")
    print("This simulation runs multiple trials with the same parameters.")
    print()

    # Get dynamics choice
    dynamics_options = DYNAMICS_OPTIONS
    print("Dynamics options:")
    print("  1. forward_ramp, up")
    print("  2. forward_ramp, down") 
    print("  3. backward_ramp, up")
    print("  4. backward_ramp, down")
    
    while True:
        choice = input("\nSelect dynamics (1-4): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(dynamics_options):
                dynamics = dynamics_options[idx]
                print(f"Selected dynamics: {dynamics}")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("Invalid input. Please enter 1, 2, 3, or 4.")
    
    BLOCK_FRICTION = get_user_input(
        "Enter block friction (press enter for default)", 
        DEFAULT_BLOCK_FRICTION, 
        lambda x, d: validate_float(x, d, 0)
    )
    
    RAMP_FRICTION = get_user_input(
        "Enter ramp friction (press enter for default)", 
        DEFAULT_RAMP_FRICTION, 
        lambda x, d: validate_float(x, d, 0)
    )
    
    block_noise_sd = get_user_input(
        "Enter block noise SD (press enter for default)", 
        DEFAULT_BLOCK_NOISE_SD, 
        lambda x, d: validate_float(x, d, 0)
    )
    
    ramp_noise_sd = get_user_input(
        "Enter ramp noise SD (press enter for default)", 
        DEFAULT_RAMP_NOISE_SD, 
        lambda x, d: validate_float(x, d, 0)
    )
    
    if not dynamics.endswith("down"):
        force_magnitude = get_user_input(
            "Enter force magnitude (press enter for default)", 
            DEFAULT_FORCE_MAGNITUDE, 
            lambda x, d: validate_float(x, d, 0)
        )
    else:
        force_magnitude = 0
        print("Force magnitude: Not applicable for 'down' dynamics (no force applied)")
    
    n_trials = get_user_input(
        "Enter number of trials to run (press enter for 1)", 
        1, 
        lambda x, d: validate_float(x, d, 1)
    )
    n_trials = int(n_trials)
    
    viz_choice = input("Show visualization? (y/n, press enter for y): ").strip().lower()
    show_visualization = viz_choice != 'n'
    
    print(f"\nRunning {n_trials} trials with {dynamics}...")
    
    # Run trials
    results = run_trials(
        dynamics, BLOCK_FRICTION, RAMP_FRICTION, block_noise_sd, ramp_noise_sd, 
        force_magnitude, n_trials, show_visualization
    )
    
    # Show final results
    if results:
        print(f"\nFinal positions: {[f'{pos:.2f}' for pos in results]}")
    else:
        print("\nNo successful trials!")
    
    pygame.quit()

if __name__ == "__main__":
    main() 