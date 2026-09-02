import math
import random
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from scipy.optimize import differential_evolution
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

# Optimization parameters 
NUM_TRIALS = 1
MIN_VALID_RESULTS = 1

# Objective function weights
TARGET_WEIGHT = 1.0
DIFFERENCE_WEIGHT = 0.7
VARIANCE_WEIGHT = 0.5
CONSISTENCY_WEIGHT = 0.5  

# Convergence parameters
CONVERGENCE_WINDOW = 15
EXCELLENT_SOLUTION_THRESHOLD = 1000

# Global variables for tracking
best_error = float('inf')
best_params = None
iteration_count = 0
convergence_count = 0
last_error = float('inf')

# =========================
# SETTING UP THE SIMULATION
# =========================

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
            screen.fill((255, 255, 255))
            space.debug_draw(draw_options)
            pygame.display.flip()
            clock.tick(FPS)

    return None

def is_valid_result(position, scenario_name):
    """Check if simulation result is valid"""
    if position is None or position <= 0:
        return False
    
    # Check position bounds
    if position < 200 or position > 1200:
        return False
    
    # Check ramp side constraints
    if "up" in scenario_name:
        # Up cases should land on left side of ramp
        if position > 520:
            return False
    elif "down" in scenario_name:
        # Down cases should land on right side of ramp  
        if position < 790:
            return False
    
    return True

def test_parameter_combination(RED_FRICTION, BLACK_FRICTION, GREY_FRICTION, GREY_RAMP_FRICTION, YELLOW_RAMP_FRICTION, BLUE_RAMP_FRICTION, force_magnitude):
    """Test parameter combination with multiple trials"""
    all_results = {}
    
    # Define scenarios
    scenarios = {
        'red_up': (red, GREY_RAMP_FRICTION, "forward", "up"),
        'red_down': (red, GREY_RAMP_FRICTION, "forward", "down"),
        'black_up': (black, GREY_RAMP_FRICTION, "forward", "up"),
        'black_down': (black, GREY_RAMP_FRICTION, "forward", "down"),
        'blue_up': (grey, BLUE_RAMP_FRICTION, "forward", "up"),
        'blue_down': (grey, BLUE_RAMP_FRICTION, "forward", "down"),
        'yellow_up': (grey, YELLOW_RAMP_FRICTION, "forward", "up"),
        'yellow_down': (grey, YELLOW_RAMP_FRICTION, "forward", "down")
    }
    
    for scenario_name, (block_color, ramp_friction, ramp_direction, dynamics) in scenarios.items():
        valid_results = []
        
        for trial in range(NUM_TRIALS):
            # Deterministic setup: no randomness, seeding not required
            
            # Create block and ramp
            if block_color == grey:
                block = create_block(grey, GREY_FRICTION)
            elif block_color == red:
                block = create_block(red, RED_FRICTION)
            elif block_color == black:
                block = create_block(black, BLACK_FRICTION)
            
            ramp = create_ramp(block_color if block_color != grey else (blue if 'blue' in scenario_name else yellow), 
                             ramp_friction, ramp_direction)
            
            # Create world and run simulation
            space = create_world(block, ramp, visualization=False)
            final_x = run_simulation(space, dynamics, force_magnitude, visualization=False)
            
            if is_valid_result(final_x, scenario_name):
                valid_results.append(final_x)
        
        if len(valid_results) >= MIN_VALID_RESULTS:
            all_results[scenario_name] = np.mean(valid_results)
        else:
            all_results[scenario_name] = 0
    
    return all_results

def calculate_consistency_penalty(positions):
    """Calculate penalty for inconsistent positioning within trials"""
    consistency_error = 0
    
    # Trial A: red_up and black_up should be on same side of targets
    red_up_error = positions['red_up'] - POSITION_1
    black_up_error = positions['black_up'] - POSITION_2
    # If one is positive and other is negative, add penalty
    if (red_up_error > 0) != (black_up_error > 0):
        consistency_error += abs(red_up_error) + abs(black_up_error)
    
    # Trial A: red_down and black_down should be on same side of targets
    red_down_error = positions['red_down'] - POSITION_4
    black_down_error = positions['black_down'] - POSITION_3
    if (red_down_error > 0) != (black_down_error > 0):
        consistency_error += abs(red_down_error) + abs(black_down_error)
    
    # Trial B: blue_up and yellow_up should be on same side of targets
    blue_up_error = positions['blue_up'] - POSITION_1
    yellow_up_error = positions['yellow_up'] - POSITION_2
    if (blue_up_error > 0) != (yellow_up_error > 0):
        consistency_error += abs(blue_up_error) + abs(yellow_up_error)
    
    # Trial B: blue_down and yellow_down should be on same side of targets
    blue_down_error = positions['blue_down'] - POSITION_4
    yellow_down_error = positions['yellow_down'] - POSITION_3
    if (blue_down_error > 0) != (yellow_down_error > 0):
        consistency_error += abs(blue_down_error) + abs(yellow_down_error)
    
    return consistency_error

def objective_function(params):
    """Enhanced objective function with consistency penalties"""
    global best_error, best_params, iteration_count, convergence_count, last_error
    
    RED_FRICTION, BLACK_FRICTION, GREY_FRICTION, GREY_RAMP_FRICTION, YELLOW_RAMP_FRICTION, BLUE_RAMP_FRICTION, force_magnitude = params
    
    # Ensure all values are positive
    RED_FRICTION = max(0.01, RED_FRICTION)
    BLACK_FRICTION = max(0.01, BLACK_FRICTION)
    GREY_FRICTION = max(0.01, GREY_FRICTION)
    GREY_RAMP_FRICTION = max(0.01, GREY_RAMP_FRICTION)
    YELLOW_RAMP_FRICTION = max(0.01, YELLOW_RAMP_FRICTION)
    BLUE_RAMP_FRICTION = max(0.01, BLUE_RAMP_FRICTION)
    force_magnitude = max(100, force_magnitude)
    
    positions = test_parameter_combination(
        RED_FRICTION, BLACK_FRICTION, GREY_FRICTION, GREY_RAMP_FRICTION, YELLOW_RAMP_FRICTION, BLUE_RAMP_FRICTION, force_magnitude
    )
    
    # Check if we have valid results
    valid_positions = [pos for pos in positions.values() if pos > 0]
    if len(valid_positions) < 6:  # Need at least 6 out of 8 scenarios
        return 1000000  # High penalty for insufficient valid results
    
    # 1. TARGET POSITION ERRORS (weight: 0.7)
    target_error = 0
    target_error += (positions['red_up'] - POSITION_1) ** 2
    target_error += (positions['red_down'] - POSITION_4) ** 2
    target_error += (positions['black_up'] - POSITION_2) ** 2
    target_error += (positions['black_down'] - POSITION_3) ** 2
    target_error += (positions['blue_up'] - POSITION_1) ** 2
    target_error += (positions['blue_down'] - POSITION_4) ** 2
    target_error += (positions['yellow_up'] - POSITION_2) ** 2
    target_error += (positions['yellow_down'] - POSITION_3) ** 2
    
    # 2. RELATIVE DIFFERENCE ERRORS (weight: 1.0)
    # Trial A: (black_up - red_up) vs (red_down - black_down)
    trial_a_diff1 = positions['black_up'] - positions['red_up']
    trial_a_diff2 = positions['red_down'] - positions['black_down']
    diff_error_A = (trial_a_diff1 - trial_a_diff2) ** 2
    
    # Trial B: (yellow_up - blue_up) vs (blue_down - yellow_down)
    trial_b_diff1 = positions['yellow_up'] - positions['blue_up']
    trial_b_diff2 = positions['blue_down'] - positions['yellow_down']
    diff_error_B = (trial_b_diff1 - trial_b_diff2) ** 2
    
    diff_error = diff_error_A + diff_error_B
    
    # 3. POSITION VARIANCE ERRORS (weight: 0.3)
    # Group 1: red_up and blue_up (both should be POSITION_1 = 223)
    pos1_values = [positions['red_up'], positions['blue_up']]
    # Group 2: black_up and yellow_up (both should be POSITION_2 = 337)
    pos2_values = [positions['black_up'], positions['yellow_up']]
    # Group 3: red_down and blue_down (both should be POSITION_4 = 1087)
    pos3_values = [positions['red_down'], positions['blue_down']]
    # Group 4: black_down and yellow_down (both should be POSITION_3 = 973)
    pos4_values = [positions['black_down'], positions['yellow_down']]
    
    var_error = 0
    var_error += np.var(pos1_values)  # red_up and blue_up variance
    var_error += np.var(pos2_values)  # black_up and yellow_up variance
    var_error += np.var(pos3_values)  # red_down and blue_down variance
    var_error += np.var(pos4_values)  # black_down and yellow_down variance
    
    # 4. CONSISTENCY PENALTY 
    consistency_error = calculate_consistency_penalty(positions)
    
    # COMBINE WEIGHTED ERRORS
    total_error = (target_error * TARGET_WEIGHT + 
                   diff_error * DIFFERENCE_WEIGHT + 
                   var_error * VARIANCE_WEIGHT +
                   consistency_error * CONSISTENCY_WEIGHT)
    
    # Track best solution
    iteration_count += 1
    if total_error < best_error:
        best_error = total_error
        best_params = params.copy()
        convergence_count = 0
        print(f"NEW BEST! Iteration {iteration_count}: Error = {total_error:.0f}")
        print(f"  Target: {target_error:.0f}, Diff: {diff_error:.0f}, Var: {var_error:.0f}, Consistency: {consistency_error:.0f}")
        print(f"  Trial A diffs: {trial_a_diff1:.1f} vs {trial_a_diff2:.1f}")
        print(f"  Trial B diffs: {trial_b_diff1:.1f} vs {trial_b_diff2:.1f}")
        
        # Print consistency info
        print(f"  Consistency check:")
        print(f"    Trial A up: red={positions['red_up']:.1f} (target {POSITION_1}), black={positions['black_up']:.1f} (target {POSITION_2})")
        print(f"    Trial A down: red={positions['red_down']:.1f} (target {POSITION_4}), black={positions['black_down']:.1f} (target {POSITION_3})")
        print(f"    Trial B up: blue={positions['blue_up']:.1f} (target {POSITION_1}), yellow={positions['yellow_up']:.1f} (target {POSITION_2})")
        print(f"    Trial B down: blue={positions['blue_down']:.1f} (target {POSITION_4}), yellow={positions['yellow_down']:.1f} (target {POSITION_3})")
    else:
        convergence_count += 1
    
    # Check for convergence
    if abs(total_error - last_error) < 1.0:  # Very small change
        convergence_count += 1
    else:
        convergence_count = 0
    
    last_error = total_error
    
    # Early termination if we have a very good solution
    if total_error < 1000:  # Very good solution
        print(f"EXCELLENT SOLUTION FOUND! Error = {total_error:.0f}")
        print(f"Parameters: R={RED_FRICTION:.3f} B={BLACK_FRICTION:.3f} G={GREY_FRICTION:.3f} GR={GREY_RAMP_FRICTION:.3f} YR={YELLOW_RAMP_FRICTION:.3f} BR={BLUE_RAMP_FRICTION:.3f} F={force_magnitude:.0f}")
        return total_error
    
    # Early termination if converged
    if convergence_count > 15:  # No improvement for 15 iterations
        print(f"CONVERGED! Best error = {best_error:.0f}")
        return total_error
    
    # Print progress every 20 iterations
    if iteration_count % 20 == 0:
        print(f"Iteration {iteration_count}: Error = {total_error:.0f} (Best: {best_error:.0f})")
        print(f"  Target: {target_error:.0f}, Diff: {diff_error:.0f}, Var: {var_error:.0f}, Consistency: {consistency_error:.0f}")
    
    return total_error

def find_optimal_parameters():
    """Find optimal parameters using differential evolution with early stopping"""
    global best_error, best_params, iteration_count, convergence_count, last_error
    
    print("Starting ENHANCED parameter optimization with CONSISTENCY penalties...")
    print("FEATURES:")
    print(f"  - Target position errors (weight: {TARGET_WEIGHT})")
    print(f"  - Relative difference errors (weight: {DIFFERENCE_WEIGHT})")
    print(f"    * Trial A: (black_up - red_up) vs (red_down - black_down)")
    print(f"    * Trial B: (yellow_up - blue_up) vs (blue_down - yellow_down)")
    print(f"  - Position variance errors (weight: {VARIANCE_WEIGHT})")
    print(f"    * Group 1: red_up and blue_up")
    print(f"    * Group 2: black_up and yellow_up")
    print(f"    * Group 3: red_down and blue_down")
    print(f"    * Group 4: black_down and yellow_down")
    print(f"  - CONSISTENCY penalties (weight: {CONSISTENCY_WEIGHT})")
    print(f"    * Trial A: red_up and black_up should be on same side of targets")
    print(f"    * Trial A: red_down and black_down should be on same side of targets")
    print(f"    * Trial B: blue_up and yellow_up should be on same side of targets")
    print(f"    * Trial B: blue_down and yellow_down should be on same side of targets")
    print()
    
    # Parameter bounds
    bounds = [
        (0.01, 1.0),    # RED_FRICTION
        (0.01, 1.0),    # BLACK_FRICTION
        (0.01, 1.0),    # GREY_FRICTION
        (0.01, 1.0),    # GREY_RAMP_FRICTION
        (0.01, 1.0),    # YELLOW_RAMP_FRICTION
        (0.01, 1.0),    # BLUE_RAMP_FRICTION
        (400, 1000)     # force_magnitude
    ]
    
    # Callback function for tracking progress
    def callback(xk, convergence):
        global best_error, convergence_count
        current_error = objective_function(xk)
        if current_error < best_error:
            best_error = current_error
            convergence_count = 0
        else:
            convergence_count += 1
        
        # Early termination if converged
        if convergence_count > CONVERGENCE_WINDOW:
            return True
        
        # Early termination if excellent solution found
        if best_error < EXCELLENT_SOLUTION_THRESHOLD:
            return True
        
        return False
    
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            seed=1,
            maxiter=1000,
            popsize=15,
            callback=callback,
            atol=1e-6,
            tol=1e-6
        )
        
        print(f"\nOptimization completed!")
        print(f"Final error: {result.fun:.0f}")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        
        return result
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None

def main():
    """Main function"""
    global screen, clock, draw_options
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    draw_options = CustomDrawOptions(screen)
    
    print("=== ENHANCED GROUND TRUTH PARAMETER FINDING ===")
    print("With CONSISTENCY penalties for same-trial elements")
    print()
    
    # Find optimal parameters
    result = find_optimal_parameters()
    
    if result is not None:
        print(f"\n=== OPTIMAL PARAMETERS FOUND ===")
        print(f"Red friction: {result.x[0]:.4f}")
        print(f"Black friction: {result.x[1]:.4f}")
        print(f"Grey friction: {result.x[2]:.4f}")
        print(f"Grey ramp friction: {result.x[3]:.4f}")
        print(f"Yellow ramp friction: {result.x[4]:.4f}")
        print(f"Blue ramp friction: {result.x[5]:.4f}")
        print(f"Force magnitude: {result.x[6]:.1f}")
        print(f"Final error: {result.fun:.0f}")
        
        # Test final parameters
        print(f"\n=== TESTING FINAL PARAMETERS ===")
        final_positions = test_parameter_combination(
            result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], result.x[5], result.x[6]
        )
        
        print(f"Final positions:")
        for scenario, pos in final_positions.items():
            print(f"  {scenario}: {pos:.1f}")
        
        # Calculate consistency
        consistency_error = calculate_consistency_penalty(final_positions)
        print(f"\nConsistency error: {consistency_error:.0f}")
        
        # Print individual differences
        print(f"\nINDIVIDUAL DIFFERENCES:")
        print(f"blue_up - red_up: {final_positions['blue_up']:.1f} - {final_positions['red_up']:.1f} = {final_positions['blue_up'] - final_positions['red_up']:.1f}")
        print(f"red_down - blue_down: {final_positions['red_down']:.1f} - {final_positions['blue_down']:.1f} = {final_positions['red_down'] - final_positions['blue_down']:.1f}")
        print(f"yellow_up - blue_up: {final_positions['yellow_up']:.1f} - {final_positions['blue_up']:.1f} = {final_positions['yellow_up'] - final_positions['blue_up']:.1f}")
        print(f"blue_down - yellow_down: {final_positions['blue_down']:.1f} - {final_positions['yellow_down']:.1f} = {final_positions['blue_down'] - final_positions['yellow_down']:.1f}")
        
    else:
        print("Failed to find optimal parameters!")
    
    pygame.quit()

if __name__ == "__main__":
    main()
