"""Physics simulations.

Keeps drawing until each hypothesis has N_TRIALS valid landings (physics/agent
share the down sim; ramp uses the up sim). Opposite-side outliers are rejected. 
On-ramp stalls are kept. If the cube does not move, record the start x. 
If the sim hits the frame cap, record x at timeout.
"""
import os

from run_paths import DATA_DIR, print_progress

from conditions import (
    MID_POINT_RAMP,
    scenarios,
    trial_configs,
    black, 
    grey,
    white,
    BLOCK_FRICTION_RED,
    BLOCK_FRICTION_BLACK,
    RAMP_FRICTION_YELLOW,
    RAMP_FRICTION_BLUE,
)
from collections import defaultdict
from datetime import datetime
from itertools import product
import math
import hashlib
import multiprocessing as mp
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import random
import time
import pandas as pd

# =========================
# CONFIGURATION CONSTANTS
# =========================

# Random seed for reproducibility
DEFAULT_SEED = 1

# Number of simulations
N_TRIALS = 2000

N_PROCESSES = 6

# Simuation settings 
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
GRAVITY = (0, -981)
TIME_STEP = 1 / 50.0
FPS = 50
VELOCITY_THRESHOLD = 0.01
FORCE_MAGNITUDE = 557.4
MAX_SIMULATION_FRAMES = 750
SHOW_VISUALIZATION = False
PYGAME_INITIALIZED = False
SCREEN = None
CLOCK = None
DRAW_OPTIONS = None

# Block properties
BLOCK_SIZE = 35
BLOCK_MASS = 1
BLOCK_START_X = 645 
BLOCK_START_Y = 251
BLOCK_BACKWARD_START_X = 665 

# Ramp properties
RAMP_LEFT_X = 520
RAMP_RIGHT_X = 790
RAMP_POSITION_X = 0
RAMP_POSITION_Y = 150
RAMP_HEIGHT = 150
RAMP_ANGLE = 29.05

# Ground properties
GROUND_POSITION_X = 450
GROUND_POSITION_Y = 150
GROUND_FRICTION = 0.5
GROUND_LENGTH = 5000
GROUND_WIDTH = 10

# Finish line properties
FINISH_LINE_POSITION_X = 1125
FINISH_LINE_POSITION_Y = 150
FINISH_LINE_LENGTH = 18
FINISH_LINE_WIDTH = 9

# Parameter ranges for grid search
BLOCK_NOISE_SD_MIN = 0.05
BLOCK_NOISE_SD_MAX = 0.35
BLOCK_NOISE_SD_STEPS = 16

RAMP_NOISE_SD_MIN = 0.5
RAMP_NOISE_SD_MAX = 2
RAMP_NOISE_SD_STEPS = 16

# =========================
#  SETUP FUNCTIONS
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

def initialize_pygame_visualization():
    """Initialize pygame graphics system for physics simulation visualization."""
    global PYGAME_INITIALIZED, SCREEN, CLOCK, DRAW_OPTIONS
    if not PYGAME_INITIALIZED:
        pygame.init()
        SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Physics Simulation Visualization")
        CLOCK = pygame.time.Clock()
        DRAW_OPTIONS = CustomDrawOptions(SCREEN)
        PYGAME_INITIALIZED = True

def set_reproducibility_seeds(seed=DEFAULT_SEED):
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)


def stable_seed_int(*parts):
    """Deterministic 32-bit seed from parts (stable across machines/Python runs)."""
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.md5(payload.encode("ascii")).hexdigest()
    return int(digest[:8], 16)


def task_seed(base_seed, trial_type, noise_sd):
    """Unique seed for one parallel (trial_type, noise_sd) task."""
    return stable_seed_int(base_seed, "task", trial_type, round(float(noise_sd), 6))


def trial_config_seed(task_seed_value, trial_key):
    """Unique seed for one trial config within a parallel task."""
    return stable_seed_int(task_seed_value, "config", trial_key)

def create_block(color, friction, sd):
    block_noise = random.gauss(0, sd)
    # This is to make sure the block noise is not negative
    while block_noise < -friction:
        block_noise = random.gauss(0, sd)
    mass = BLOCK_MASS
    moment = pymunk.moment_for_box(mass, (BLOCK_SIZE, BLOCK_SIZE))
    body = pymunk.Body(mass, moment)
    body.angle = math.radians(RAMP_ANGLE)
    shape = pymunk.Poly.create_box(body, (BLOCK_SIZE, BLOCK_SIZE))
    shape.friction = friction + block_noise
    shape.color = color
    body.position = (BLOCK_START_X, BLOCK_START_Y)
    return body, shape, block_noise

def create_ramp(color, friction, sd, direction):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    if direction == "forward":
        body.vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_LEFT_X, RAMP_HEIGHT)]
    else:
        body.vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_RIGHT_X, RAMP_HEIGHT)]
    ramp_noise = random.gauss(0, sd)
    while ramp_noise < -friction:
        ramp_noise = random.gauss(0, sd)
    body.position = (0, RAMP_POSITION_Y)
    shape = pymunk.Poly(body, body.vertices)
    shape.friction = friction + ramp_noise
    shape.color = color
    return body, shape, ramp_noise

def create_world(block, ramp, visualization=True):
    space = pymunk.Space()
    space.gravity = GRAVITY
    forward_ramp_vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_LEFT_X, RAMP_HEIGHT)]
    backward_ramp_vertices = [(RAMP_LEFT_X, 0), (RAMP_RIGHT_X, 0), (RAMP_RIGHT_X, RAMP_HEIGHT)]
    
    if ramp[0].vertices == forward_ramp_vertices:
        block[0].angle = math.radians(180 - RAMP_ANGLE)
    elif ramp[0].vertices == backward_ramp_vertices:
        block[0].angle = math.radians(180 + RAMP_ANGLE)
        block[0].position = (BLOCK_BACKWARD_START_X, BLOCK_START_Y)
    
    space.add(ramp[0], ramp[1])
    space.add(block[0], block[1])
    
    # Create ground
    ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    ground_body.position = (GROUND_POSITION_X, GROUND_POSITION_Y)
    ground_shape = pymunk.Poly.create_box(ground_body, (GROUND_LENGTH, GROUND_WIDTH))
    ground_shape.friction = GROUND_FRICTION
    ground_shape.color = grey
    space.add(ground_body, ground_shape)
    
    # Create Finish line
    finish_line_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    finish_line_body.position = (FINISH_LINE_POSITION_X, FINISH_LINE_POSITION_Y)
    finish_line_shape = pymunk.Poly.create_box(finish_line_body, (FINISH_LINE_LENGTH, FINISH_LINE_WIDTH))
    finish_line_shape.friction = GROUND_FRICTION
    finish_line_shape.color = black
    space.add(finish_line_body, finish_line_shape)
    
    if visualization:
        initialize_pygame_visualization()  
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            SCREEN.fill(white)
            space.debug_draw(DRAW_OPTIONS)
            pygame.display.flip()
            CLOCK.tick(FPS)
    return space

# =========================
# SIMULATION FUNCTIONS
# =========================
def run_simulation(space, dynamics, visualization=False):
    """Run simulation, return (final_x, stop_reason).

    stop_reason is 'settled' when speed drops below VELOCITY_THRESHOLD,
    otherwise 'timeout'. On timeout, record the cube's x at the last frame
    (not 0). If it never moves, that x is the start position.
    """
    block = [body for body in space.bodies if body.body_type != pymunk.Body.STATIC][0]
    
    if dynamics == "up":
        ramp_angle = math.radians(RAMP_ANGLE)
        force_vector = pymunk.Vec2d(
            FORCE_MAGNITUDE * math.cos(ramp_angle), 
            FORCE_MAGNITUDE * math.sin(ramp_angle)
        )
        block.apply_impulse_at_local_point(force_vector)
    elif dynamics == "up-backward ramp":
        # The up backward ramp motion is only used in simulation_visualization.py
        ramp_angle = math.radians(180 - RAMP_ANGLE)
        force_vector = pymunk.Vec2d(
            FORCE_MAGNITUDE * math.cos(ramp_angle), 
            FORCE_MAGNITUDE * math.sin(ramp_angle)
        )
        block.apply_impulse_at_local_point(force_vector)

    for frame in range(MAX_SIMULATION_FRAMES):
        space.step(TIME_STEP)
        if block is None:
            return 0, "timeout"
        if block.velocity.length < VELOCITY_THRESHOLD:
            return block.position.x, "settled"
        if visualization:
            initialize_pygame_visualization()  
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return block.position.x, "timeout"
            SCREEN.fill(white)
            space.debug_draw(DRAW_OPTIONS)
            pygame.display.flip()
            CLOCK.tick(FPS)
    return block.position.x, "timeout"

def _trial_letter(trial_name):
    if trial_name in ("a", "b", "c", "d"):
        return trial_name
    if isinstance(trial_name, str) and trial_name.startswith("trial_") and len(trial_name) > 6:
        return trial_name[6]
    return str(trial_name)


def position_allowed_for_hypothesis(trial, model_type, final_x_position):
    """Drop only opposite-side outliers; keep on-ramp stalls.

    Forward (A, B): physics/agent may not exit left of the ramp (x > 520);
    ramp may not exit right (x < 790). Backward (C, D) is mirrored.
    """
    if final_x_position == 0:
        return False
    trial = _trial_letter(trial)
    if trial in ("a", "b"):
        if model_type in ("agent", "physics"):
            return final_x_position > RAMP_LEFT_X
        if model_type == "ramp":
            return final_x_position < RAMP_RIGHT_X
    if trial in ("c", "d"):
        if model_type == "physics":
            return final_x_position < RAMP_RIGHT_X
        if model_type in ("agent", "ramp"):
            return final_x_position > RAMP_LEFT_X
    return True


def is_valid_trial_position(trial_name, model_type, final_x_position):
    return position_allowed_for_hypothesis(trial_name, model_type, final_x_position)

def _create_trial_record(trial_letter, element_name, model, block_noise_sd, ramp_noise_sd, 
                        block_noise_down=None, ramp_noise_down=None, 
                        block_noise_up=None, ramp_noise_up=None, 
                        final_position=None, stop_reason=None, would_keep=None):
    """Create a trial record with consistent column ordering (final_position last)."""
    record = {
        'trial': trial_letter,
        'element': element_name,
        'model': model,
        'block_noise_sd': block_noise_sd,
        'ramp_noise_sd': ramp_noise_sd,
        # Always include all noise fields with default values
        'block_noise_down': block_noise_down if block_noise_down is not None else 0,
        'ramp_noise_down': ramp_noise_down if ramp_noise_down is not None else 0,
        'block_noise_up': block_noise_up if block_noise_up is not None else 0,
        'ramp_noise_up': ramp_noise_up if ramp_noise_up is not None else 0,
    }
    if stop_reason is not None:
        record['stop_reason'] = stop_reason
    if would_keep is not None:
        record['would_keep'] = would_keep
    record['final_position'] = final_position if final_position is not None else 0
    return record

def run_single_trial(trial_config, block_noise_sd, ramp_noise_sd, n_trials=N_TRIALS, seed=DEFAULT_SEED):
    """Keep drawing until each hypothesis has n_trials valid landings."""
    trial_name = trial_config["name"]
    trial_data = []
    failed_data = []
    failed_runs = 0
    run_index = 1
    trial_letter = trial_name[6]
    element_name = trial_name[8:]
    n_down_kept = 0
    n_ramp_kept = 0

    max_attempts = n_trials * 10
    while (n_down_kept < n_trials or n_ramp_kept < n_trials) and run_index <= max_attempts:
        block_noise_down = ramp_noise_down = block_noise_up = ramp_noise_up = None
        final_x_position_down = final_x_position_up = 0
        down_reason = up_reason = "timeout"

        trial_block_noise_sd = block_noise_sd if trial_config["override_block_sd"] else 0
        trial_ramp_noise_sd = ramp_noise_sd if trial_config["override_ramp_sd"] else 0

        try:
            set_reproducibility_seeds(seed + run_index)

            if n_down_kept < n_trials:
                block_down, block_shape_down, block_noise_down = create_block(
                    trial_config["block_color"], trial_config["block_friction"], trial_block_noise_sd
                )
                ramp_down, ramp_shape_down, ramp_noise_down = create_ramp(
                    trial_config["ramp_color"], trial_config["ramp_friction"], trial_ramp_noise_sd, trial_config["ramp_direction"]
                )
                space_down = create_world((block_down, block_shape_down), (ramp_down, ramp_shape_down), visualization=False)
                final_x_position_down, down_reason = run_simulation(space_down, "down", visualization=SHOW_VISUALIZATION)
                agent_valid = is_valid_trial_position(trial_name, "agent", final_x_position_down)
                physics_valid = is_valid_trial_position(trial_name, "physics", final_x_position_down)
                if agent_valid and physics_valid:
                    trial_data.append(_create_trial_record(
                        trial_letter, element_name, 'agent', block_noise_sd, ramp_noise_sd,
                        block_noise_down=block_noise_down, ramp_noise_down=ramp_noise_down,
                        stop_reason=down_reason, would_keep=True,
                        final_position=final_x_position_down
                    ))
                    trial_data.append(_create_trial_record(
                        trial_letter, element_name, 'physics', block_noise_sd, ramp_noise_sd,
                        block_noise_down=block_noise_down, ramp_noise_down=ramp_noise_down,
                        stop_reason=down_reason, would_keep=True,
                        final_position=final_x_position_down
                    ))
                    n_down_kept += 1
                else:
                    failed_runs += 1
                    failed_data.append({
                        'trial': trial_letter,
                        'element': element_name,
                        'model': 'down',
                        'block_noise_sd': block_noise_sd,
                        'ramp_noise_sd': ramp_noise_sd,
                        'block_noise_down': block_noise_down if block_noise_down is not None else 0,
                        'ramp_noise_down': ramp_noise_down if ramp_noise_down is not None else 0,
                        'final_position_down': final_x_position_down,
                        'agent_valid': agent_valid,
                        'physics_valid': physics_valid,
                        'down_stop_reason': down_reason,
                    })

            if n_ramp_kept < n_trials:
                block_up, block_shape_up, block_noise_up = create_block(
                    trial_config["block_color"], trial_config["block_friction"], trial_block_noise_sd
                )
                ramp_up, ramp_shape_up, ramp_noise_up = create_ramp(
                    trial_config["ramp_color"], trial_config["ramp_friction"], trial_ramp_noise_sd, trial_config["ramp_direction"]
                )
                space_up = create_world((block_up, block_shape_up), (ramp_up, ramp_shape_up), visualization=False)
                final_x_position_up, up_reason = run_simulation(space_up, "up", visualization=SHOW_VISUALIZATION)
                ramp_valid = is_valid_trial_position(trial_name, "ramp", final_x_position_up)
                if ramp_valid:
                    trial_data.append(_create_trial_record(
                        trial_letter, element_name, 'ramp', block_noise_sd, ramp_noise_sd,
                        block_noise_up=block_noise_up, ramp_noise_up=ramp_noise_up,
                        stop_reason=up_reason, would_keep=True,
                        final_position=final_x_position_up
                    ))
                    n_ramp_kept += 1
                else:
                    failed_runs += 1
                    failed_data.append({
                        'trial': trial_letter,
                        'element': element_name,
                        'model': 'ramp',
                        'block_noise_sd': block_noise_sd,
                        'ramp_noise_sd': ramp_noise_sd,
                        'block_noise_up': block_noise_up if block_noise_up is not None else 0,
                        'ramp_noise_up': ramp_noise_up if ramp_noise_up is not None else 0,
                        'final_position_up': final_x_position_up,
                        'ramp_valid': False,
                        'up_stop_reason': up_reason,
                    })

        except Exception as e:
            print(f"Simulation failed: {e}")
            failed_runs += 1
            failed_data.append({
                'trial': trial_letter if 'trial_letter' in locals() else 'unknown',
                'element': element_name if 'element_name' in locals() else 'unknown',
                'block_noise_sd': block_noise_sd,
                'ramp_noise_sd': ramp_noise_sd,
                'failure_reason': f'simulation_error: {str(e)}'
            })

        run_index += 1

    if n_down_kept < n_trials or n_ramp_kept < n_trials:
        print(
            f"WARNING: Only completed down={n_down_kept}/{n_trials} "
            f"ramp={n_ramp_kept}/{n_trials} after {max_attempts} attempts for {trial_name}"
        )

    # Add Trial C and D records using symmetry calculation
    if n_down_kept > 0 or n_ramp_kept > 0:
        # Get the physics records (from down simulation) and ramp records (from up simulation)
        physics_records = [record for record in trial_data if record['trial'] == trial_letter and record['model'] == 'physics']
        ramp_records = [record for record in trial_data if record['trial'] == trial_letter and record['model'] == 'ramp']
        
        # Debug: Check if records exist
        if not physics_records:
            print(f"WARNING: No physics records found for trial {trial_letter}")
        if not ramp_records:
            print(f"WARNING: No ramp records found for trial {trial_letter}")
        
        # Calculate symmetric positions
        symmetric_physics_positions = [2 * MID_POINT_RAMP - record['final_position'] for record in physics_records]
        symmetric_ramp_positions = [2 * MID_POINT_RAMP - record['final_position'] for record in ramp_records]
        
        # Create Trial C records 
        if trial_letter == 'a': 
            #Symmetric of physics model for trial A → physics records for trial C
            for i, pos in enumerate(symmetric_physics_positions):
                # Use the same noise values as the original physics record (only has down values)
                original_record = physics_records[i]
                
                trial_data.append(_create_trial_record(
                    'c', element_name, 'physics', block_noise_sd, ramp_noise_sd,
                    block_noise_down=original_record['block_noise_down'],
                    ramp_noise_down=original_record['ramp_noise_down'],
                    stop_reason=original_record.get('stop_reason'),
                    would_keep=position_allowed_for_hypothesis('c', 'physics', pos),
                    final_position=pos
                ))
            
            # Symmetric of ramp model for trial A → ramp and agent records for trial C
            for i, pos in enumerate(symmetric_ramp_positions):
                # Use the same noise values as the original ramp record (only has up values)
                original_record = ramp_records[i]
                
                # Safety check for missing fields
                block_noise_up = original_record.get('block_noise_up', 0)
                ramp_noise_up = original_record.get('ramp_noise_up', 0)
                
                trial_data.append(_create_trial_record(
                    'c', element_name, 'ramp', block_noise_sd, ramp_noise_sd,
                    block_noise_up=block_noise_up, ramp_noise_up=ramp_noise_up,
                    would_keep=position_allowed_for_hypothesis('c', 'ramp', pos),
                    final_position=pos
                ))
                trial_data.append(_create_trial_record(
                    'c', element_name, 'agent', block_noise_sd, ramp_noise_sd,
                    block_noise_up=block_noise_up, ramp_noise_up=ramp_noise_up,
                    would_keep=position_allowed_for_hypothesis('c', 'agent', pos),
                    final_position=pos
                ))
        
        # Create Trial D records 
        if trial_letter == 'b':  
            # Symmetric of physics model → physics record for trial D
            for i, pos in enumerate(symmetric_physics_positions):
                # Use the same noise values as the original physics record (only has down values)
                original_record = physics_records[i]
                
                trial_data.append(_create_trial_record(
                    'd', element_name, 'physics', block_noise_sd, ramp_noise_sd,
                    block_noise_down=original_record['block_noise_down'],
                    ramp_noise_down=original_record['ramp_noise_down'],
                    would_keep=position_allowed_for_hypothesis('d', 'physics', pos),
                    final_position=pos
                ))
            
            # Symmetric of ramp model → ramp and agent records for trial D
            for i, pos in enumerate(symmetric_ramp_positions):
                original_record = ramp_records[i]
                block_noise_up = original_record.get('block_noise_up', 0)
                ramp_noise_up = original_record.get('ramp_noise_up', 0)
                
                trial_data.append(_create_trial_record(
                    'd', element_name, 'ramp', block_noise_sd, ramp_noise_sd,
                    block_noise_up=block_noise_up, ramp_noise_up=ramp_noise_up,
                    would_keep=position_allowed_for_hypothesis('d', 'ramp', pos),
                    final_position=pos
                ))
                trial_data.append(_create_trial_record(
                    'd', element_name, 'agent', block_noise_sd, ramp_noise_sd,
                    block_noise_up=block_noise_up, ramp_noise_up=ramp_noise_up,
                    would_keep=position_allowed_for_hypothesis('d', 'agent', pos),
                    final_position=pos
                ))
    
    return trial_data, failed_data

def run_trials_parallel(args):
    """Helper function for parallel trial execution (one trial_key per task)."""
    trial_type, trial_key, noise_sd, n_trials, seed = args
    noise_sd = round(noise_sd, 3)
    trial_config = trial_configs[trial_type][trial_key]
    trial_config_with_name = {**trial_config, "name": trial_key}
    config_seed = trial_config_seed(seed, trial_key)

    if trial_type == "a":
        trial_data, failed_data = run_single_trial(
            trial_config_with_name, noise_sd, 0.0, n_trials, config_seed
        )
    else:
        trial_data, failed_data = run_single_trial(
            trial_config_with_name, 0.0, noise_sd, n_trials, config_seed
        )

    return trial_type, noise_sd, trial_data, failed_data

def _create_parameter_ranges(block_min=None, block_max=None, block_steps=None,
                            ramp_min=None, ramp_max=None, ramp_steps=None):
    """Create parameter ranges using provided values or defaults."""
    block_range = np.linspace(
        block_min or BLOCK_NOISE_SD_MIN,
        block_max or BLOCK_NOISE_SD_MAX, 
        block_steps or BLOCK_NOISE_SD_STEPS
    )
    ramp_range = np.linspace(
        ramp_min or RAMP_NOISE_SD_MIN,
        ramp_max or RAMP_NOISE_SD_MAX,
        ramp_steps or RAMP_NOISE_SD_STEPS
    )
    return block_range, ramp_range

def run_parallel_simulations(block_range, ramp_range, n_trials, seed):
    """Run all unique simulations in parallel and return cached results."""
    trial_args = []
    for block_sd in block_range:
        task_s = task_seed(seed, "a", block_sd)
        for trial_key in trial_configs["a"]:
            trial_args.append(("a", trial_key, block_sd, n_trials, task_s))
    for ramp_sd in ramp_range:
        task_s = task_seed(seed, "b", ramp_sd)
        for trial_key in trial_configs["b"]:
            trial_args.append(("b", trial_key, ramp_sd, n_trials, task_s))

    n_proc = min(N_PROCESSES, len(trial_args)) or 1
    print(f"Running {len(trial_args)} simulation tasks using {n_proc} processes")
    start = time.time()
    all_results = []
    with mp.Pool(processes=n_proc) as pool:
        for i, result in enumerate(
            pool.imap_unordered(run_trials_parallel, trial_args, chunksize=1), 1
        ):
            all_results.append(result)
            print_progress(i, len(trial_args), start, label="sim")

    trial_a_cache = defaultdict(list)
    trial_b_cache = defaultdict(list)
    failed_a_cache = defaultdict(list)
    failed_b_cache = defaultdict(list)

    print("Processing simulation results...")
    for trial_type, noise_sd, trial_data, failed_data in all_results:
        if trial_type == "a":
            trial_a_cache[noise_sd].extend(trial_data)
            failed_a_cache[noise_sd].extend(failed_data)
        else:
            trial_b_cache[noise_sd].extend(trial_data)
            failed_b_cache[noise_sd].extend(failed_data)

    return (
        dict(trial_a_cache),
        dict(trial_b_cache),
        dict(failed_a_cache),
        dict(failed_b_cache),
    )

def combine_and_save_results(all_combinations, trial_a_cache, trial_b_cache, 
                             failed_a_cache, failed_b_cache, save_results=True):
    """Combine cached results and optionally save all combination files."""
    
    combination_summary_data = []
    
    for i, (block_noise, ramp_noise) in enumerate(all_combinations):
        # Combine trial data
        trial_data = (trial_a_cache[block_noise] + trial_b_cache[ramp_noise])
        failed_data = (failed_a_cache[block_noise] + failed_b_cache[ramp_noise])
        
        # Save combination files only if save_results is True
        if save_results:
            save_combination_files(block_noise, ramp_noise, trial_data, failed_data)
        
        # Track summary statistics
        combination_summary_data.append(_create_failure_summary(block_noise, ramp_noise, trial_data, failed_data))
    
    return combination_summary_data

def physics_sim_path(block_noise, ramp_noise):
    return os.path.join(
        DATA_DIR,
        "physics_simulations",
        f"trial_results_blk{block_noise:.3f}_rmp{ramp_noise:.3f}.csv",
    )


def save_combination_files(block_noise, ramp_noise, trial_data, failed_data):
    """Save trial results and failed attempts for a specific combination."""
    base_dir = os.path.join(DATA_DIR, 'physics_simulations')
    trial_file = physics_sim_path(block_noise, ramp_noise)
    pd.DataFrame(trial_data).to_csv(trial_file, index=False)
    if failed_data:
        failed_file = os.path.join(base_dir, f'failed_attempts_blk{block_noise:.3f}_rmp{ramp_noise:.3f}.csv')
        pd.DataFrame(failed_data).to_csv(failed_file, index=False)

def _create_failure_summary(block_noise, ramp_noise, trial_data, failed_data):
    """Create failure summary record for a combination."""
    total_attempts = len(trial_data) + len(failed_data)
    return {
        'block_noise_sd': block_noise,
        'ramp_noise_sd': ramp_noise,
        'total_failed_attempts': len(failed_data),
        'total_successful_records': len(trial_data),
        'failure_rate': len(failed_data) / total_attempts if total_attempts > 0 else 0
    }

def save_summary_files(combination_summary_data):
    """Save all summary and tracking files."""
    base_dir = os.path.join(DATA_DIR, 'physics_simulations')
    summary_file = os.path.join(base_dir, 'combination_summary.csv')
    pd.DataFrame(combination_summary_data).to_csv(summary_file, index=False)
    print(f"Combination summary saved to: {summary_file}")

def _print_final_summary(combination_summary_data):
    """Print comprehensive final summary."""
    print(f"\n=== FINAL SUMMARY ===")
    
    total_failed = sum(r['total_failed_attempts'] for r in combination_summary_data)
    total_successful = sum(r['total_successful_records'] for r in combination_summary_data)
    overall_failure_rate = total_failed / (total_failed + total_successful) if (total_failed + total_successful) > 0 else 0
    
    print(f"Completed all {len(combination_summary_data)} parameter combinations!")
    print(f"Total successful records: {total_successful:,}")
    print(f"Total failed attempts: {total_failed:,}")
    print(f"Overall failure rate: {overall_failure_rate:.2%}")
    


def run_all_combinations(n_trials=N_TRIALS, seed=DEFAULT_SEED, 
                        block_noise_sd_min=None, block_noise_sd_max=None, block_noise_sd_steps=None,
                        ramp_noise_sd_min=None, ramp_noise_sd_max=None, ramp_noise_sd_steps=None, save_results=True):
    """Run simulations for all parameter combinations and optionally save results.

    Existing trial_results CSVs are reused. Only missing noise pairs are simulated.
    """
    os.makedirs(os.path.join(DATA_DIR, 'physics_simulations'), exist_ok=True)
    block_range, ramp_range = _create_parameter_ranges(
        block_noise_sd_min, block_noise_sd_max, block_noise_sd_steps,
        ramp_noise_sd_min, ramp_noise_sd_max, ramp_noise_sd_steps
    )
    all_combinations = [(round(b, 3), round(r, 3)) for b in block_range for r in ramp_range]
    missing = [pair for pair in all_combinations if not os.path.exists(physics_sim_path(*pair))]
    print(
        f"Physics sims: {len(all_combinations) - len(missing)} exist, "
        f"{len(missing)} missing"
    )
    if not missing:
        print("All physics sim files already exist; skipping.")
        return pd.DataFrame()

    block_values = sorted({b for b, _ in missing})
    ramp_values = sorted({r for _, r in missing})
    trial_a_cache, trial_b_cache, failed_a_cache, failed_b_cache = run_parallel_simulations(
        block_values, ramp_values, n_trials, seed
    )

    if save_results:
        print("Combining results and saving files...")
    else:
        print("Combining results (files not saved)...")

    combination_summary_data = combine_and_save_results(
        missing, trial_a_cache, trial_b_cache, failed_a_cache, failed_b_cache, save_results
    )

    if save_results:
        save_summary_files(combination_summary_data)

    return pd.DataFrame(combination_summary_data)

def inspect_symmetry(block_noise_sd, ramp_noise_sd, n_trials=2000, seed=DEFAULT_SEED):
    """Helper function to inspect symmetry calculations."""
    all_results = []
    
    # Run Trial A (red and black blocks)
    for trial_name, trial_config in trial_configs["a"].items():
        trial_config_with_name = {**trial_config, "name": trial_name}
        trial_data, failed_data = run_single_trial(trial_config_with_name, block_noise_sd, ramp_noise_sd, n_trials, seed)
        all_results.extend(trial_data)
    
    # Run Trial B (blue and yellow ramps)  
    for trial_name, trial_config in trial_configs["b"].items():
        trial_config_with_name = {**trial_config, "name": trial_name}
        trial_data, failed_data = run_single_trial(trial_config_with_name, block_noise_sd, ramp_noise_sd, n_trials, seed)
        all_results.extend(trial_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate symmetry analysis
    print(f"\nSymmetry Analysis:")
    print(f"Ramp midpoint: {MID_POINT_RAMP}")
    print(f"Total records: {len(df)}")
    
    # Analyze symmetry for each element - show individual trials
    symmetry_analysis = []
    
    for element in ['red_block', 'black_block', 'blue_ramp', 'yellow_ramp']:
        # Get trial A and C data for this element
        trial_a_data = df[(df['trial'] == 'a') & (df['element'] == element)]
        trial_c_data = df[(df['trial'] == 'c') & (df['element'] == element)]
        
        if len(trial_a_data) > 0 and len(trial_c_data) > 0:
            print(f"\n{element} - Symmetry Check:")
            print(f"  (All values are averages of {len(trial_a_data)} simulations)")
            
            # Get physics positions from trial A and trial C (these should be symmetric)
            trial_a_physics = trial_a_data[trial_a_data['model'] == 'physics']['final_position'].values
            trial_c_physics = trial_c_data[trial_c_data['model'] == 'physics']['final_position'].values
            
            # Get ramp positions from trial A and trial C (these should be symmetric)
            trial_a_ramp = trial_a_data[trial_a_data['model'] == 'ramp']['final_position'].values
            trial_c_ramp = trial_c_data[trial_c_data['model'] == 'ramp']['final_position'].values
            
            # Check physics model symmetry
            a_physics_mean = trial_a_physics.mean()
            c_physics_mean = trial_c_physics.mean()
            a_dist = abs(a_physics_mean - MID_POINT_RAMP)
            c_dist = abs(c_physics_mean - MID_POINT_RAMP)
            physics_mean = (a_physics_mean + c_physics_mean) / 2
            print(f"  Physics Model (down motion):")
            print(f"    Trial A avg position: {a_physics_mean:.1f} (distance from midpoint: {a_dist:.1f})")
            print(f"    Trial C avg position: {c_physics_mean:.1f} (distance from midpoint: {c_dist:.1f})")
            
            # Check ramp model symmetry  
            a_ramp_mean = trial_a_ramp.mean()
            c_ramp_mean = trial_c_ramp.mean()
            a_dist_ramp = abs(a_ramp_mean - MID_POINT_RAMP)
            c_dist_ramp = abs(c_ramp_mean - MID_POINT_RAMP)
            print(f"  Ramp Model (up motion):")
            print(f"    Trial A avg position: {a_ramp_mean:.1f} (distance from midpoint: {a_dist_ramp:.1f})")
            print(f"    Trial C avg position: {c_ramp_mean:.1f} (distance from midpoint: {c_dist_ramp:.1f})")
            
            # Store for summary (just first trial for simplicity)
            symmetry_analysis.append({
                'element': element,
                'model': 'physics',
                'trial_a_position': trial_a_physics[0],
                'trial_c_position': trial_c_physics[0],
                'trial_a_distance': abs(trial_a_physics[0] - MID_POINT_RAMP),
                'trial_c_distance': abs(trial_c_physics[0] - MID_POINT_RAMP),
                'symmetry_error': abs(abs(trial_a_physics[0] - MID_POINT_RAMP) - abs(trial_c_physics[0] - MID_POINT_RAMP)),
                'symmetry_ok': abs(abs(trial_a_physics[0] - MID_POINT_RAMP) - abs(trial_c_physics[0] - MID_POINT_RAMP)) < 10
            })
            
            symmetry_analysis.append({
                'element': element,
                'model': 'ramp',
                'trial_a_position': trial_a_ramp[0],
                'trial_c_position': trial_c_ramp[0],
                'trial_a_distance': abs(trial_a_ramp[0] - MID_POINT_RAMP),
                'trial_c_distance': abs(trial_c_ramp[0] - MID_POINT_RAMP),
                'symmetry_error': abs(abs(trial_a_ramp[0] - MID_POINT_RAMP) - abs(trial_c_ramp[0] - MID_POINT_RAMP)),
                'symmetry_ok': abs(abs(trial_a_ramp[0] - MID_POINT_RAMP) - abs(trial_c_ramp[0] - MID_POINT_RAMP)) < 10
            })
    
    # Create summary DataFrame
    symmetry_df = pd.DataFrame(symmetry_analysis)
    all_trials = {**trial_configs["a"], **trial_configs["b"]}
    forward_ramp_down_sim_results = {trial_name: [] for trial_name in all_trials.keys()}
    forward_ramp_up_sim_results = {trial_name: [] for trial_name in all_trials.keys()}
    
    # Group results by trial and model
    for _, row in df.iterrows():
        trial_key = f"trial_{row['trial']}_{row['element']}"
        if trial_key in forward_ramp_down_sim_results:
            if row['model'] == 'agent':
                forward_ramp_down_sim_results[trial_key].append(row['final_position'])
            elif row['model'] == 'physics':
                forward_ramp_up_sim_results[trial_key].append(row['final_position'])
    
    return df, forward_ramp_down_sim_results, forward_ramp_up_sim_results, symmetry_df
   
if __name__ == "__main__":
   run_all_combinations()


