# Block friction values
BLOCK_FRICTION_RED = 0.2640
BLOCK_FRICTION_BLACK = 0.3929
BLOCK_FRICTION_GREY = 0.3015

# Ramp friction values
RAMP_FRICTION_YELLOW = 0.5698
RAMP_FRICTION_BLUE = 0.1023
RAMP_FRICTION_GREY = 0.2012

FORCE_MAGNITUDE = 557.4

# Colors
red = (255, 0, 0, 1.0)
black = (0, 0, 0, 1.0)
yellow = (204, 204, 0, 1.0)
blue = (0, 6, 139, 1.0)
grey = (191, 191, 191, 1.0)
white = (255, 255, 255, 1.0)

# Ramp geometry 
RAMP_LEFT_X = 520
RAMP_RIGHT_X = 790
MID_POINT_RAMP = (RAMP_LEFT_X + RAMP_RIGHT_X) / 2 

# Block configurations
block_config = {
    "red": {"block_color": red, "block_friction": BLOCK_FRICTION_RED, "block_noise_sd": 0.0},
    "black": {"block_color": black, "block_friction": BLOCK_FRICTION_BLACK, "block_noise_sd": 0.0},
    "grey": {"block_color": grey, "block_friction": BLOCK_FRICTION_GREY, "block_noise_sd": 0.0}
}

# Ramp configurations
ramp_config = {
    "yellow": {"ramp_color": yellow, "ramp_friction": RAMP_FRICTION_YELLOW, "ramp_noise_sd": 0.0},
    "blue": {"ramp_color": blue, "ramp_friction": RAMP_FRICTION_BLUE, "ramp_noise_sd": 0.0},
    "grey": {"ramp_color": grey, "ramp_friction": RAMP_FRICTION_GREY, "ramp_noise_sd": 0.0}
}

# Groundtruth positions for each trial (left-most to right-most)
# Trial A positions 
trial_a_position_one = 231.0
trial_a_position_two = 327.1
trial_a_position_three = 972.8
trial_a_position_four = 1089.0

# Trial B positions 
trial_b_position_one = 223.0
trial_b_position_two = 332.3
trial_b_position_three = 981.0
trial_b_position_four = 1073.4

# Trial C positions (flipped from Trial A through ramp midpoint = 655)
trial_c_position_one = 221.0
trial_c_position_two = 337.2
trial_c_position_three = 982.9
trial_c_position_four = 1079.0

# Trial D positions (flipped from Trial B through ramp midpoint = 655)
trial_d_position_one = 236.6 
trial_d_position_two = 329.0 
trial_d_position_three = 977.7
trial_d_position_four = 1087.0

groundtruth_positions = {
    "trial_a_red": [trial_a_position_one, trial_a_position_two, trial_a_position_three, trial_a_position_four],
    "trial_a_black": [trial_a_position_two, trial_a_position_one, trial_a_position_four, trial_a_position_three],
    "trial_b_blue": [trial_b_position_two, trial_b_position_one, trial_b_position_four, trial_b_position_three],
    "trial_b_yellow": [trial_b_position_one, trial_b_position_two, trial_b_position_three, trial_b_position_four],
    "trial_c_red": [trial_c_position_one, trial_c_position_two, trial_c_position_three, trial_c_position_four],
    "trial_c_black": [trial_c_position_two, trial_c_position_one, trial_c_position_four, trial_c_position_three],
    "trial_d_blue": [trial_d_position_two, trial_d_position_one, trial_d_position_four, trial_d_position_three],
    "trial_d_yellow": [trial_d_position_one, trial_d_position_two, trial_d_position_three, trial_d_position_four],
}

# Only simulate trials A & B, calculate C & D through symmetry

base_trials = [
      {"name": "trial_a_red_block", **block_config["red"], **ramp_config["grey"],
       "ramp_direction": "forward",
       "override_block_sd": True, "override_ramp_sd": False},

      {"name": "trial_a_black_block", **block_config["black"], **ramp_config["grey"],
       "ramp_direction": "forward",
       "override_block_sd": True, "override_ramp_sd": False},

      {"name": "trial_b_blue_ramp", **block_config["grey"], **ramp_config["blue"],
       "ramp_direction": "forward",
       "override_block_sd": False, "override_ramp_sd": True},

      {"name": "trial_b_yellow_ramp", **block_config["grey"], **ramp_config["yellow"],
       "ramp_direction": "forward",
       "override_block_sd": False, "override_ramp_sd": True},
  ]

# Create trial_configs structure for step1_physics_simulations.py
trial_configs = {
    "a": {
        "trial_a_red_block": base_trials[0],
        "trial_a_black_block": base_trials[1]
    },
    "b": {
        "trial_b_blue_ramp": base_trials[2],
        "trial_b_yellow_ramp": base_trials[3]
    }
}

# Create scenarios directly - each trial gets two simulation types
scenarios = []
for trial in base_trials:
    scenarios.append(("forward_ramp_down", {**trial, "model_type": "agent_physics"}))
    scenarios.append(("forward_ramp_up", {**trial, "model_type": "ramp"}))
