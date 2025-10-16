# Readme 

Information about the different blender files. 

## Experiment 2

- `exp2_ramp_cube.blend`: Renders stimuli for the long and short conditions of Experiment 2. The color of the cube and ramp, as well as the friction of the cube and ramp, were modified in order to render each unique stimulus. When the cube determined the scenario outcome, the friction values were: `cube(0.585, 0.445)`, `ramp(0.53, 0.4)`. When the ramp determined the outcome, the friction values were: `cube(0.5, 0.45)`, `ramp(0.655, 0.41)`. These values created four equally spaced outcomes.

- `exp2_ramp_cube_conjunctive.blend`: Renders stimuli for the conjunctive condition of Experiment 2. It is identical to `exp2_ramp_cube.blend` except that the finish line is moved such that the cube only crosses the finish line when it moves to the farthest position. The procedure and friction values used were identical to the other conditions of Experiment 2.

## Experiment 3

- `exp3_start_positions.blend`: Creates the 'start position' images for Experiment 3. For this experiment, the cube started in the middle of the ramp rather than the top. The colors of the cube and ramp and the ramp orientation (180-degree rotation) were altered to create the unique scenarios in both the forward and backward facing conditions.

- `exp3_end_positions.blend`: Create the 'end position' images for Experiment 3, showing the outcome of each scenario. The 'end position' images were created by running simulations, using the same friction values used for Experiment 2, and capturing images of the cube’s final resting positions. In the conditions where the ramp faced backwards, the ramp was rotated 180 degrees after the cube reached its final position and prior to rendering the images.

- `exp3_multi_ramp_cube.blend`: Generates the 'generalization check' images for Experiment 3. The colors and frictions of the ramps and cubes were altered, and images of the initial and final positions were rendered to create unique stimuli. The cubes and/or ramps were rotated 180 degrees prior to, or after, the simulation in order to create stimuli with differing ramp orientations and outcomes. Rotations preserved the location of the ramp and the outcome positions for the cubes. Because these stimuli featured novel ramps and cubes, new friction values were used. The ramp’s friction was always 0.5, and the cube’s friction was 0.5 or 0.6 for further or closer positions, respectively.