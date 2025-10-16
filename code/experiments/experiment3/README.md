# Readme

To run the experiment, click on the `index.html` file and then change the ending of URL in the browser to one of the following: 

Syntax: `index.html?condition=x_y`

- x = 1:
    + cube determines line crossing
    + high friction cube red, low friction cube black
    + high friction ramp blue, low friction ramp yellow

- x = 2:
    + cube determines line crossing
    + high friction cube black, low friction cube red
    + high friction ramp yellow, low friction ramp blue

- x = 3:
    + ramp determines line crossing
    + high friction cube black, low friction cube red
    + high friction ramp yellow, low friction ramp blue

- x = 4:
    + ramp determines line crossing
    + high friction cube red, low friction cube black
    + high friction ramp blue, low friction ramp yellow

- y = f:
    + Training stimuli feature a forward-facing ramp

- y = b:
    + Training stimuli feature a backward-facing ramp

Example: `index.html?condition=1_f`

- cube determines line crossing
- high friction cube red, low friction cube black
- high friction ramp blue, low friction ramp yellow
- Training stimuli feature a forward-facing ramp

## Randomization 

- The following is determined randomly for each participant
    + The order of y/n buttons on prediction trials
    + The order of appearance of the 4 surprise quiz trials
    + The order of appearance of the 2 generalization trials with consistent ramp orientation
    + The order of appearance of the 2 generalization trials with reversed ramp orientation
    + The order of presentation of the 16 prediction trials
    + Which cube or ramp is in the foreground, or background, in each generalization check trial
