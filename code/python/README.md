# Causal Abstraction Simulation & Model Fitting 

## Overview
This codebase simulates physical scenarios presented in the paper (generalization task - experiment 3), fits model parameters to behavioral data, and generates figures for analysis. 

## Installation 

```
conda create -name causal_stories python=3.12.8 
conda activate causal_stories
pip install -r requirements.txt
```

## Python Files

### Simulation & Model Fitting
- To reproduce the results from the paper, run the following 3 scripts in order.

- **`step1_physics_simulations.py`**
  - Generates physics simulation data. Produces final x-positions for three model hypotheses (physics, agent, ramp) across four trials (A–D). 
  - How data are generated:
    - Trial A 
      - Noise: block friction noise only (ramp noise is 0).
      - Downward motion: simulates final positions for the physics and agent models.
      - Upward motion: simulates final positions for the ramp model.
    - Trial B 
      - Noise: ramp friction noise only (block noise is 0).
      - Downward motion: simulates final positions for the physics and agent models.
      - Upward motion: simulates final positions for the ramp model.
    - Trial C
      - Final positions obtained by symmetry transformation of Trial A results around the ramp midpoint `MID_POINT_RAMP`.
    - Trial D 
      - Final positions obtained by symmetry transformation of Trial B results around the ramp midpoint `MID_POINT_RAMP`.
  
  - Inputs:
    - Uses constants and trial configurations from `conditions.py` 
  
  - Outputs:
    - `data/physics_simulations/trial_results_blk{BLOCK}_rmp{RAMP}.csv`
      - Aggregated records for all models and elements for a given noise pair.
    - `data/physics_simulations/failed_attempts_blk{...}_rmp{...}.csv` 
      - Details of runs that failed.
    - `data/physics_simulations/simulation_tracking.csv`
      - Tracking of generated unique simulations and combined outputs.
  
- `step2_bayesian_inference.py`
  - Converts physics simulation outputs into choice likelihoods for each possible target position using KDE.
    - Processing steps (KDE path):
      1) Load a simulation CSV into a DataFrame.
      2) Fit KDE per trial–element–model using `scipy.stats.gaussian_kde` on final positions.
      3) Compute trial-level choice scores by multiplying independent element likelihoods at the four ground-truth positions per trial (`compute_combined_choice_scores`).
      4) For each trial-model combination, save the result with columns `choice_1..choice_4` to a CSV in `data/kde_results/`.
  - Optional (for testing Bayesian inference process): combine with Bayesian posteriors and a softmax with temperature `beta` to produce predicted choice probabilities.
      1) Build prior `p(hypothesis)` using `define_prior(p, q)`.
      2) Build a probability table (`create_probability_table`) and compute posteriors for forward and backward conditions using `calculate_posterior`.
      3) Weight KDE choice scores by posteriors (per condition) to get posterior-weighted scores (`posterior_weighted_score`).
      4) Convert to probabilities per trial using `softmax_choice_probabilities(weighted_results, beta)`.
    
  
  - Inputs:
    - CSV files from Step 1: `data/physics_simulations/trial_results_*.csv`
    - `conditions.py` for ground-truth choice positions used when combining element likelihoods into trial scores (`groundtruth_positions`).
  
  - Outputs:
    - KDE results:
      - `data/kde_results/kde_results_blk{BLOCK}_rmp{RAMP}_bw{BANDWIDTH}_seed{SEED}.csv`
    - Optional Bayesian probabilities (if enabled in code):
      - `data/kde_results/bayesian_results_blk{...}_rmp{...}_bw{...}_p{...}_q{...}_beta{...}_seed{...}_{timestamp}.csv`
        
- `step3_optimize_parameters.py`
  - Fits Bayesian parameters `p`, `q`, and `beta` by maximizing the sum of `n_choice × log(p_choice)` over the four choices in every trial. `n_choice` is the participant count and `p_choice` is the model-predicted probability derived from posterior-weighted KDE scores and a softmax with temperature `beta`.
  
  - Inputs:
    - Participant data: `../R/cache/exp3_generalization_results.csv`.
    - KDE results from Step 2: `data/kde_results/kde_results_*.csv`.
  
  - Outputs 
    - `data/bestfit/best_parameters.csv` 
    - `data/bestfit/model_vs_human_comparison.csv` 
    - Copy of the best-fitting KDE CSV and corresponding physics simulation CSV 
  
### Other scripts
- `causal_abstraction.ipynb` 
  - Interactive walkthrough of the key functions used in the pipeline. 

- `conditions.py`
  - Experimental constants (frictions, colors), scenarios, trial configurations, ground-truth positions.

- `create_graphs.py`
  - Reads the results produced in Step 3 and renders a figure to show participant data vs. model prediction.
  - Input: `data/bestfit/model_vs_human_comparison.csv`. 
  - Output: `figures/model_vs_human_predictions.png`.

- `finding_groundtruth_parameters.py`, `finding_groundtruth_visualization.py`
  - For finding the “ground-truth” settings and visualizing the ground truth outcomes. 

- `simulation_visualization.py`
  - Runs a single interactive pygame/pymunk scene to visualize the block–ramp motion. 

