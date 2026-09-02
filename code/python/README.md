# Causal Abstraction Simulation & Model Fitting 

## Overview
This codebase simulates physical scenarios presented in the paper (generalization task - experiment 3), fits model parameters to behavioral data, and generates figures for analysis. 

## Installation 

```
conda create --name causal_stories python=3.12.8 
conda activate causal_stories
pip install -r requirements.txt
```

## Python Files

### Simulation & Model Fitting
- To reproduce results, run the following scripts in order(on a laptop using 6 CPU cores, a full run is about 20 minutes):

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
  - Keep drawing until each hypothesis has N valid landings. Opposite-side outliers are dropped; on-ramp stalls are kept. Timeouts record last x, not 0.
  
  - Inputs:
    - Uses constants and trial configurations from `conditions.py` 
  
  - Outputs:
    - `data/physics_simulations/trial_results_blk{BLOCK}_rmp{RAMP}.csv`
      - Aggregated records for all models and elements for a given noise pair.
    - `data/physics_simulations/failed_attempts_blk{...}_rmp{...}.csv` 
      - Details of runs that failed.
  
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
  - Fits Bayesian parameters `p`, `q`, `beta`, and `r` by maximizing the sum of `n_choice × log(p_choice)` over the four choices in every trial. `n_choice` is the participant count and `p_choice` is the model-predicted probability derived from posterior-weighted KDE scores and a softmax with temperature `beta`.
  
  - Inputs:
    - Participant data: `../R/cache/exp3_generalization_results.csv`.
    - KDE results from Step 2: `data/kde_results/kde_results_*.csv`.
  
  - Outputs 
    - `data/bestfit/best_parameters.csv` 
    - `data/bestfit/model_vs_human_comparison.csv` 
    - Copy of the best-fitting KDE CSV and corresponding physics simulation CSV 

- `step4_parameter_grid.py`
  - Grid search over all 7 parameters: `block_noise`, `ramp_noise`, `bandwidth`, `p`, `q`, `beta`, `r`. Same log-likelihood as step 3 (`Σ n log P` after softmax), evaluated on a discrete grid instead of continuous optimization.
  - Core loop (`evaluate_kde_cell`): for each KDE file `(block_noise, ramp_noise, bandwidth)`, compute posterior-weighted KDE scores once per `(p, q, r)` (`compute_weighted_results`), then for each `beta` apply softmax and log-likelihood (`softmax_log_likelihood`).
  - After the search (`mean_ll_by_parameter`): for each parameter value, mean log-likelihood averaging over the other six parameters, plus the SD of that spread. These 1-D summaries are what the sensitivity plots show. 
  - Runs in four sub-steps:
    1) `step1_setup` (`--step 1`) — write grid config to `step1_config.json`; reuse physics/KDE from steps 1–2, or generate any missing files
    2) `step2_grid_search` (`--step 2`) — evaluate all `(p, q, beta, r)` combinations per KDE cell; save one gzip shard per cell under `results/` and progress to `step2_progress.json`
    3) `step3_summarize` (`--step 3`) — write `step3_marginal_summary.csv` (mean and SD log-likelihood by parameter) and `step3_global_best.json` (joint best combo on the grid)
    4) `step4_plot` (`--step 4`) — figures in `figures/parameter_sensitivity/` (means only, zoomed y-axis) and `figures/parameter_sensitivity_with_sd/` (same means with SD bars)
  - Each plot: black dots = mean log-likelihood at that grid value. Colored dots: red = best mean for that parameter; purple = step 3 optimizer (`data/bestfit/best_parameters.csv`); yellow = joint grid best (`step3_global_best.json`).
  - Results are stored as shards (one CSV.gz per KDE cell), not one merged table.

  - Inputs:
    - `data/bestfit/best_parameters.csv` 
    - Participant data and KDE files 

  - Outputs:
    - `data/grid_search/step1_config.json`
    - `data/grid_search/step2_progress.json`
    - `data/grid_search/results/*.csv.gz`
    - `data/grid_search/step3_marginal_summary.csv`
    - `data/grid_search/step3_global_best.json`
    - `data/grid_search/grid_search_summary.json`
    - `figures/parameter_sensitivity/loglikelihood_vs_*.png`
    - `figures/parameter_sensitivity_with_sd/loglikelihood_vs_*.png`

### Other scripts
- `causal_abstraction.ipynb` 
  - Interactive walkthrough of the key functions used in the pipeline. 

- `conditions.py`
  - Experimental constants (frictions, colors), scenarios, trial configurations, ground-truth positions.

- `run_paths.py`
  - Shared `./data` folders and thread/display limits.

- `finding_groundtruth_parameters.py`, `finding_groundtruth_visualization.py`
  - For finding the “ground-truth” settings and visualizing the ground truth outcomes. 

- `simulation_visualization.py`
  - Runs a single interactive pygame/pymunk scene to visualize the block–ramp motion. 

