import numpy as np
import pandas as pd
import os
import argparse
import glob
import re
from datetime import datetime
import json
from scipy.special import softmax
from scipy.stats import gaussian_kde
from conditions import groundtruth_positions

# =========================
# CONFIGURATION CONSTANTS
# =========================
# Default parameters for running one demo beysian inference
DEFAULT_BANDWIDTH = 20
DEFAULT_P = 0.8
DEFAULT_Q = 0.6
DEFAULT_BETA = 0.4

# Bandwidth range configuration
USE_BANDWIDTH_RANGE = True
BANDWIDTH_RANGE = "0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 5.0"

# =========================
# BAYESIAN INFERENCE FUNCTIONS
# =========================

def define_prior(p, q, show_details=False):
    prior = {
        'physics': p,
        'agent': (1 - p) * q,
        'ramp': (1 - p) * (1 - q)
    }
    
    if show_details:
        print("Prior probabilities:")
        print(f"  p(H_physics) = p = {p:.3f}")
        print(f"  p(H_agent) = (1-p) * q = (1-{p}) * {q} = {prior['agent']:.3f}")
        print(f"  p(H_ramp) = (1-p) * (1-q) = (1-{p}) * (1-{q}) = {prior['ramp']:.3f}")
    
    return prior

def create_probability_table(r=0):
    # Define the consistency matrix
    consistency_data = {
        'Hypothesis': ['Physics', 'Agent', 'Ramp'],
        'Block on left side of forward ramp': [r, r, 1],
        'Block on right side of forward ramp': [1, 1, r],
        'Block on left side of backward ramp': [1, r, r],
        'Block on right side of backward ramp': [r, 1, 1]
    }
    
    df = pd.DataFrame(consistency_data)

    # Set the hypothesis column as index to make it the row labels
    df_indexed = df.set_index('Hypothesis')
    return df_indexed

def calculate_posterior(prior, cube_position, probability_table, show_details=False):
    """Calculate posterior distribution p(hypothesis|cube position) using Bayes' theorem."""
    
    position_map = {
        'forward_ramp_condition': 'Block on right side of forward ramp',
        'backward_ramp_condition': 'Block on right side of backward ramp'
    }
    
    if cube_position not in position_map:
        raise ValueError(f"Invalid cube position: {cube_position}. Must be one of: {list(position_map.keys())}")
    
    column_name = position_map[cube_position]
    
    # Get likelihood values for this position
    # 1 = consistent with the hypothesis (likelihood = 1), 0 = inconsistent (likelihood = 0)
    likelihoods = {}
    for hypothesis in ['Physics', 'Agent', 'Ramp']:
        likelihoods[hypothesis.lower()] = probability_table.loc[hypothesis, column_name]
    
    if show_details:
        print("Likelihood values p(cube_position|hypothesis):")
        for hyp, likelihood in likelihoods.items():
            print(f"  p({cube_position}|H_{hyp}) = {likelihood}")
        print()
    
    # Calculate unnormalized posterior (numerator of Bayes' theorem)
    unnormalized_posterior = {}
    for hypothesis in ['physics', 'agent', 'ramp']:
        likelihood = likelihoods[hypothesis]
        prior_prob = prior[hypothesis]
        unnormalized_posterior[hypothesis] = likelihood * prior_prob
        if show_details:
            print(f"  p({cube_position}|H_{hypothesis}) * p(H_{hypothesis}) = {likelihood} * {prior_prob:.4f} = {unnormalized_posterior[hypothesis]:.4f}")
    
    # Calculate normalization constant (denominator)
    marginal_likelihood = sum(unnormalized_posterior.values())
    if show_details:
        print(f"\nΣ p({cube_position}|H_i) * p(H_i) = {marginal_likelihood:.4f}")
    
    # Calculate normalized posterior probabilities
    posterior = {}
    for hypothesis in ['physics', 'agent', 'ramp']:
        if marginal_likelihood > 0:
            posterior[hypothesis] = unnormalized_posterior[hypothesis] / marginal_likelihood
        else:
            posterior[hypothesis] = 0.0
    
    if show_details:
        print("\n p(hypothesis|cube_position):")
        for hypothesis in ['physics', 'agent', 'ramp']:
            print(f"  p(H_{hypothesis}|{cube_position}) = {posterior[hypothesis]:.4f}")
    
    return posterior

def posterior_weighted_score(kde_results, posterior_forward, posterior_backward, show_details=False):
    """Combine KDE likelihoods with posterior probabilities (models = hypotheses).

    Glossary:
      - hypothesis/model: the latent explanation class among {'physics','agent','ramp'}
      - posterior: p(model | condition), computed from prior and consistency table

    For each condition and trial, compute a posterior-weighted sum of KDE likelihoods:
      score(choice) = Σ_model p(model | condition) * L_kde(choice | model)
    """
    trials = ['trial_a', 'trial_b', 'trial_c', 'trial_d']
    models = ['physics', 'agent', 'ramp']
    choices = ['choice_1', 'choice_2', 'choice_3', 'choice_4']
    
    if isinstance(kde_results, pd.DataFrame):
        kde_dict = {}
        for _, row in kde_results.iterrows():
            trial = row['trial']
            model = row['model']
            if trial not in kde_dict:
                kde_dict[trial] = {}
            kde_dict[trial][model] = {
                'choice_1': row['choice_1'],
                'choice_2': row['choice_2'],
                'choice_3': row['choice_3'],
                'choice_4': row['choice_4']
            }
        kde_results = kde_dict
    
    weighted_results = {
        'forward_ramp_condition': {},
        'backward_ramp_condition': {}
    }
    
    for condition in ['forward_ramp_condition', 'backward_ramp_condition']:
        if show_details:
            print(f"--- {condition.upper()} ---")
        posterior = posterior_forward if condition == 'forward_ramp_condition' else posterior_backward
        if show_details:
            print("Using posterior probabilities (p(model | condition)):")
            for model_name, prob in posterior.items():
                print(f"  {model_name:<8}: {prob:.4f}")
            print()
        
        weighted_results[condition] = {}
        
        for trial in trials:
            if show_details:
                print(f"  {trial.upper()}:")
            weighted_results[condition][trial] = {}
            
            for choice in choices:
                # Posterior-weighted sum over models
                weighted_sum = 0.0
                for model_name in models:
                    if trial in kde_results and model_name in kde_results[trial]:
                        kde_likelihood = kde_results[trial][model_name][choice]
                        posterior_weight = posterior.get(model_name, 0.0)
                        likelihood_contribution = posterior_weight * kde_likelihood
                        weighted_sum += likelihood_contribution
                        if show_details:
                            print(f"    {choice}: {model_name} = {posterior_weight:.4f} × {kde_likelihood:.10f} = {likelihood_contribution:.10f}")
                weighted_results[condition][trial][choice] = weighted_sum
                if show_details:
                    print(f"    {choice} TOTAL: {weighted_sum:.20f}")
            
    return weighted_results

def softmax_choice_probabilities(weighted_results, beta=0.5, show_details=False):
    """Convert weighted likelihoods to probabilities for each option for each trial and condition using softmax with beta parameter."""
    probability_results = {}
    choices = ['choice_1', 'choice_2', 'choice_3', 'choice_4']
    
    for condition in ['forward_ramp_condition', 'backward_ramp_condition']:
        probability_results[condition] = {}
        
        for trial in ['trial_a', 'trial_b', 'trial_c', 'trial_d']:
            if show_details:
                print(f"{condition} - {trial}")
            likelihoods = np.array([
                weighted_results[condition][trial]['choice_1'],
                weighted_results[condition][trial]['choice_2'],
                weighted_results[condition][trial]['choice_3'],
                weighted_results[condition][trial]['choice_4']
            ])
            # Use likelihoods directly with softmax
            probabilities = softmax(beta * likelihoods)

            probability_results[condition][trial] = {
                'choice_1': probabilities[0],
                'choice_2': probabilities[1],
                'choice_3': probabilities[2],
                'choice_4': probabilities[3]
            }
            if show_details:
                print(f"    Choice 1: {probabilities[0]:.4f}")
                print(f"    Choice 2: {probabilities[1]:.4f}")
                print(f"    Choice 3: {probabilities[2]:.4f}")
                print(f"    Choice 4: {probabilities[3]:.4f}")
                print(f"    Sum: {sum(probabilities):.6f}")
                print()
    
    return probability_results

# =========================
# KDE IMPLEMENTATION FUNCTIONS
# =========================

def fit_kde_models(df, bandwidth=0.1):
    """Fit KDE models for each trial-element-model combination."""
    trials = {
        'trial_a_red_block': {'trial': 'a', 'element': 'red_block'},
        'trial_a_black_block': {'trial': 'a', 'element': 'black_block'},
        'trial_b_blue_ramp': {'trial': 'b', 'element': 'blue_ramp'},
        'trial_b_yellow_ramp': {'trial': 'b', 'element': 'yellow_ramp'},
        'trial_c_red_block': {'trial': 'c', 'element': 'red_block'},
        'trial_c_black_block': {'trial': 'c', 'element': 'black_block'},
        'trial_d_blue_ramp': {'trial': 'd', 'element': 'blue_ramp'},
        'trial_d_yellow_ramp': {'trial': 'd', 'element': 'yellow_ramp'}
    }
    
    kde_models = {}
    model_types = ['physics', 'agent', 'ramp']
    
    for trial_name, mapping in trials.items():
        kde_models[trial_name] = {}
        
        for model in model_types:
            # Filter data for this trial-element-model combination
            trial_data = df[
                (df['trial'] == mapping['trial']) & 
                (df['element'] == mapping['element']) & 
                (df['model'] == model)
            ]
            
            if not trial_data.empty:
                positions = trial_data['final_position'].values
                kde = gaussian_kde(positions)
                kde.set_bandwidth(float(bandwidth))
                kde_models[trial_name][model] = kde
                print(f"Fitted KDE for {trial_name} - {model}: {len(positions)} points")
            else:
                print(f"Warning: No data for {trial_name} - {model}")
                kde_models[trial_name][model] = None
    
    return kde_models

# Returns the kernel density estimate at the position pos
def likelihood(kde, pos):
    """Calculate likelihood (probability density) for a position using KDE model."""
    if kde is None:
        return 0.0
    try:
        return float(kde.pdf([pos])[0])
    except Exception:
        return 0.0

def compute_combined_choice_scores(kde_models):
    """Combine element scores into trial-level choice scores by multiplying independent element likelihoods."""
    rows = []

    # Trial A uses red_block and black_block
    for model in ['physics', 'agent', 'ramp']:
        red_kde = kde_models.get('trial_a_red_block', {}).get(model)
        black_kde = kde_models.get('trial_a_black_block', {}).get(model)
        if red_kde is not None and black_kde is not None:
            red_pos = groundtruth_positions['trial_a_red']
            black_pos = groundtruth_positions['trial_a_black']
            # Each choice evaluates red and black at their respective positions for that choice
            choice_1 = likelihood(red_kde, red_pos[0]) * likelihood(black_kde, black_pos[0])  # red at pos1, black at pos2
            choice_2 = likelihood(red_kde, red_pos[1]) * likelihood(black_kde, black_pos[1])  # red at pos2, black at pos1
            choice_3 = likelihood(red_kde, red_pos[2]) * likelihood(black_kde, black_pos[2])  # red at pos3, black at pos4
            choice_4 = likelihood(red_kde, red_pos[3]) * likelihood(black_kde, black_pos[3])  # red at pos4, black at pos3
            rows.append({'trial': 'trial_a', 'model': model,
                         'choice_1': choice_1, 'choice_2': choice_2,
                         'choice_3': choice_3, 'choice_4': choice_4})

    # Trial B uses yellow_ramp and blue_ramp
    for model in ['physics', 'agent', 'ramp']:
        yellow_kde = kde_models.get('trial_b_yellow_ramp', {}).get(model)
        blue_kde = kde_models.get('trial_b_blue_ramp', {}).get(model)
        if yellow_kde is not None and blue_kde is not None:
            yellow_pos = groundtruth_positions['trial_b_yellow']
            blue_pos = groundtruth_positions['trial_b_blue']
            choice_1 = likelihood(yellow_kde, yellow_pos[0]) * likelihood(blue_kde, blue_pos[0])
            choice_2 = likelihood(yellow_kde, yellow_pos[1]) * likelihood(blue_kde, blue_pos[1])
            choice_3 = likelihood(yellow_kde, yellow_pos[2]) * likelihood(blue_kde, blue_pos[2])
            choice_4 = likelihood(yellow_kde, yellow_pos[3]) * likelihood(blue_kde, blue_pos[3])
            rows.append({'trial': 'trial_b', 'model': model,
                         'choice_1': choice_1, 'choice_2': choice_2,
                         'choice_3': choice_3, 'choice_4': choice_4})

    # Trial C uses red_block and black_block
    for model in ['physics', 'agent', 'ramp']:
        red_kde = kde_models.get('trial_c_red_block', {}).get(model)
        black_kde = kde_models.get('trial_c_black_block', {}).get(model)
        if red_kde is not None and black_kde is not None:
            red_pos = groundtruth_positions['trial_c_red']
            black_pos = groundtruth_positions['trial_c_black']
            choice_1 = likelihood(red_kde, red_pos[0]) * likelihood(black_kde, black_pos[0])
            choice_2 = likelihood(red_kde, red_pos[1]) * likelihood(black_kde, black_pos[1])
            choice_3 = likelihood(red_kde, red_pos[2]) * likelihood(black_kde, black_pos[2])
            choice_4 = likelihood(red_kde, red_pos[3]) * likelihood(black_kde, black_pos[3])
            rows.append({'trial': 'trial_c', 'model': model,
                         'choice_1': choice_1, 'choice_2': choice_2,
                         'choice_3': choice_3, 'choice_4': choice_4})

    # Trial D uses yellow_ramp and blue_ramp
    for model in ['physics', 'agent', 'ramp']:
        yellow_kde = kde_models.get('trial_d_yellow_ramp', {}).get(model)
        blue_kde = kde_models.get('trial_d_blue_ramp', {}).get(model)
        if yellow_kde is not None and blue_kde is not None:
            yellow_pos = groundtruth_positions['trial_d_yellow']
            blue_pos = groundtruth_positions['trial_d_blue']
            choice_1 = likelihood(yellow_kde, yellow_pos[0]) * likelihood(blue_kde, blue_pos[0])
            choice_2 = likelihood(yellow_kde, yellow_pos[1]) * likelihood(blue_kde, blue_pos[1])
            choice_3 = likelihood(yellow_kde, yellow_pos[2]) * likelihood(blue_kde, blue_pos[2])
            choice_4 = likelihood(yellow_kde, yellow_pos[3]) * likelihood(blue_kde, blue_pos[3])
            rows.append({'trial': 'trial_d', 'model': model,
                         'choice_1': choice_1, 'choice_2': choice_2,
                         'choice_3': choice_3, 'choice_4': choice_4})

    return pd.DataFrame(rows)

def save_results(results_df, output_file):
    """Save results to CSV file."""
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

# =========================
# MAIN ANALYSIS FUNCTIONS
# =========================

def run_kde_analysis(simulation_file, bandwidth=DEFAULT_BANDWIDTH):
    """Run KDE analysis on a simulation file."""
    print(f"Loading simulation data from: {simulation_file}")
    
    df = pd.read_csv(simulation_file)
    
    print(f"Fitting KDE models with bandwidth={bandwidth}")
    
    # Fit KDE models
    kde_models = fit_kde_models(df, bandwidth)
    
    # Compute choice scores
    choice_scores = compute_combined_choice_scores(kde_models)
    
    return {
        'kde_models': kde_models,
        'choice_scores': choice_scores,
        'simulation_data': df
    }

def run_bayesian_analysis(kde_results, p=DEFAULT_P, q=DEFAULT_Q, beta=DEFAULT_BETA):
    """Run Bayesian analysis on KDE results."""
    print(f"Running Bayesian analysis with p={p:.3f}, q={q:.3f}, beta={beta:.3f}")
    
    # Define prior
    prior = define_prior(p, q)
    
    # Create probability table
    consistency_table = create_probability_table(r=0)
    
    # Calculate posteriors
    posterior_forward = calculate_posterior(prior, 'forward_ramp_condition', consistency_table)
    posterior_backward = calculate_posterior(prior, 'backward_ramp_condition', consistency_table)
    
    # Combine KDE with posteriors
    weighted_results = posterior_weighted_score(kde_results, posterior_forward, posterior_backward)
    
    # Convert to probabilities
    probability_results = softmax_choice_probabilities(weighted_results, beta)
    
    return {
        'prior': prior,
        'posterior_forward': posterior_forward,
        'posterior_backward': posterior_backward,
        'weighted_results': weighted_results,
        'probability_results': probability_results
    }

def process_single_simulation(simulation_file, bandwidth=DEFAULT_BANDWIDTH, 
                            p=DEFAULT_P, q=DEFAULT_Q, beta=DEFAULT_BETA, 
                            save_results_flag=True, save_kde=True, save_bayesian=False):
    """Process a single simulation file through the full Bayesian inference pipeline."""
    print(f"Processing simulation: {simulation_file}")
    
    # Extract parameters from filename for output naming
    basename = os.path.basename(simulation_file)
    if 'trial_results_' in basename:
        # Try element-specific format first: "trial_results_red0.050_blk0.050_ylw0.500_blu0.500.csv"
        element_match = re.search(r'trial_results_red([0-9.]+)_blk([0-9.]+)_ylw([0-9.]+)_blu([0-9.]+)\.csv', basename)
        if element_match:
            red_noise, black_noise, yellow_noise, blue_noise = element_match.groups()
            block_noise = f"red{red_noise}_blk{black_noise}"  # Combined block noise identifier
            ramp_noise = f"ylw{yellow_noise}_blu{blue_noise}"  # Combined ramp noise identifier
            seed = '1'  # Default seed for element-specific format
        else:
            # Legacy format: "trial_results_blk0.010_rmp0.030_seed1.csv"
            parts = basename.replace('trial_results_', '').replace('.csv', '').split('_')
            block_noise = parts[0].replace('blk', '') if len(parts) > 0 else 'unknown'
            ramp_noise = parts[1].replace('rmp', '') if len(parts) > 1 else 'unknown'
            seed = parts[2].replace('seed', '') if len(parts) > 2 else '1'
    else:
        block_noise = 'unknown'
        ramp_noise = 'unknown'
        seed = 'unknown'
    
    try:
        # Step 1: Run KDE analysis
        kde_analysis = run_kde_analysis(simulation_file, bandwidth)
        
        # Step 2: Run Bayesian analysis only if saving Bayesian outputs
        bayesian_analysis = None
        if save_bayesian:
            bayesian_analysis = run_bayesian_analysis(
                kde_analysis['choice_scores'], p, q, beta
            )
        
        # Prepare results
        results = {
            'simulation_file': simulation_file,
            'block_noise_sd': block_noise,
            'ramp_noise_sd': ramp_noise,
            'seed': seed,
            'bandwidth': bandwidth,
            'bayesian_params': {'p': p, 'q': q, 'beta': beta},
            'kde_models': kde_analysis['kde_models'],
            'choice_scores': kde_analysis['choice_scores'],
            'bayesian_results': bayesian_analysis if bayesian_analysis is not None else {},
            'status': 'success'
        }
        
        # Save results if requested
        if save_results_flag:
            # Create output directory
            os.makedirs('data/kde_results', exist_ok=True)
            
            # Save KDE results (optional)
            if save_kde:
                kde_output_file = f'data/kde_results/kde_results_blk{block_noise}_rmp{ramp_noise}_bw{bandwidth:.3f}_seed{seed}.csv'
                kde_analysis['choice_scores'].to_csv(kde_output_file, index=False)
                results['kde_output_file'] = kde_output_file
                
            
            # Save Bayesian results (probabilities) (optional)
            if save_bayesian and bayesian_analysis is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                bayesian_output_file = f'data/kde_results/bayesian_results_blk{block_noise}_rmp{ramp_noise}_bw{bandwidth:.3f}_p{p:.3f}_q{q:.3f}_beta{beta:.3f}_seed{seed}_{timestamp}.csv'
                
                # Convert probability results to DataFrame
                prob_records = []
                for condition, trials in bayesian_analysis['probability_results'].items():
                    for trial, choices in trials.items():
                        for choice, prob in choices.items():
                            prob_records.append({
                                'condition': condition,
                                'trial': trial,
                                'choice': choice,
                                'probability': prob
                            })
                
                prob_df = pd.DataFrame(prob_records)
                prob_df.to_csv(bayesian_output_file, index=False)
                results['bayesian_output_file'] = bayesian_output_file
            
            print(f"Results saved:")
            if save_kde:
                print(f"  KDE: {results.get('kde_output_file', 'skipped')}")
            else:
                print("  KDE: skipped")
            if save_bayesian and bayesian_analysis is not None:
                print(f"  Bayesian: {results.get('bayesian_output_file', 'skipped')}")
            else:
                print("  Bayesian: skipped")
        
        return results
        
    except Exception as e:
        print(f"Error processing simulation: {e}")
        return {
            'simulation_file': simulation_file,
            'block_noise_sd': block_noise,
            'ramp_noise_sd': ramp_noise,
            'seed': seed,
            'bandwidth': bandwidth,
            'bayesian_params': {'p': p, 'q': q, 'beta': beta},
            'status': 'error',
            'error': str(e)
        }

def parse_bandwidth_range(bandwidth_range_str):
    """Parse bandwidth range string into a list of values."""
    if not bandwidth_range_str:
        return []
    
    # Check if it's a range format (start:end:step)
    if ':' in bandwidth_range_str:
        try:
            parts = bandwidth_range_str.split(':')
            if len(parts) == 3:
                start, end, step = map(float, parts)
                return list(np.arange(start, end + step, step))
            else:
                raise ValueError("Range format should be 'start:end:step'")
        except ValueError as e:
            print(f"Error parsing bandwidth range '{bandwidth_range_str}': {e}")
            return []
    
    try:
        return [float(x.strip()) for x in bandwidth_range_str.split(',')]
    except ValueError as e:
        print(f"Error parsing bandwidth values '{bandwidth_range_str}': {e}")
        return []

def process_all_simulations(bandwidth=DEFAULT_BANDWIDTH, p=DEFAULT_P, q=DEFAULT_Q, beta=DEFAULT_BETA,
                            save_results=True, save_kde=True, save_bayesian=False):
    """Process all simulation files in the physics_simulations directory."""
    # Find all simulation files
    simulation_pattern = "data/physics_simulations/trial_results_*.csv"
    simulation_files = glob.glob(simulation_pattern)
    
    if not simulation_files:
        print(f"No simulation files found matching pattern: {simulation_pattern}")
        print("Please run Step 1 (physics simulations) first:")
        print("  python step1_run_physics_simulations.py --all_combinations")
        return {'status': 'no_files', 'results': []}
    
    print(f"Found {len(simulation_files)} simulation files:")
    for file in simulation_files:
        print(f"  {file}")
    print()
    
    # Process each simulation
    results = []
    successful = 0
    failed = 0
    
    for i, simulation_file in enumerate(simulation_files, 1):
        print(f"Processing {i}/{len(simulation_files)}: {os.path.basename(simulation_file)}")
        result = process_single_simulation(
            simulation_file,
            bandwidth,
            p,
            q,
            beta,
            save_results_flag=save_results,
            save_kde=save_kde,
            save_bayesian=save_bayesian
        )

        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
        print()
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"data/kde_results/bayesian_inference_summary_{timestamp}.csv"
    
    summary_records = []
    for result in results:
        summary_records.append({
            'simulation_file': result['simulation_file'],
            'block_noise_sd': result['block_noise_sd'],
            'ramp_noise_sd': result['ramp_noise_sd'],
            'seed': result['seed'],
            'bandwidth': result['bandwidth'],
            'p': result['bayesian_params']['p'],
            'q': result['bayesian_params']['q'],
            'beta': result['bayesian_params']['beta'],
            'status': result['status'],
            'kde_output_file': result.get('kde_output_file', ''),
            'bayesian_output_file': result.get('bayesian_output_file', ''),
            'error': result.get('error', '')
        })
    
    df = pd.DataFrame(summary_records)
    df.to_csv(summary_file, index=False)
    
    print("=== SUMMARY ===")
    print(f"Total files processed: {len(simulation_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(simulation_files))*100:.1f}%")
    print(f"Summary saved to: {summary_file}")
    
    return {
        'total_files': len(simulation_files),
        'successful': successful,
        'failed': failed,
        'success_rate': (successful/len(simulation_files))*100,
        'results': results,
        'summary_file': summary_file
    }

def check_prerequisites():
    """Check if Step 1 has been run and simulation files exist."""
    simulation_pattern = "data/physics_simulations/trial_results_*.csv"
    simulation_files = glob.glob(simulation_pattern)
    
    if not simulation_files:
        print("No simulation files found!")
        print(f"   Looking for: {simulation_pattern}")
        print()
        print("Please run Step 1 first:")
        print("   python step1_physics_simulations.py --all_combinations --optimized")
        print()
        return False
    
    print(f"Found {len(simulation_files)} simulation files")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Step 2: KDE or Bayesian processing on physics simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1) KDE only (default). Edit parameters at top of file.
  python step2_bayesian_inference.py --kde

  # 2) Bayesian only. Edit parameters at top of file.
  python step2_bayesian_inference.py --bayesian

  # Optionally run on a single file instead of all
  python step2_bayesian_inference.py --kde --file data/physics_simulations/trial_results_*.csv
        """
    )
    
    # Minimal arguments
    parser.add_argument('--file', type=str, default=None,
                       help='Path to specific simulation CSV file (optional)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--kde', action='store_true', help='Run KDE only (default)')
    mode.add_argument('--bayesian', action='store_true', help='Run Bayesian only')
    
    args = parser.parse_args()

    if not check_prerequisites():
        return
    
    # Always save outputs; parameters are edited in-file
    save_results = True
    
    # Determine bandwidth values to use (from configuration only)
    if USE_BANDWIDTH_RANGE:
        bandwidths = parse_bandwidth_range(BANDWIDTH_RANGE)
        if not bandwidths:
            print(f"Invalid bandwidth range in configuration: {BANDWIDTH_RANGE}")
            return
        print(f"Using configuration bandwidth range: {bandwidths}")
    else:
        # Use single bandwidth from configuration
        bandwidths = [DEFAULT_BANDWIDTH]
        print(f"Using configuration single bandwidth: {DEFAULT_BANDWIDTH}")
    
    # Mode selection: default KDE only
    if args.kde or not args.bayesian:
        save_kde_flag, save_bayesian_flag = True, False
    else:
        save_kde_flag, save_bayesian_flag = False, True

    if args.file:
        # Process single simulation file
        if not os.path.exists(args.file):
            print(f"Simulation file not found: {args.file}")
            print()
            print("Available simulation files:")
            simulation_files = glob.glob("data/physics_simulations/trial_results_*.csv")
            for file in simulation_files[:5]:  
                print(f"   {file}")
            if len(simulation_files) > 5:
                print(f"   ... and {len(simulation_files) - 5} more files")
            return
        
        print("=== SINGLE FILE PROCESSING ===")
        print(f"File: {args.file}")
        print(f"Parameters (from config): p={DEFAULT_P:.3f}, q={DEFAULT_Q:.3f}, beta={DEFAULT_BETA:.3f}")
        print(f"Bandwidths: {bandwidths}")
        print("Save results: Yes")
        print()
        
        # Process for each bandwidth
        for bandwidth in bandwidths:
            print(f"\n--- Processing with bandwidth={bandwidth} ---")
            result = process_single_simulation(
                args.file,
                bandwidth=bandwidth,
                p=DEFAULT_P,
                q=DEFAULT_Q,
                beta=DEFAULT_BETA,
                save_results_flag=save_results,
                save_kde=save_kde_flag,
                save_bayesian=save_bayesian_flag
            )
            
            if result['status'] == 'success':
                print(f"Processing completed for bandwidth={bandwidth}!")
                if save_results:
                    print(f"   KDE results: {result.get('kde_output_file', 'Not saved')}")
                    print(f"   Bayesian results: {result.get('bayesian_output_file', 'Not saved')}")
            else:
                print(f"Processing failed for bandwidth={bandwidth}: {result.get('error', 'Unknown error')}")
    else:
        # Process all simulations (default behavior)
        print("=== PROCESSING ALL SIMULATIONS ===")
        print(f"Parameters (from config): p={DEFAULT_P:.3f}, q={DEFAULT_Q:.3f}, beta={DEFAULT_BETA:.3f}")
        print(f"Bandwidths: {bandwidths}")
        print("Save results: Yes")
        print()
        
        # Process for each bandwidth
        all_results = []
        for bandwidth in bandwidths:
            print(f"\n--- Processing all simulations with bandwidth={bandwidth} ---")
            results = process_all_simulations(
                bandwidth=bandwidth,
                p=DEFAULT_P,
                q=DEFAULT_Q,
                beta=DEFAULT_BETA,
                save_results=save_results,
                save_kde=save_kde_flag,
                save_bayesian=save_bayesian_flag
            )
            all_results.append(results)
            
        
        # Summary
        total_processed = sum(r['total_files'] for r in all_results)
        if total_processed > 0:
            print(f"\nOverall completion: {total_processed} files processed across {len(bandwidths)} bandwidth values.")
            print()
            print("Next step: Run parameter optimization")
            print("python step3_optimize_parameters.py")
        else:
            print("\nNo files were processed across any bandwidth")

    # Step 2 completed successfully
    print("Step 2 completed successfully!")

if __name__ == "__main__":
    main()
