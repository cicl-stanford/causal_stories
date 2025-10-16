import numpy as np
import pandas as pd
import os
import glob
import time
from scipy.optimize import minimize
from scipy.special import softmax
from datetime import datetime
from step2_bayesian_inference import (
    define_prior,
    create_probability_table,
    calculate_posterior,
    posterior_weighted_score,
    softmax_choice_probabilities
)
# =========================
# CONFIGURATION CONSTANTS
# =========================
P_BOUNDS = (0.01, 0.99)
Q_BOUNDS = (0.01, 0.99)
BETA_BOUNDS = (100, 10000000)
R_BOUNDS = (0.0, 1.0)

# Progress tracking
DETAILED_OUTPUT = False

# Multiple random restarts configuration
N_RANDOM_RESTARTS = 20

# =========================
# MAIN OPTIMIZATION FUNCTIONS
# =========================

def load_participant_data(show_details=False):
    """Load actual participant choice data from CSV file."""
    if show_details:
        print("Loading participant data from exp3_generalization_results.csv...")
    
    # Load the participant data
    participant_file = "../R/cache/exp3_generalization_results.csv"
    
    df = pd.read_csv(participant_file)
    
    # Convert participant data to optimization format
    # Mapping: trial 1,2,3,4 -> trial_a,trial_b,trial_c,trial_d
    trial_mapping = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
    condition_mapping = {'forward': 'forward_ramp_condition', 'backward': 'backward_ramp_condition'}
    
    participant_data = {
        'forward_ramp_condition': {},
        'backward_ramp_condition': {}
    }
    
    for condition in ['forward', 'backward']:
        for trial_num in [1, 2, 3, 4]:
            trial_letter = trial_mapping[trial_num]
            condition_name = condition_mapping[condition]
            
            # Get counts for this trial-condition combination
            trial_data = df[(df['training'] == condition) & (df['trial'] == trial_num)]
            
            choice_counts = {}
            for _, row in trial_data.iterrows():
                choice_key = f"choice_{int(row['selection'])}"
                choice_counts[choice_key] = int(row['n'])
            
            # Ensure all choices are represented (fill with 0 if missing)
            for choice in ['choice_1', 'choice_2', 'choice_3', 'choice_4']:
                if choice not in choice_counts:
                    choice_counts[choice] = 0
            
            participant_data[condition_name][f"trial_{trial_letter}"] = choice_counts
            
            # Print summary
            total_count = sum(choice_counts.values())
            if show_details:
                print(f"  {condition_name} - trial_{trial_letter}: {choice_counts} (total: {total_count})")
    
    return participant_data

def print_progress(current, total, start_time, prefix="Progress", filename=None):
    """Print progress with time estimation."""
    elapsed = time.time() - start_time
    if current > 0:
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        estimated_remaining = avg_time_per_item * remaining_items
        
        progress_pct = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Format time estimates
        if estimated_remaining < 60:
            eta_str = f"{estimated_remaining:.0f}s"
        elif estimated_remaining < 3600:
            eta_str = f"{estimated_remaining/60:.1f}m"
        else:
            eta_str = f"{estimated_remaining/3600:.1f}h"
            
        if elapsed < 60:
            elapsed_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            elapsed_str = f"{elapsed/60:.1f}m"
        else:
            elapsed_str = f"{elapsed/3600:.1f}h"
        
        # Show current file being processed
        current_info = f" ({os.path.basename(filename)})" if filename else ""
        
        print(f"\r{prefix}: |{bar}| {progress_pct:.1f}% ({current}/{total}) "
              f"Elapsed: {elapsed_str} ETA: {eta_str}{current_info}", end='', flush=True)
        
        if current == total:
            print(f"Completed in {elapsed_str}")

def calculate_log_likelihood(p, q, beta, r, kde_results, participant_data, show_details=False):
    """ Calculate log-likelihood using actual participant choice counts."""
    try:
        if show_details:
            print(f"Calculating likelihood: p={p:.4f}, q={q:.4f}, beta={beta:.4f}, r={r:.4f}")
        consistency_table = create_probability_table(r=r)
        
        # Calculate posteriors
        prior = define_prior(p, q, show_details=show_details)
        posterior_forward = calculate_posterior(prior, 'forward_ramp_condition', consistency_table, show_details=show_details)
        posterior_backward = calculate_posterior(prior, 'backward_ramp_condition', consistency_table, show_details=show_details)
        
        # Check if posteriors are valid
        if (posterior_forward is None or posterior_backward is None or
            any(np.isnan(list(posterior_forward.values()))) or
            any(np.isnan(list(posterior_backward.values())))):
            if show_details:
                print("Invalid posteriors detected - returning -inf")
            return float('-inf')
        
        # Combine KDE with posteriors and convert to choice probabilities
        weighted_results = posterior_weighted_score(kde_results, posterior_forward, posterior_backward, show_details=show_details)
        probability_results = softmax_choice_probabilities(weighted_results, beta, show_details=show_details)
        
        total_log_likelihood = 0.0
        
        for condition in ['forward_ramp_condition', 'backward_ramp_condition']:
            condition_log_likelihood = 0.0
            for trial in ['trial_a', 'trial_b', 'trial_c', 'trial_d']:
                if condition in probability_results and trial in probability_results[condition]:
                    predicted_probs = probability_results[condition][trial]
                    choice_counts = participant_data[condition][trial]
                    
                    trial_log_likelihood = 0.0
                    
                    # Calculate log-likelihood for each choice
                    for choice in ['choice_1', 'choice_2', 'choice_3', 'choice_4']:
                        if choice in predicted_probs and choice in choice_counts:
                            prob = predicted_probs[choice]
                            count = choice_counts[choice]
                            choice_log_likelihood = count * np.log(prob)
                            trial_log_likelihood += choice_log_likelihood
                    
                    condition_log_likelihood += trial_log_likelihood
            
            total_log_likelihood += condition_log_likelihood
        
        if show_details:
            print(f"Total log-likelihood: {total_log_likelihood:.6f}")
        
        return total_log_likelihood
        
    except Exception as e:
        print(f"Error in calculate_log_likelihood: {e}")
        return float('-inf')

def optimize_parameters(kde_results, participant_data, seed=1, show_details=False, n_restarts=10):
    """Optimize p, q, beta parameters to maximize likelihood using multiple random restarts."""
    np.random.seed(seed)
    best_result = None
    best_log_likelihood = float('-inf')
    best_predictions = None
    
    def objective(params):
        p, q, beta, r = params
        return calculate_log_likelihood(p, q, beta, r, kde_results, participant_data, show_details=show_details)

    # Use only random restarts (no predefined guesses)
    all_guesses = []
    
    # Generate random restarts
    for i in range(n_restarts):
        p_random = np.random.uniform(P_BOUNDS[0], P_BOUNDS[1])
        q_random = np.random.uniform(Q_BOUNDS[0], Q_BOUNDS[1])
        beta_random = np.random.uniform(BETA_BOUNDS[0], BETA_BOUNDS[1])
        r_random = np.random.uniform(R_BOUNDS[0], R_BOUNDS[1])
        all_guesses.append([p_random, q_random, beta_random, r_random])
    
    print(f"Running optimization with {len(all_guesses)} random restarts (no predefined guesses)")
    
    for i, guess in enumerate(all_guesses):
        print(f"Restart {i+1}/{len(all_guesses)} (random): p={guess[0]:.3f}, q={guess[1]:.3f}, beta={guess[2]:.3f}, r={guess[3]:.3f}")
        
        result = minimize(
            lambda params: -objective(params),  # Minimize negative to maximize log-likelihood
            guess,
            bounds=[P_BOUNDS, Q_BOUNDS, BETA_BOUNDS, R_BOUNDS],
            method='L-BFGS-B'  # Constrained optimization with bounds
        )
        
        # Extract log-likelihood (we minimized -log_likelihood to maximize log_likelihood)
        log_likelihood = -result.fun
        p_opt, q_opt, beta_opt, r_opt = result.x
        
        print(f"  Result: p={p_opt:.4f}, q={q_opt:.4f}, beta={beta_opt:.4f}, r={r_opt:.4f}, log_likelihood={log_likelihood:.6f}")
        
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_result = result
            print(f"  New best: {best_log_likelihood:.6f}")
        else:
            print(f"  Log-likelihood: {log_likelihood:.6f}")
    
    print(f"\nOptimization Summary:")
    print(f"=" * 30)
    
    if best_result is None:
        print(f"All optimization attempts failed!")
        return {
            'status': 'failed',
            'p_opt': None,
            'q_opt': None,
            'beta_opt': None,
            'r_opt': None,
            'log_likelihood': float('-inf'),
            'predictions': None,
            'error': 'All optimization attempts failed'
        }
    
    p_opt, q_opt, beta_opt, r_opt = best_result.x
    print(f"Best parameters:")
    print(f"  p (physics prior): {p_opt:.6f}")
    print(f"  q (agent vs ramp): {q_opt:.6f}")
    print(f"  beta (temperature): {beta_opt:.6f}")
    print(f"  r (consistency): {r_opt:.6f}")
    print(f"  Log-likelihood: {best_log_likelihood:.6f}")
    print(f"  Convergence: {'Yes' if best_result.success else 'No'}")
    print(f"  Iterations: {best_result.nit}")
    print(f"  Function evaluations: {best_result.nfev}")
    print()
    
    # Get final predictions for the best result
    final_log_likelihood = calculate_log_likelihood(p_opt, q_opt, beta_opt, r_opt, kde_results, participant_data, show_details=show_details)
    
    # Generate predictions for display
    try:
        consistency_table = create_probability_table(r=r_opt)
        prior = define_prior(p_opt, q_opt)
        posterior_forward = calculate_posterior(prior, 'forward_ramp_condition', consistency_table, show_details=show_details)
        posterior_backward = calculate_posterior(prior, 'backward_ramp_condition', consistency_table, show_details=show_details)
        weighted_results = posterior_weighted_score(kde_results, posterior_forward, posterior_backward, show_details=show_details)
        predictions = softmax_choice_probabilities(weighted_results, beta_opt, show_details=show_details)
        
    except Exception as e:
        predictions = None
        print(f"Warning: Could not generate predictions for display: {e}")
    
    print(f"Optimization Complete!")
    print(f"Status: Success")
    print(f"Final Results:")
    print(f"  p = {p_opt:.6f}")
    print(f"  q = {q_opt:.6f}")
    print(f"  beta = {beta_opt:.6f}")
    print(f"  r = {r_opt:.6f}")
    print(f"  Log-likelihood = {final_log_likelihood:.6f}")
    print(f"  Predictions: {'Generated' if predictions is not None else 'Failed'}")
    print()
    
    return {
        'status': 'success',
        'p_opt': p_opt,
        'q_opt': q_opt,
        'beta_opt': beta_opt,
        'r_opt': r_opt,
        'log_likelihood': final_log_likelihood,
        'predictions': predictions,
        'optimization_result': best_result
    }

def save_best_fit_results(best_result, predictions, participant_data, kde_file_info):
    """Save the best fitting results to the bestfit folder"""
    try:
        import shutil
        import re
        
        # Create bestfit directory
        os.makedirs('data/bestfit', exist_ok=True)
        
        # 1. Save best parameters
        params_data = {
            'parameter': ['p', 'q', 'beta', 'r', 'block_noise', 'ramp_noise', 'bandwidth'],
            'value': [best_result['p_opt'], best_result['q_opt'], best_result['beta_opt'], best_result['r_opt'],
                     kde_file_info.get('block_noise', 'unknown'), kde_file_info.get('ramp_noise', 'unknown'), kde_file_info.get('bandwidth', 'unknown')]
        }
        
        params_df = pd.DataFrame(params_data)
        params_file = 'data/bestfit/best_parameters.csv'
        params_df.to_csv(params_file, index=False)
        print(f"Best parameters saved to: {params_file}")
        
        # 2. Save model vs human comparison with specified column names
        if predictions is not None:
            comparison_records = []
            for condition in ['forward_ramp_condition', 'backward_ramp_condition']:
                if condition in predictions:
                    for trial in ['trial_a', 'trial_b', 'trial_c', 'trial_d']:
                        if trial in predictions[condition]:
                            predicted = predictions[condition][trial]
                            counts = participant_data[condition][trial]
                            total_count = sum(counts.values())
                            
                            comparison_records.append({
                                'condition': condition,
                                'trial': trial,
                                'choice_1_prediction': predicted['choice_1'],
                                'choice_2_prediction': predicted['choice_2'],
                                'choice_3_prediction': predicted['choice_3'],
                                'choice_4_prediction': predicted['choice_4'],
                                'choice_1_participant_data': counts['choice_1']/total_count,
                                'choice_2_participant_data': counts['choice_2']/total_count,
                                'choice_3_participant_data': counts['choice_3']/total_count,
                                'choice_4_participant_data': counts['choice_4']/total_count
                            })
            
            comparison_df = pd.DataFrame(comparison_records)
            comparison_file = 'data/bestfit/model_vs_human_comparison.csv'
            comparison_df.to_csv(comparison_file, index=False)
            print(f"Model vs human comparison saved to: {comparison_file}")
            print(f"  → Ready for create_model_human_graphs.py to generate comparison graphs")
            
            # 3. Copy KDE file from kde_results folder
            kde_source = best_result['kde_file']
            kde_dest = 'data/bestfit/' + os.path.basename(kde_source)
            if os.path.exists(kde_source):
                shutil.copy2(kde_source, kde_dest)
                print(f"KDE file copied to: {kde_dest}")
            else:
                print(f"Warning: KDE file not found: {kde_source}")
            
            # 4. Copy physics simulation file from physics_simulations folder
            # Extract block and ramp noise from KDE filename to find corresponding physics file
            kde_basename = os.path.basename(kde_source)
            # Expected format: kde_results_blk{block_noise}_rmp{ramp_noise}_bw{bandwidth}_seed{seed}.csv
            match = re.search(r'kde_results_blk([0-9.]+)_rmp([0-9.]+)_bw([0-9.]+)_seed([0-9]+)\.csv', kde_basename)
            if match:
                block_noise, ramp_noise, bandwidth, seed = match.groups()
                physics_file = f"data/physics_simulations/trial_results_blk{block_noise}_rmp{ramp_noise}.csv"
                physics_dest = 'data/bestfit/' + os.path.basename(physics_file)
                if os.path.exists(physics_file):
                    shutil.copy2(physics_file, physics_dest)
                    print(f"Physics simulation file copied to: {physics_dest}")
                else:
                    print(f"Warning: Physics simulation file not found: {physics_file}")
            else:
                print(f"Warning: Could not parse KDE filename to find physics file: {kde_basename}")
        
        print(f"\nBest fit results saved to data/bestfit/ folder!")
        print(f"Files created:")
        print(f"   • best_parameters.csv - Best p, q, beta, r, noise, and bandwidth parameters")
        print(f"   • model_vs_human_comparison.csv - Model vs human performance comparison")
        print(f"   • KDE file (copied from kde_results)")
        print(f"   • Physics simulation file (copied from physics_simulations)")
        return True
        
    except Exception as e:
        print(f"Error saving best fit results: {e}")
        return False

def optimize_single_kde_file(kde_file, participant_data, save_results=True, save_to_bestfit=True):
    """Optimize parameters for a single KDE file using participant data."""
    print(f"Optimizing parameters for: {kde_file}")
    
    # Extract parameters from filename
    basename = os.path.basename(kde_file)
    if 'kde_results_' in basename:
        # Extract parameters from filename like "kde_results_blk0.010_rmp0.030_bw20.000_seed1.csv"
        parts = basename.replace('kde_results_', '').replace('.csv', '').split('_')
        block_noise = parts[0].replace('blk', '')
        ramp_noise = parts[1].replace('rmp', '')
        bandwidth = parts[2].replace('bw', '')
        seed = parts[3].replace('seed', '') if len(parts) > 3 else '1'
    else:
        block_noise = 'unknown'
        ramp_noise = 'unknown'
        bandwidth = 'unknown'
        seed = 'unknown'
    
    try:
        # Load KDE results
        kde_results = pd.read_csv(kde_file)
        
        # Optimize parameters with multiple random restarts
        opt_results = optimize_parameters(kde_results, participant_data, seed=int(seed) if seed != 'unknown' else 1, n_restarts=N_RANDOM_RESTARTS)
        
        if opt_results['status'] == 'success':
            print(f"✓ Optimization completed!")
            print(f"  Best parameters: p={opt_results['p_opt']:.4f}, q={opt_results['q_opt']:.4f}, beta={opt_results['beta_opt']:.4f}, r={opt_results['r_opt']:.4f}")
            print(f"  Best Log-Likelihood: {opt_results['log_likelihood']:.6f}")

            # Save results if requested
            if save_results:
                os.makedirs('data/kde_results', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"data/kde_results/optimized_parameters_blk{block_noise}_rmp{ramp_noise}_bw{bandwidth}_seed{seed}_{timestamp}.csv"
                
                # Save optimization results
                opt_df = pd.DataFrame([{
                    'simulation_file': kde_file,
                    'block_noise_sd': block_noise,
                    'ramp_noise_sd': ramp_noise,
                    'bandwidth': bandwidth,
                    'seed': seed,
                    'p_optimized': opt_results['p_opt'],
                    'q_optimized': opt_results['q_opt'],
                    'beta_optimized': opt_results['beta_opt'],
                    'r_optimized': opt_results['r_opt'],
                    'log_likelihood': opt_results['log_likelihood'],
                    'status': opt_results['status']
                }])
                
                opt_df.to_csv(output_file, index=False)
                print(f"  Optimization results saved to: {output_file}")
                
                # Save detailed predictions if available
                if opt_results['predictions'] is not None:
                    predictions_file = f"data/kde_results/optimized_predictions_blk{block_noise}_rmp{ramp_noise}_bw{bandwidth}_seed{seed}_{timestamp}.csv"
                    
                    # Convert predictions to DataFrame
                    pred_records = []
                    for condition, trials in opt_results['predictions'].items():
                        for trial, choices in trials.items():
                            for choice, prob in choices.items():
                                count = participant_data[condition][trial][choice]
                                log_lik_contrib = np.log(prob) * count
                                pred_records.append({
                                    'condition': condition,
                                    'trial': trial,
                                    'choice': choice,
                                    'predicted_probability': prob,
                                    'participant_count': count,
                                    'log_likelihood_contribution': log_lik_contrib
                                })
                    
                    pred_df = pd.DataFrame(pred_records)
                    pred_df.to_csv(predictions_file, index=False)
                    print(f"  Detailed predictions saved to: {predictions_file}")
                
                opt_results['output_file'] = output_file
                opt_results['predictions_file'] = predictions_file if opt_results['predictions'] is not None else None
                
                # Save to bestfit folder (only for single file runs)
                if save_to_bestfit:
                    kde_file_info = {
                        'filename': os.path.basename(kde_file),
                        'block_noise': block_noise,
                        'ramp_noise': ramp_noise,
                        'bandwidth': bandwidth,
                        'seed': seed
                    }
                    save_best_fit_results(opt_results, opt_results['predictions'], participant_data, kde_file_info)

        else:
            print(f"✗ Optimization failed: {opt_results.get('error', 'Unknown error')}")

        # Add metadata
        opt_results.update({
            'kde_file': kde_file,
            'block_noise_sd': block_noise,
            'ramp_noise_sd': ramp_noise,
            'bandwidth': bandwidth,
            'seed': seed
        })

        return opt_results

    except Exception as e:
        print(f"✗ Error processing file: {e}")
        return {
            'status': 'error',
            'kde_file': kde_file,
            'block_noise_sd': block_noise,
            'ramp_noise_sd': ramp_noise,
            'bandwidth': bandwidth,
            'seed': seed,
            'error': str(e)
        }

def optimize_all_kde_files(save_results=True, specific_files=None):
    """Optimize parameters for all KDE files using participant data"""
    print("=== STEP 3: OPTIMIZING BAYESIAN PARAMETERS ===")
    print("Using participant data for log-likelihood maximization (finding BEST fit)")
    
    # Load participant data
    participant_data = load_participant_data()
    
    # Find KDE files to optimize
    if specific_files:
        kde_files = specific_files
        print(f"Optimizing {len(kde_files)} specific KDE files")
    else:
        kde_pattern = "data/kde_results/kde_results_*.csv"
        kde_files = glob.glob(kde_pattern)
        
        if not kde_files:
            print(f"No KDE files found matching pattern: {kde_pattern}")
            print("Please run Steps 1 and 2 first:")
            print("  python step1_physics_simulations.py --all_combinations")
            print("  python step2_run_bayesian_inference.py --all_simulations")
            return {'status': 'no_files', 'results': []}
    
    print(f"Found {len(kde_files)} KDE files to optimize")
    print("Starting optimization with progress tracking...\n")
    
    # Optimize parameters for each file
    results = []
    successful = 0
    failed = 0
    best_overall_result = None
    best_log_likelihood = float('-inf')
    
    start_time = time.time()
    
    for i, kde_file in enumerate(kde_files, 1):
        # Print progress
        print_progress(i-1, len(kde_files), start_time, "Optimizing files", kde_file)
        
        result = optimize_single_kde_file(kde_file, participant_data, save_results=False, save_to_bestfit=False)  # Don't save individual results to bestfit
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
            # Track the best result across all files
            if result['log_likelihood'] > best_log_likelihood:
                best_log_likelihood = result['log_likelihood']
                best_overall_result = result
                print(f" → NEW BEST! Log-likelihood: {best_log_likelihood:.4f}")
        else:
            failed += 1
            print(f" → Failed: {result.get('error', 'Unknown error')}")
    
    # Final progress update
    print_progress(len(kde_files), len(kde_files), start_time, "Optimization")

    # Find best overall result
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['log_likelihood'])
        print("=== BEST OVERALL RESULT ===")
        print(f"Best Log-Likelihood: {best_result['log_likelihood']:.6f}")
        print(f"Best parameters:")
        print(f"  p: {best_result['p_opt']:.4f}")
        print(f"  q: {best_result['q_opt']:.4f}")
        print(f"  beta: {best_result['beta_opt']:.4f}")
        print(f"File: {best_result['kde_file']}")
        print()

        # Save the best result to bestfit folder
        kde_file_info = {
            'filename': os.path.basename(best_result['kde_file']),
            'block_noise': best_result['block_noise_sd'],
            'ramp_noise': best_result['ramp_noise_sd'],
            'bandwidth': best_result['bandwidth'],
            'seed': best_result['seed']
        }
        save_best_fit_results(best_result, best_result['predictions'], participant_data, kde_file_info)

    # No summary file needed - only saving KDE files and bestfit results

    print("=== SUMMARY ===")
    print(f"Total files processed: {len(kde_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(kde_files))*100:.1f}%")

    return {
        'total_files': len(kde_files),
        'successful': successful,
        'failed': failed,
        'success_rate': (successful/len(kde_files))*100,
        'results': results,
        'best_result': best_result if successful_results else None
    }

if __name__ == "__main__":
    optimize_all_kde_files(save_results=True)
