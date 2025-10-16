import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_graphs():
    """
    Create graphs comparing model predictions vs human data by trial.
    Input: data/bestfit/model_vs_human_comparison.csv
    Output: figures/model_vs_human_predictions.png
    """
    
    # Read the model vs human comparison data
    try:
        df = pd.read_csv('data/bestfit/model_vs_human_comparison.csv')
        print(f"Loaded {len(df)} trials from data/bestfit/model_vs_human_comparison.csv")
        
        # Print summary statistics
        print(f"Total trials: {len(df)}")
        print(f"Conditions: {df['condition'].unique()}")
        print(f"Trials: {df['trial'].unique()}")
        
    except FileNotFoundError:
        print("Error: data/bestfit/model_vs_human_comparison.csv not found.")
        print("Please run the parameter fitting first to generate bestfit results.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Get unique trials and conditions
    trials = df['trial'].unique()
    conditions = df['condition'].unique()
    
    print(f"Found {len(trials)} unique trials: {trials}")
    print(f"Found {len(conditions)} unique conditions: {conditions}")
    
    # Create a figure with subplots for each trial
    fig, axes = plt.subplots(2, len(trials), figsize=(4*len(trials), 8))
    if len(trials) == 1:
        axes = axes.reshape(2, 1)
    
    # Colors for visualization
    human_color = 'royalblue'
    model_color = 'red'
    positions = ['Choice 1', 'Choice 2', 'Choice 3', 'Choice 4']
    
    # Create output directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    for i, trial in enumerate(trials):
        trial_data = df[df['trial'] == trial]
        
        # Forward condition
        forward_data = trial_data[trial_data['condition'] == 'forward_ramp_condition']
        if not forward_data.empty:
            ax_forward = axes[0, i]
            
            # Get model predictions (multiply by 100 for percentage)
            model_predictions = [
                forward_data['choice_1_prediction'].iloc[0] * 100,
                forward_data['choice_2_prediction'].iloc[0] * 100,
                forward_data['choice_3_prediction'].iloc[0] * 100,
                forward_data['choice_4_prediction'].iloc[0] * 100
            ]
            
            # Get human data (multiply by 100 for percentage)
            human_data = [
                forward_data['choice_1_participant_data'].iloc[0] * 100,
                forward_data['choice_2_participant_data'].iloc[0] * 100,
                forward_data['choice_3_participant_data'].iloc[0] * 100,
                forward_data['choice_4_participant_data'].iloc[0] * 100
            ]
            
            # Create human data bars
            bars_human = ax_forward.bar(positions, human_data, color=human_color, alpha=0.7, label='Human Data')
            
            # Create model predictions as red dots
            ax_forward.plot(positions, model_predictions, 'o', color=model_color, 
                           markersize=8, label='Model Predictions')
            
            # Set title
            ax_forward.set_title(f'{trial.replace("_", " ").title()} - Forward', 
                               fontsize=10, fontweight='bold')
            ax_forward.set_ylabel('Percentage Selected (%)')
            ax_forward.set_ylim(0, 100)
            ax_forward.tick_params(axis='x', rotation=45)
            ax_forward.legend()
            ax_forward.grid(True, alpha=0.3)
        
        # Backward condition
        backward_data = trial_data[trial_data['condition'] == 'backward_ramp_condition']
        if not backward_data.empty:
            ax_backward = axes[1, i]
            
            # Get model predictions (multiply by 100 for percentage)
            model_predictions = [
                backward_data['choice_1_prediction'].iloc[0] * 100,
                backward_data['choice_2_prediction'].iloc[0] * 100,
                backward_data['choice_3_prediction'].iloc[0] * 100,
                backward_data['choice_4_prediction'].iloc[0] * 100
            ]
            
            # Get human data (multiply by 100 for percentage)
            human_data = [
                backward_data['choice_1_participant_data'].iloc[0] * 100,
                backward_data['choice_2_participant_data'].iloc[0] * 100,
                backward_data['choice_3_participant_data'].iloc[0] * 100,
                backward_data['choice_4_participant_data'].iloc[0] * 100
            ]
            
            # Create human data bars
            bars_human = ax_backward.bar(positions, human_data, color=human_color, alpha=0.7, label='Human Data')
            
            # Create model predictions as red dots
            ax_backward.plot(positions, model_predictions, 'o', color=model_color, 
                            markersize=8, label='Model Predictions')
            
            # Set title
            ax_backward.set_title(f'{trial.replace("_", " ").title()} - Backward', 
                                fontsize=10, fontweight='bold')
            ax_backward.set_ylabel('Percentage Selected (%)')
            ax_backward.set_ylim(0, 100)
            ax_backward.tick_params(axis='x', rotation=45)
            ax_backward.legend()
            ax_backward.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Model vs Human Predictions by Trial and Condition', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'figures/model_vs_human_predictions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Overall correlation statistics:")
    print(f"  Mean correlation: {df['correlation_predicted_vs_human'].mean():.4f}")
    print(f"  Std correlation: {df['correlation_predicted_vs_human'].std():.4f}")
    print(f"  Min correlation: {df['correlation_predicted_vs_human'].min():.4f}")
    print(f"  Max correlation: {df['correlation_predicted_vs_human'].max():.4f}")
    
    print(f"\nModel-Human match statistics:")
    matches = df['model_human_match'].sum()
    total = len(df)
    print(f"  Matches: {matches}/{total} ({matches/total*100:.1f}%)")
    
    print(f"\nBy condition:")
    for condition in conditions:
        cond_data = df[df['condition'] == condition]
        cond_matches = cond_data['model_human_match'].sum()
        cond_total = len(cond_data)
        cond_corr = cond_data['correlation_predicted_vs_human'].mean()
        print(f"  {condition}: {cond_matches}/{cond_total} matches ({cond_matches/cond_total*100:.1f}%), avg correlation: {cond_corr:.3f}")

if __name__ == "__main__":
    print("Creating model vs human prediction graphs...")
    create_graphs()
