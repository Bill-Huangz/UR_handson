import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_evaluation_results(csv_path, save_dir=None):
    """
    Reads eval.csv and plots rewards, episode lengths, and success rates over steps.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: The file '{csv_path}' does not exist yet. Run your training agent first!")
        return

    # 1. Load the data
    df = pd.read_csv(csv_path)
    
    # Handle optional success rate (convert "N/A" strings back to NaN)
    if 'success_rate' in df.columns:
        df['success_rate'] = pd.to_numeric(df['success_rate'], errors='coerce')

    # 2. Setup the subplots dynamically based on columns present
    has_success = 'success_rate' in df.columns and not df['success_rate'].isna().all()
    num_plots = 3 if has_success else 2
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]  # Ensure it's iterable if only 1 plot exists

    # Subplot 1: Evaluation Rewards
    axes[0].plot(df['step'], df['eval_rewards'], color='tab:blue', linewidth=2, label='Eval Reward')
    axes[0].set_ylabel('Rewards', fontsize=12)
    axes[0].set_title('Evaluation Metrics Over Training Steps', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper left')

    # Subplot 2: Episode Lengths
    axes[1].plot(df['step'], df['eval_lengths'], color='tab:orange', linewidth=2, label='Episode Length')
    axes[1].set_ylabel('Lengths', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper left')

    # Subplot 3: Success Rate (If tracked)
    if has_success:
        axes[2].plot(df['step'], df['success_rate'], color='tab:green', linewidth=2, label='Success Rate')
        axes[2].set_ylabel('Success Rate', fontsize=12)
        axes[2].grid(True, linestyle='--', alpha=0.6)
        axes[2].legend(loc='upper left')

    # Shared X-axis label
    axes[-1].set_xlabel('Training Steps', fontsize=12)
    
    plt.tight_layout()

    # 3. Save the plot
    if save_dir is None:
        save_dir = os.path.dirname(csv_path) or '.'
    
    save_path = os.path.join(save_dir, 'eval_performance_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot successfully saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Point this to your actual evaluation CSV file path
    CSV_FILE_PATH = "./log/2leg_cheetah/SAC/0/eval.csv"
    
    plot_evaluation_results(CSV_FILE_PATH)