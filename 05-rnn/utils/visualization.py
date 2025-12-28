import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_curve(train_losses, val_losses=None, learning_rates=None,
                   save_path='loss_curve.png', title='Training Loss'):
    
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if learning_rates is not None and len(learning_rates) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, np.exp(np.array(train_losses)), 'b--', alpha=0.5, label='Train Perplexity')
    if val_losses is not None:
        ax1_twin.plot(epochs, np.exp(np.array(val_losses)), 'r--', alpha=0.5, label='Val Perplexity')
    ax1_twin.set_ylabel('Perplexity')
    ax1_twin.legend(loc='upper right')
    
    if learning_rates is not None and len(learning_rates) > 0:
        ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curve saved to: {save_path}")

def plot_sampling_comparison(samples_dict, save_path='sampling_comparison.png'):
    
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(len(samples_dict), 1, figsize=(12, 3*len(samples_dict)))
    
    if len(samples_dict) == 1:
        axes = [axes]
    
    for idx, (strategy_name, metrics) in enumerate(samples_dict.items()):
        ax = axes[idx]
        
        diversity = metrics.get('avg_diversity', 0)
        coherence = metrics.get('avg_coherence', 0)
        repetition = metrics.get('avg_repetition', 0)
        
        x_labels = ['Diversity', 'Coherence', 'Repetition Rate']
        values = [diversity, coherence, repetition]
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        
        bars = ax.bar(x_labels, values, color=colors, alpha=0.8)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title(f'{strategy_name} Sampling Strategy')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sampling strategy comparison saved to: {save_path}")

def plot_dropout_analysis(dropout_results, save_path='dropout_analysis.png'):
    
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['No Dropout', 'With Dropout']
    losses = [dropout_results['without_dropout']['loss'],
              dropout_results['with_dropout']['loss']]
    perplexities = [dropout_results['without_dropout']['perplexity'],
                    dropout_results['with_dropout']['perplexity']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, losses, width, label='Loss', color='#FF5722', alpha=0.8)
    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Dropout Effect on Loss')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    ax2.bar(x - width/2, perplexities, width, label='Perplexity', color='#9C27B0', alpha=0.8)
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Dropout Effect on Perplexity')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    
    loss_diff = losses[1] - losses[0]
    perp_diff = perplexities[1] - perplexities[0]
    
    ax1.text(1, losses[1] + 0.01, f'Δ={loss_diff:.3f}', ha='center')
    ax2.text(1, perplexities[1] + 0.5, f'Δ={perp_diff:.2f}', ha='center')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dropout analysis saved to: {save_path}")