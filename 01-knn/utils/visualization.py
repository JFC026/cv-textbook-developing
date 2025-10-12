import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

def plot_accuracy_vs_k(k_scores, save_path=None):
    """
    Plot accuracy vs K value relationship
    """
    k_values = list(k_scores.keys())
    accuracies = [k_scores[k] * 100 for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('K Value', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('K Value vs Classification Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Mark best K value
    best_k = max(k_scores.items(), key=lambda x: x[1])[0]
    best_acc = k_scores[best_k] * 100
    plt.plot(best_k, best_acc, 'ro', markersize=10, label=f'Best K: {best_k}')
    plt.legend()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Accuracy vs K plot saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_nearest_neighbors(classifier, X_test, y_test, n_examples=5, save_path=None):
    """
    Visualize test samples and their nearest neighbors
    """
    # Randomly select some test samples
    indices = np.random.choice(len(X_test), n_examples, replace=False)
    
    fig = plt.figure(figsize=(15, 3 * n_examples))
    gs = gridspec.GridSpec(n_examples, 6)  # Each row: test sample + 5 nearest neighbors
    
    for i, idx in enumerate(indices):
        # Get test sample
        test_sample = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        
        # Calculate distances and find nearest neighbors
        dists = classifier.compute_distances_no_loops(X_test[idx:idx+1])
        nearest_indices = np.argsort(dists[0])[:5]
        
        # Plot test sample
        ax_test = plt.subplot(gs[i, 0])
        ax_test.imshow(test_sample, cmap='gray')
        ax_test.set_title(f'Test Sample\nTrue: {true_label}', fontsize=10)
        ax_test.axis('off')
        
        # Plot 5 nearest neighbors
        for j, neighbor_idx in enumerate(nearest_indices):
            neighbor_sample = classifier.X_train[neighbor_idx].reshape(28, 28)
            neighbor_label = classifier.y_train[neighbor_idx]
            distance = dists[0, neighbor_idx]
            
            ax_neighbor = plt.subplot(gs[i, j+1])
            ax_neighbor.imshow(neighbor_sample, cmap='gray')
            ax_neighbor.set_title(f'Neighbor {j+1}\nLabel: {neighbor_label}\nDist: {distance:.2f}', fontsize=9)
            ax_neighbor.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Nearest neighbors visualization saved to: {save_path}")
    
    plt.show()

def plot_class_accuracy(y_true, y_pred, save_path=None):
    """
    Plot accuracy for each class
    """
    class_accuracies = []
    for i in range(10):
        mask = y_true == i
        if np.sum(mask) > 0:
            accuracy = np.mean(y_pred[mask] == y_true[mask]) * 100
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(10), class_accuracies, color='skyblue', alpha=0.7)
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Classification Accuracy by Class', fontsize=14)
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Class accuracy plot saved to: {save_path}")
    
    plt.show()