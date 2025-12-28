"""
compare_models.py - 原始与改进RNN模型对比分析
"""

import numpy as np
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def load_training_log(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def compare_training_results():
    print("对比分析原始与改进RNN训练结果")
    print("="*70)
    
    original_log = load_training_log('results/training_history_original.json')
    improved_log = load_training_log('results/training_history_improved.json')
    
    if not original_log:
        print("缺少原始模型训练记录")
        return
    
    if not improved_log:
        print("缺少改进模型训练记录")
        return
    
    print("\n关键指标对比:")
    print("-" * 60)
    print(f"{'指标':<25} {'原始RNN':<15} {'改进RNN':<15} {'变化':<10}")
    print("-" * 60)
    
    comparison_items = [
        ('最优验证损失', 'best_val_loss', '越低越好'),
        ('测试复杂度', 'test_perplexity', '越低越好'),
        ('最优训练轮次', 'best_epoch', ''),
        ('总训练时间(分钟)', 'total_time', '/60'),
        ('提前终止', 'early_stopped', ''),
        ('实际训练轮次', 'epochs_trained', '')
    ]
    
    for item_name, key, note in comparison_items:
        orig_value = original_log.get(key, '未记录')
        impr_value = improved_log.get(key, '未记录')
        
        if key == 'total_time' and isinstance(orig_value, (int, float)) and isinstance(impr_value, (int, float)):
            orig_value = orig_value / 60
            impr_value = impr_value / 60
        
        change_rate = '未计算'
        if isinstance(orig_value, (int, float)) and isinstance(impr_value, (int, float)):
            if key in ['best_val_loss', 'test_perplexity']:
                if orig_value > 0:
                    change_pct = ((impr_value - orig_value) / orig_value) * 100
                    change_rate = f"{change_pct:+.1f}%"
            elif key == 'total_time':
                if orig_value > 0:
                    change_pct = ((impr_value - orig_value) / orig_value) * 100
                    change_rate = f"{change_pct:+.1f}%"
        
        if isinstance(orig_value, float):
            orig_display = f"{orig_value:.4f}" if '损失' in item_name else f"{orig_value:.2f}"
        else:
            orig_display = str(orig_value)
            
        if isinstance(impr_value, float):
            impr_display = f"{impr_value:.4f}" if '损失' in item_name else f"{impr_value:.2f}"
        else:
            impr_display = str(impr_value)
        
        print(f"{item_name:<25} {orig_display:<15} {impr_display:<15} {change_rate:<10}")
    
    create_comparison_plots(original_log, improved_log)
    
    if 'improved_params' in improved_log:
        print("\n改进模型参数配置:")
        print("-" * 40)
        for param_name, param_value in improved_log['improved_params'].items():
            print(f"  {param_name}: {param_value}")

def create_comparison_plots(original_log, improved_log):
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练损失对比
    ax1 = axes[0, 0]
    if 'train_losses' in original_log:
        orig_loss = original_log['train_losses']
        ax1.plot(range(1, len(orig_loss)+1), orig_loss, 
                label='原始RNN', alpha=0.7, linewidth=2, color='blue')
    
    if 'train_losses' in improved_log:
        impr_loss = improved_log['train_losses']
        ax1.plot(range(1, len(impr_loss)+1), impr_loss, 
                label='改进RNN', alpha=0.7, linewidth=2, color='red')
    
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('训练损失')
    ax1.set_title('训练损失对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证损失对比
    ax2 = axes[0, 1]
    if 'val_losses' in original_log:
        orig_val = original_log['val_losses']
        ax2.plot(range(1, len(orig_val)+1), orig_val, 
                label='原始RNN', alpha=0.7, linewidth=2, color='blue')
    
    if 'val_losses' in improved_log:
        impr_val = improved_log['val_losses']
        ax2.plot(range(1, len(impr_val)+1), impr_val, 
                label='改进RNN', alpha=0.7, linewidth=2, color='red')
    
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('验证损失')
    ax2.set_title('验证损失对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率变化
    ax3 = axes[1, 0]
    if 'learning_rates' in original_log and len(original_log['learning_rates']) > 0:
        orig_lr = original_log['learning_rates']
        ax3.plot(range(1, len(orig_lr)+1), orig_lr, 
                label='原始RNN', alpha=0.7, linewidth=2, color='blue')
    
    if 'learning_rates' in improved_log and len(improved_log['learning_rates']) > 0:
        impr_lr = improved_log['learning_rates']
        ax3.plot(range(1, len(impr_lr)+1), impr_lr, 
                label='改进RNN', alpha=0.7, linewidth=2, color='red')
    
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('学习率')
    ax3.set_title('学习率变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 复杂度对比
    ax4 = axes[1, 1]
    if 'val_losses' in original_log:
        orig_perp = np.exp(np.array(original_log['val_losses']))
        ax4.plot(range(1, len(orig_perp)+1), orig_perp, 
                label='原始RNN', alpha=0.7, linewidth=2, color='blue')
    
    if 'val_losses' in improved_log:
        impr_perp = np.exp(np.array(improved_log['val_losses']))
        ax4.plot(range(1, len(impr_perp)+1), impr_perp, 
                label='改进RNN', alpha=0.7, linewidth=2, color='red')
    
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('复杂度')
    ax4.set_title('验证复杂度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_dir = os.path.join('results', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图表已保存至: {save_path}")

def compare_evaluation_metrics():
    print("\n评估结果对比分析")
    print("="*70)
    
    orig_eval_path = os.path.join('results', 'evaluation_original_comprehensive.json')
    impr_eval_path = os.path.join('results', 'evaluation_improved_comprehensive.json')
    
    orig_eval = load_training_log(orig_eval_path)
    impr_eval = load_training_log(impr_eval_path)
    
    if not orig_eval:
        print("缺少原始模型评估结果")
        return
    
    if not impr_eval:
        print("缺少改进模型评估结果")
        return
    
    print("\n性能指标对比:")
    print("-" * 65)
    print(f"{'指标':<30} {'原始RNN':<15} {'改进RNN':<15} {'变化':<10}")
    print("-" * 65)
    
    if 'loss_and_perplexity' in orig_eval and 'loss_and_perplexity' in impr_eval:
        orig_loss = orig_eval['loss_and_perplexity'].get('test_loss', '未记录')
        impr_loss = impr_eval['loss_and_perplexity'].get('test_loss', '未记录')
        
        if isinstance(orig_loss, float) and isinstance(impr_loss, float):
            change_pct = ((impr_loss - orig_loss) / orig_loss) * 100 if orig_loss > 0 else '未计算'
            if isinstance(change_pct, float):
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = '未计算'
            
            print(f"{'测试损失':<30} {orig_loss:.4f} {'':<8} {impr_loss:.4f} {'':<8} {change_str:<10}")
        
        orig_perp = orig_eval['loss_and_perplexity'].get('test_perplexity', '未记录')
        impr_perp = impr_eval['loss_and_perplexity'].get('test_perplexity', '未记录')
        
        if isinstance(orig_perp, float) and isinstance(impr_perp, float):
            change_pct = ((impr_perp - orig_perp) / orig_perp) * 100 if orig_perp > 0 else '未计算'
            if isinstance(change_pct, float):
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = '未计算'
            
            print(f"{'测试复杂度':<30} {orig_perp:.2f} {'':<8} {impr_perp:.2f} {'':<8} {change_str:<10}")
    
    if 'inference_speed' in orig_eval and 'inference_speed' in impr_eval:
        orig_speed = orig_eval['inference_speed'].get('chars_per_second', 0)
        impr_speed = impr_eval['inference_speed'].get('chars_per_second', 0)
        
        if orig_speed and impr_speed:
            change_pct = ((impr_speed - orig_speed) / orig_speed) * 100 if orig_speed > 0 else '未计算'
            if isinstance(change_pct, float):
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = '未计算'
            
            print(f"{'推理速度(字符/秒)':<30} {orig_speed:.0f} {'':<8} {impr_speed:.0f} {'':<8} {change_str:<10}")

def main():
    print("="*70)
    print("        RNN模型对比分析报告")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    compare_training_results()
    compare_evaluation_metrics()
    
    print("\n" + "="*70)
    print("对比分析完成")
    print("="*70)

if __name__ == '__main__':
    main()