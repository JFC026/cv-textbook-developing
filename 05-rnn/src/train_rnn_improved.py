"""
train_rnn_improved.py - ImprovedRNN训练脚本（修正版）
修正训练过程中的问题，但保持文件名不变
"""

import numpy as np
import os
import sys
import json
import time
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rnn_improved import ImprovedRNN
from src.rnn_original import create_batches
from src.text_loader import load_text_data
from utils.visualization import plot_loss_curve

def train_improved_rnn(text, char_to_ix, ix_to_char, config):
    """
    训练改进版RNN模型（修正）
    """
    vocab_size = len(char_to_ix)
    hidden_size = config['model']['hidden_size']
    seq_length = config['data']['seq_length']
    batch_size = config['training']['batch_size']
    learning_rate = config['model']['learning_rate']
    epochs = config['training']['epochs']
    print_every = config['training']['print_every']
    sample_every = config['training']['sample_every']
    sample_length = config['training']['sample_length']
    
    dropout_rate = config['improved'].get('dropout_rate', 0.1)
    use_lr_schedule = config['improved'].get('use_lr_schedule', True)
    top_k_sampling = config['improved'].get('top_k_sampling', 5)
    warmup_epochs = config['improved'].get('warmup_epochs', 5)
    
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    train_idx = int(len(text) * train_split)
    val_idx = train_idx + int(len(text) * val_split)
    
    train_text = text[:train_idx]
    val_text = text[train_idx:val_idx]
    test_text = text[val_idx:]
    
    train_data = [char_to_ix[ch] for ch in train_text]
    val_data = [char_to_ix[ch] for ch in val_text]
    test_data = [char_to_ix[ch] for ch in test_text]
    
    print(f"训练集: {len(train_data)} 字符")
    print(f"验证集: {len(val_data)} 字符")
    print(f"测试集: {len(test_data)} 字符")
    
    print("初始化ImprovedRNN模型（修正版）...")
    print(f"初始学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    print(f"Dropout率: {dropout_rate}")
    print(f"学习率调度: {'warmup+余弦退火' if use_lr_schedule else '无'}")
    print(f"Top-k采样: k={top_k_sampling}")
    
    rnn = ImprovedRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        seq_length=seq_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )
    
    patience = config['training']['patience']
    min_delta = config['training']['min_delta']
    early_stopping_enabled = config['training']['early_stopping']
    
    loss_history = []
    train_losses = []
    val_losses = []
    learning_rates = []
    
    best_loss = float('inf')
    best_perplexity = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    early_stopped = False
    
    print(f"开始训练 ({epochs} 个epoch)...")
    print(f"每个epoch的批次数: ~{len(train_data) // (seq_length * batch_size)}")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        if use_lr_schedule:
            rnn.update_learning_rate(epoch, epochs, warmup_epochs)
        
        h_prev = np.zeros((hidden_size, batch_size))
        batch_generator = create_batches(train_data, seq_length, batch_size)
        
        epoch_loss = 0
        n_batches = 0
        
        for inputs_batch, targets_batch in batch_generator:
            loss, h_prev = rnn.train_step(inputs_batch, targets_batch, h_prev)
            epoch_loss += loss
            n_batches += 1
            loss_history.append(loss)
        
        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0
        train_losses.append(avg_train_loss)
        train_perplexity = np.exp(avg_train_loss)
        
        val_loss = evaluate_improved_model(rnn, val_data, seq_length, batch_size)
        val_perplexity = np.exp(val_loss)
        val_losses.append(val_loss)
        
        learning_rates.append(rnn.learning_rate)
        
        if early_stopping_enabled:
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_perplexity = val_perplexity
                best_epoch = epoch + 1
                patience_counter = 0
                
                best_model_state = {
                    'Wxh': rnn.Wxh.copy(),
                    'Whh': rnn.Whh.copy(),
                    'Why': rnn.Why.copy(),
                    'bh': rnn.bh.copy(),
                    'by': rnn.by.copy(),
                    'mWxh': rnn.mWxh.copy(),
                    'mWhh': rnn.mWhh.copy(),
                    'mWhy': rnn.mWhy.copy(),
                    'mbh': rnn.mbh.copy(),
                    'mby': rnn.mby.copy(),
                    'vWxh': rnn.vWxh.copy(),
                    'vWhh': rnn.vWhh.copy(),
                    'vWhy': rnn.vWhy.copy(),
                    'vbh': rnn.vbh.copy(),
                    'vby': rnn.vby.copy(),
                    't': rnn.t
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stopped = True
        
        if epoch % print_every == 0 or epoch == epochs - 1 or early_stopped:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Perp: {train_perplexity:.2f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Perp: {val_perplexity:.2f} | "
                  f"LR: {rnn.learning_rate:.6f} | "
                  f"Time: {elapsed:.1f}s")
        
        if (epoch % sample_every == 0 and epoch > 0) or early_stopped:
            if early_stopped:
                print("触发早停！")
                print(f"最佳验证损失: {best_loss:.4f}")
                print(f"最佳验证困惑度: {best_perplexity:.2f}")
            
            print("生成样本文本:")
            h_sample = np.zeros((hidden_size, 1))
            seed_ix = np.random.randint(0, vocab_size)
            sample_ix = rnn.sample(h_sample, seed_ix, 150, 
                                   temperature=0.7, top_k=top_k_sampling)
            sample_text = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f"{sample_text[:80]}...")
            print()
        
        if early_stopped:
            break
    
    total_time = time.time() - start_time
    final_epoch = min(epoch + 1, epochs)
    
    if best_model_state is not None:
        rnn.Wxh = best_model_state['Wxh']
        rnn.Whh = best_model_state['Whh']
        rnn.Why = best_model_state['Why']
        rnn.bh = best_model_state['bh']
        rnn.by = best_model_state['by']
        rnn.mWxh = best_model_state['mWxh']
        rnn.mWhh = best_model_state['mWhh']
        rnn.mWhy = best_model_state['mWhy']
        rnn.mbh = best_model_state['mbh']
        rnn.mby = best_model_state['mby']
        rnn.vWxh = best_model_state['vWxh']
        rnn.vWhh = best_model_state['vWhh']
        rnn.vWhy = best_model_state['vWhy']
        rnn.vbh = best_model_state['vbh']
        rnn.vby = best_model_state['vby']
        rnn.t = best_model_state['t']
    
    print("在测试集上最终评估...")
    test_loss = evaluate_improved_model(rnn, test_data, seq_length, batch_size)
    test_perplexity = np.exp(test_loss)
    
    print("\n" + "="*70)
    print("改进版RNN训练完成（修正版）")
    print("="*70)
    print(f"总训练时间: {total_time/60:.2f} 分钟")
    print(f"训练轮数: {final_epoch}/{epochs}")
    print(f"早停触发: {'是' if early_stopped else '否'}")
    print(f"最佳epoch: {best_epoch}")
    print(f"训练集最终损失: {avg_train_loss:.4f}")
    print(f"验证集最佳损失: {best_loss:.4f}")
    print(f"验证集最佳困惑度: {best_perplexity:.2f}")
    print(f"测试集最终困惑度: {test_perplexity:.2f}")
    print("="*70)
    
    full_history = {
        'loss_history': loss_history,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_val_loss': float(best_loss),
        'test_perplexity': float(test_perplexity),
        'best_epoch': best_epoch,
        'epochs_trained': final_epoch,
        'early_stopped': early_stopped,
        'total_time': float(total_time),
        'improved_params': {
            'dropout_rate': dropout_rate,
            'use_lr_schedule': use_lr_schedule,
            'top_k_sampling': top_k_sampling,
            'warmup_epochs': warmup_epochs
        }
    }
    
    return rnn, full_history, best_loss, test_perplexity, best_epoch, early_stopped, total_time

def evaluate_improved_model(rnn, data, seq_length, batch_size):
    """
    评估改进版模型（不使用Dropout）
    """
    eval_batch_size = min(batch_size, 8)
    batch_generator = create_batches(data, seq_length, eval_batch_size)
    
    total_loss = 0
    n_batches = 0
    
    h_prev = np.zeros((rnn.hidden_size, eval_batch_size))
    
    for inputs_batch, targets_batch in batch_generator:
        xs, hs, ys, ps, _ = rnn.forward(inputs_batch, h_prev, training=False)
        loss = rnn.compute_loss(ps, targets_batch)
        total_loss += loss
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else float('inf')

def main():
    """
    主训练函数（修正版）
    """
    print("="*70)
    print("ImprovedRNN 字符级语言模型训练（修正版）")
    print("="*70)
    
    import yaml
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    config_path = os.path.join(project_root, 'config', 'rnn_improved_config.yaml')
    
    print(f"加载配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置加载成功")
        
        print(f"Epochs: {config['training']['epochs']}")
        print(f"Learning Rate: {config['model']['learning_rate']}")
        print(f"Batch Size: {config['training']['batch_size']}")
        print(f"Dropout Rate: {config['improved'].get('dropout_rate', 0.1)}")
        
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        print("使用默认配置")
        config = {
            'data': {
                'data_dir': '../data',
                'seq_length': 25,
                'train_split': 0.85,
                'val_split': 0.10
            },
            'model': {
                'hidden_size': 128,
                'learning_rate': 0.001,
                'clip_grad': 5.0
            },
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'print_every': 5,
                'sample_every': 10,
                'sample_length': 200,
                'random_seed': 42,
                'early_stopping': True,
                'patience': 10,
                'min_delta': 0.001
            },
            'improved': {
                'dropout_rate': 0.1,
                'use_lr_schedule': True,
                'top_k_sampling': 5,
                'warmup_epochs': 5
            },
            'results': {
                'save_dir': '../results',
                'model_save_path': '../results/rnn_model_improved.pkl',
                'plots_dir': '../results/plots'
            }
        }
    
    np.random.seed(config['training']['random_seed'])
    
    print("加载文本数据...")
    
    data_dir = config['data']['data_dir']
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    
    data_path = os.path.join(data_dir, 'input.txt')
    print(f"数据文件路径: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    text, chars, char_to_ix, ix_to_char = load_text_data(data_path)
    print(f"数据集大小: {len(text)} 字符")
    print(f"词汇表大小: {len(chars)} 字符")
    
    print("训练ImprovedRNN模型（修正版）...")
    rnn, full_history, best_loss, test_perplexity, best_epoch, early_stopped, total_time = train_improved_rnn(
        text, char_to_ix, ix_to_char, config
    )
    
    print("保存最终模型...")
    os.makedirs(config['results']['save_dir'], exist_ok=True)
    model_path = config['results']['model_save_path']
    rnn.save_model(model_path)
    
    mapping_path = os.path.join(config['results']['save_dir'], 'char_mapping_improved.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'char_to_ix': char_to_ix,
            'ix_to_char': {int(k): v for k, v in ix_to_char.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"字符映射已保存到: {mapping_path}")
    
    history_path = os.path.join(config['results']['save_dir'], 'training_history_improved.json')
    with open(history_path, 'w') as f:
        json.dump(full_history, f, indent=2)
    print(f"训练历史已保存到: {history_path}")
    
    print("生成可视化...")
    os.makedirs(config['results']['plots_dir'], exist_ok=True)
    
    plot_loss_curve(
        full_history['train_losses'],
        val_losses=full_history['val_losses'],
        learning_rates=full_history.get('learning_rates', []),
        save_path=os.path.join(config['results']['plots_dir'], 'loss_curve_improved.png'),
        title='ImprovedRNN Training Loss (Fixed)'
    )
    
    print("生成最终样本文本...")
    h = np.zeros((rnn.hidden_size, 1))
    seed_ix = np.random.randint(0, len(char_to_ix))
    
    print("标准采样 (temperature=0.7):")
    sample_ix_standard = rnn.sample(h.copy(), seed_ix, 200, temperature=0.7, top_k=None)
    sample_standard = ''.join(ix_to_char[ix] for ix in sample_ix_standard)
    print(f"{sample_standard[:80]}...")
    
    print("Top-k采样 (k=5, temperature=0.7):")
    sample_ix_topk = rnn.sample(h.copy(), seed_ix, 200, temperature=0.7, top_k=5)
    sample_topk = ''.join(ix_to_char[ix] for ix in sample_ix_topk)
    print(f"{sample_topk[:80]}...")
    
    sample_path = os.path.join(config['results']['save_dir'], 'generated_sample_improved.txt')
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("=== Standard Sampling (temperature=0.7) ===\n")
        f.write(sample_standard + "\n\n")
        f.write("=== Top-k Sampling (k=5, temperature=0.7) ===\n")
        f.write(sample_topk + "\n")
    print(f"样本文本已保存到: {sample_path}")
    
    print("\n" + "="*70)
    print("训练结果总结（修正版）:")
    print("="*70)
    print(f"最佳训练轮数: {best_epoch}")
    print(f"总训练时间: {total_time/60:.1f} 分钟")
    print(f"早停触发: {'是' if early_stopped else '否'}")
    print(f"最佳验证困惑度: {np.exp(best_loss):.2f}")
    print(f"测试集困惑度: {test_perplexity:.2f}")
    print(f"最终学习率: {rnn.learning_rate:.6f}")
    print(f"使用Dropout率: {rnn.dropout_rate}")
    print("="*70)
    
    print("ImprovedRNN修正版训练完成！")

if __name__ == '__main__':
    main()