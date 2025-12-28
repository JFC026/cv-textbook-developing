"""
train_rnn_original.py - SimpleRNN训练脚本（支持批处理）

功能包括:
- 批处理数据加载（85%训练，10%验证，5%测试）
- 批处理训练RNN模型（Adam优化器）
- 早停机制（耐心值10，最小改进0.001）
- 学习率调度
- 保存模型和训练历史
- 生成样本文本
"""

import numpy as np
import os
import sys
import json
import time
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rnn_original import SimpleRNN, create_batches
from src.text_loader import load_text_data
from utils.visualization import plot_loss_curve

def train_rnn(text, char_to_ix, ix_to_char, config):
    """
    训练支持批处理的SimpleRNN模型
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
    clip_grad = config['model'].get('clip_grad', 5.0)
    
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
    
    print(f"训练集: {len(train_data)} 字符 ({train_split*100:.0f}%)")
    print(f"验证集: {len(val_data)} 字符 ({val_split*100:.0f}%)")
    print(f"测试集: {len(test_data)} 字符 ({100-(train_split+val_split)*100:.0f}%)")
    
    print("初始化SimpleRNN模型（支持批处理）...")
    print(f"使用学习率: {learning_rate} (Adam优化器)")
    print(f"批次大小: {batch_size}")
    print(f"梯度裁剪: {clip_grad}")
    
    rnn = SimpleRNN(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        seq_length=seq_length,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    print(f"词汇表大小: {vocab_size}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"序列长度: {seq_length}")
    
    patience = config['training']['patience']
    min_delta = config['training']['min_delta']
    early_stopping_enabled = config['training']['early_stopping']
    
    lr_decay = config['training'].get('lr_decay', 0.95)
    lr_decay_every = config['training'].get('lr_decay_every', 10)
    
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
    print(f"早停耐心值: {patience}, 最小改进: {min_delta}")
    print("="*70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
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
        
        val_loss = evaluate_model(rnn, val_data, seq_length, batch_size)
        val_perplexity = np.exp(val_loss)
        val_losses.append(val_loss)
        
        learning_rates.append(rnn.learning_rate)
        
        if (epoch + 1) % lr_decay_every == 0:
            rnn.learning_rate *= lr_decay
        
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
                
                print(f"[BEST] 验证损失: {best_loss:.4f}, 困惑度: {best_perplexity:.2f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stopped = True
                    print(f"触发早停！")
                    print(f"当前epoch: {epoch+1}")
                    print(f"最佳验证损失: {best_loss:.4f}")
                    print(f"最佳验证困惑度: {best_perplexity:.2f}")
                    print(f"已连续 {patience} 个epoch无改善")
                    break
        
        if epoch % print_every == 0 or epoch == epochs - 1 or early_stopped:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Perp: {train_perplexity:.2f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Perp: {val_perplexity:.2f} | "
                  f"LR: {rnn.learning_rate:.6f} | "
                  f"耗时: {elapsed:.1f}s")
        
        if (epoch % sample_every == 0 and epoch > 0) or early_stopped:
            print("生成样本文本:")
            h_sample = np.zeros((hidden_size, 1))
            seed_ix = np.random.randint(0, vocab_size)
            sample_ix = rnn.sample(h_sample, seed_ix, 200, temperature=0.7)
            sample_text = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f"{sample_text[:100]}...")
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
    test_loss = evaluate_model(rnn, test_data, seq_length, batch_size)
    test_perplexity = np.exp(test_loss)
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"总耗时: {total_time/60:.2f} 分钟")
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
        'total_time': float(total_time)
    }
    
    return rnn, full_history, best_loss, test_perplexity, best_epoch, early_stopped, total_time

def evaluate_model(rnn, data, seq_length, batch_size):
    """
    评估模型（不使用训练模式）
    """
    eval_batch_size = min(batch_size, 8)
    batch_generator = create_batches(data, seq_length, eval_batch_size)
    
    total_loss = 0
    n_batches = 0
    
    h_prev = np.zeros((rnn.hidden_size, eval_batch_size))
    
    for inputs_batch, targets_batch in batch_generator:
        xs, hs, ys, ps, _ = rnn.forward(inputs_batch, h_prev)
        loss = rnn.compute_loss(ps, targets_batch)
        total_loss += loss
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else float('inf')

def main():
    """
    主训练函数（支持批处理）
    """
    print("="*70)
    print("SimpleRNN 字符级语言模型训练（批处理版本）")
    print("="*70)
    
    import yaml
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    config_path = os.path.join(project_root, 'config', 'rnn_original_config.yaml')
    
    print(f"加载配置文件: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置加载成功")
        
        if 'training' not in config:
            config['training'] = {}
        if 'batch_size' not in config['training']:
            config['training']['batch_size'] = 32
        if 'patience' not in config['training']:
            config['training']['patience'] = 10
        if 'min_delta' not in config['training']:
            config['training']['min_delta'] = 0.001
        if 'early_stopping' not in config['training']:
            config['training']['early_stopping'] = True
        if 'lr_decay' not in config['training']:
            config['training']['lr_decay'] = 0.95
        if 'lr_decay_every' not in config['training']:
            config['training']['lr_decay_every'] = 10
        
        if 'data' not in config:
            config['data'] = {}
        if 'data_dir' not in config['data']:
            config['data']['data_dir'] = '../data'
        if 'train_split' not in config['data']:
            config['data']['train_split'] = 0.85
        if 'val_split' not in config['data']:
            config['data']['val_split'] = 0.10
        
        print(f"Epochs: {config['training']['epochs']}")
        print(f"Learning Rate: {config['model']['learning_rate']} (Adam)")
        print(f"Batch Size: {config['training']['batch_size']}")
        print(f"Hidden Size: {config['model']['hidden_size']}")
        print(f"Early Stopping: {config['training']['early_stopping']}")
        
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        print("使用默认批处理配置")
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
                'epochs': 100,
                'batch_size': 32,
                'print_every': 5,
                'sample_every': 10,
                'sample_length': 200,
                'random_seed': 42,
                'early_stopping': True,
                'patience': 10,
                'min_delta': 0.001,
                'lr_decay': 0.95,
                'lr_decay_every': 10
            },
            'results': {
                'save_dir': '../results',
                'model_save_path': '../results/rnn_model_original.pkl',
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
    
    print("训练SimpleRNN模型（批处理）...")
    rnn, full_history, best_loss, test_perplexity, best_epoch, early_stopped, total_time = train_rnn(
        text, char_to_ix, ix_to_char, config
    )
    
    print("保存最终模型...")
    os.makedirs(config['results']['save_dir'], exist_ok=True)
    model_path = config['results']['model_save_path']
    rnn.save_model(model_path)
    
    mapping_path = os.path.join(config['results']['save_dir'], 'char_mapping_original.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            'char_to_ix': char_to_ix,
            'ix_to_char': {int(k): v for k, v in ix_to_char.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"字符映射已保存到: {mapping_path}")
    
    history_path = os.path.join(config['results']['save_dir'], 'training_history_original.json')
    with open(history_path, 'w') as f:
        json.dump(full_history, f, indent=2)
    print(f"训练历史已保存到: {history_path}")
    
    print("生成可视化...")
    os.makedirs(config['results']['plots_dir'], exist_ok=True)
    
    plot_loss_curve(
        full_history['train_losses'],
        val_losses=full_history['val_losses'],
        save_path=os.path.join(config['results']['plots_dir'], 'loss_curve_original.png')
    )
    
    print("生成最终样本文本...")
    h = np.zeros((rnn.hidden_size, 1))
    seed_ix = np.random.randint(0, len(char_to_ix))
    sample_ix = rnn.sample(h, seed_ix, 500, temperature=0.7)
    sample_text = ''.join(ix_to_char[ix] for ix in sample_ix)
    
    print("\n" + "="*70)
    print("生成的文本样本:")
    print("="*70)
    print(sample_text)
    print("="*70)
    
    sample_path = os.path.join(config['results']['save_dir'], 'generated_sample_original.txt')
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    print(f"样本文本已保存到: {sample_path}")
    
    print("\n" + "="*70)
    print("训练结果总结:")
    print("="*70)
    print(f"最佳训练轮数: {best_epoch}")
    print(f"总训练时间: {total_time/60:.1f} 分钟")
    print(f"早停触发: {'是' if early_stopped else '否'}")
    print(f"最佳验证困惑度: {np.exp(best_loss):.2f}")
    print(f"测试集困惑度: {test_perplexity:.2f}")
    print(f"使用批次大小: {config['training']['batch_size']}")
    print("="*70)
    
    print("SimpleRNN批处理训练流程全部完成！")

if __name__ == '__main__':
    main()