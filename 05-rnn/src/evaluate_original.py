# evaluate_original.py - SimpleRNN模型评估脚本

import numpy as np
import os
import sys
import json
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.rnn_original import SimpleRNN, create_batches
from src.text_loader import load_text_data

class SimpleRNNEvaluator:
    """SimpleRNN模型评估器"""
    
    def __init__(self, model_path, char_mapping_path=None, config_path=None):
        self.model_path = model_path
        self.char_mapping_path = char_mapping_path
        self.config_path = config_path
        
        self.project_root = parent_dir
        
        self.config = self._load_config()
        
        self.char_to_ix, self.ix_to_char = self._load_char_mapping()
        
        self.model = self._load_model()
        
        data_dir = self.config['data']['data_dir']
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(self.project_root, data_dir)
        
        data_path = os.path.join(data_dir, 'input.txt')
        
        print(f"加载数据文件: {data_path}")
        self.text, self.chars, _, _ = load_text_data(data_path)
        
        self.test_data = self._prepare_test_data()
        
        print("SimpleRNN评估器初始化完成")
        print(f"模型类型: SimpleRNN")
        print(f"词汇表大小: {self.model.vocab_size}")
        print(f"隐藏层大小: {self.model.hidden_size}")
        print(f"序列长度: {self.model.seq_length}")
        print(f"测试集大小: {len(self.test_data)} 字符")
    
    def _load_config(self):
        import yaml
        
        config_paths = []
        
        if self.config_path and os.path.exists(self.config_path):
            config_paths.append(self.config_path)
        
        default_paths = [
            os.path.join(self.project_root, 'config', 'rnn_original_config.yaml'),
            os.path.join(self.project_root, 'config', 'rnn_config.yaml'),
            os.path.join(self.project_root, 'rnn_original_config.yaml')
        ]
        
        for path in default_paths:
            if os.path.exists(path) and path not in config_paths:
                config_paths.append(path)
        
        config = None
        used_path = None
        
        for config_path in config_paths:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                used_path = config_path
                print(f"成功加载配置文件: {config_path}")
                break
            except Exception as e:
                print(f"加载配置文件失败 {config_path}: {e}")
                continue
        
        if config is None:
            print("无法加载配置文件，使用默认配置")
            config = {
                'data': {
                    'data_dir': 'data',
                    'seq_length': 25,
                    'train_split': 0.85,
                    'val_split': 0.10
                },
                'model': {
                    'hidden_size': 128
                },
                'training': {
                    'batch_size': 32
                }
            }
        
        if 'data' in config and 'data_dir' in config['data']:
            data_dir = config['data']['data_dir']
            if not os.path.isabs(data_dir):
                config['data']['data_dir'] = os.path.join(self.project_root, data_dir)
        
        return config
    
    def _load_char_mapping(self):
        if self.char_mapping_path and os.path.exists(self.char_mapping_path):
            try:
                with open(self.char_mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                
                char_to_ix = mapping['char_to_ix']
                ix_to_char = {int(k): v for k, v in mapping['ix_to_char'].items()}
                print(f"从文件加载字符映射: {self.char_mapping_path}")
                return char_to_ix, ix_to_char
            except Exception as e:
                print(f"从文件加载字符映射失败: {e}")
        
        print("从数据重新生成字符映射...")
        
        data_dir = self.config['data']['data_dir']
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(self.project_root, data_dir)
        
        data_path = os.path.join(data_dir, 'input.txt')
        
        text, chars, char_to_ix, ix_to_char = load_text_data(data_path)
        
        if self.char_mapping_path:
            save_dir = os.path.dirname(self.char_mapping_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            try:
                with open(self.char_mapping_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'char_to_ix': char_to_ix,
                        'ix_to_char': {int(k): v for k, v in ix_to_char.items()}
                    }, f, ensure_ascii=False, indent=2)
                print(f"字符映射已保存到: {self.char_mapping_path}")
            except Exception as e:
                print(f"保存字符映射失败: {e}")
        
        return char_to_ix, ix_to_char
    
    def _load_model(self):
        print(f"加载模型文件: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        model = SimpleRNN(
            vocab_size=len(self.char_to_ix),
            hidden_size=self.config['model']['hidden_size'],
            seq_length=self.config['data']['seq_length'],
            batch_size=self.config['training'].get('batch_size', 32),
            learning_rate=0.001
        )
        
        model.load_model(self.model_path)
        
        print(f"模型加载成功，词汇表大小: {model.vocab_size}")
        return model
    
    def _prepare_test_data(self):
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        train_idx = int(len(self.text) * train_split)
        val_idx = train_idx + int(len(self.text) * val_split)
        
        test_text = self.text[val_idx:]
        test_data = [self.char_to_ix[ch] for ch in test_text]
        
        print(f"测试集划分: {len(test_text)} 字符 ({100-(train_split+val_split)*100:.1f}%)")
        return test_data
    
    def evaluate_loss_and_perplexity(self):
        print("\n评估测试集损失和困惑度...")
        
        total_loss = 0
        total_batches = 0
        
        eval_batch_size = min(self.model.batch_size, 8)
        batch_gen = create_batches(
            self.test_data, 
            self.model.seq_length, 
            eval_batch_size
        )
        
        h_eval = np.zeros((self.model.hidden_size, eval_batch_size))
        start_time = time.time()
        
        for inputs_batch, targets_batch in batch_gen:
            xs, hs, ys, ps, _ = self.model.forward(inputs_batch, h_eval)
            loss = self.model.compute_loss(ps, targets_batch)
            
            total_loss += loss
            total_batches += 1
            h_eval = hs[-1]
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        perplexity = np.exp(avg_loss)
        eval_time = time.time() - start_time
        
        results = {
            'test_loss': float(avg_loss),
            'test_perplexity': float(perplexity),
            'evaluation_time': eval_time,
            'num_batches': total_batches,
            'samples_per_second': len(self.test_data) / eval_time if eval_time > 0 else 0,
            'avg_batch_size': eval_batch_size
        }
        
        print(f"测试集损失: {avg_loss:.4f}")
        print(f"测试集困惑度: {perplexity:.2f}")
        print(f"评估时间: {eval_time:.2f}秒")
        print(f"评估速度: {results['samples_per_second']:.0f} 字符/秒")
        
        return results
    
    def evaluate_generation_quality(self, num_samples=5, sample_length=200):
        print(f"\n评估文本生成质量（{num_samples}个样本）...")
        
        metrics = {
            'samples': [],
            'diversity_scores': [],
            'coherence_scores': [],
            'repetition_rates': [],
            'vocabulary_usage': []
        }
        
        for i in range(num_samples):
            h = np.zeros((self.model.hidden_size, 1))
            seed_ix = np.random.randint(0, self.model.vocab_size)
            
            start_time = time.time()
            sample_ix = self.model.sample(h, seed_ix, sample_length, temperature=0.7)
            gen_time = time.time() - start_time
            
            sample_text = ''.join(self.ix_to_char[ix] for ix in sample_ix)
            
            diversity = self._calculate_diversity(sample_text)
            coherence = self._calculate_coherence(sample_text)
            repetition = self._calculate_repetition_rate(sample_text)
            vocab_usage = self._calculate_vocabulary_usage(sample_text)
            
            metrics['samples'].append({
                'text': sample_text,
                'generation_time': gen_time,
                'chars_per_second': sample_length / gen_time if gen_time > 0 else 0,
                'temperature': 0.7
            })
            metrics['diversity_scores'].append(diversity)
            metrics['coherence_scores'].append(coherence)
            metrics['repetition_rates'].append(repetition)
            metrics['vocabulary_usage'].append(vocab_usage)
            
            if i == 0:
                print(f"样本 {i+1}:")
                print(f"生成速度: {sample_length / gen_time:.0f} 字符/秒")
                print(f"多样性: {diversity:.3f}")
                print(f"连贯性: {coherence:.3f}")
                print(f"重复率: {repetition:.3f}")
                print(f"词汇使用率: {vocab_usage:.3f}")
                print(f"预览: {sample_text[:80]}...")
        
        metrics['avg_diversity'] = np.mean(metrics['diversity_scores'])
        metrics['avg_coherence'] = np.mean(metrics['coherence_scores'])
        metrics['avg_repetition'] = np.mean(metrics['repetition_rates'])
        metrics['avg_vocabulary_usage'] = np.mean(metrics['vocabulary_usage'])
        metrics['avg_generation_speed'] = np.mean([
            s['chars_per_second'] for s in metrics['samples']
        ])
        
        print(f"\n平均生成速度: {metrics['avg_generation_speed']:.0f} 字符/秒")
        print(f"平均多样性: {metrics['avg_diversity']:.3f}")
        print(f"平均连贯性: {metrics['avg_coherence']:.3f}")
        print(f"平均重复率: {metrics['avg_repetition']:.3f}")
        print(f"平均词汇使用率: {metrics['avg_vocabulary_usage']:.3f}")
        
        return metrics
    
    def _calculate_diversity(self, text):
        if len(text) == 0:
            return 0
        unique_chars = len(set(text))
        return unique_chars / len(text)
    
    def _calculate_coherence(self, text, window_size=10):
        if len(text) < window_size:
            return 0
        
        ngrams = []
        for i in range(len(text) - window_size + 1):
            ngram = text[i:i+window_size]
            ngrams.append(ngram)
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        return 1.0 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0
    
    def _calculate_repetition_rate(self, text, min_repeat=3):
        if len(text) < min_repeat:
            return 0
        
        repeated_chars = 0
        
        i = 0
        while i < len(text):
            j = i
            while j < len(text) and text[j] == text[i]:
                j += 1
            
            if j - i >= min_repeat:
                repeated_chars += (j - i)
            
            i = j
        
        return repeated_chars / len(text) if len(text) > 0 else 0
    
    def _calculate_vocabulary_usage(self, text):
        if len(self.char_to_ix) == 0:
            return 0
        
        unique_chars_in_sample = set(text)
        return len(unique_chars_in_sample) / len(self.char_to_ix)
    
    def evaluate_model_size_and_efficiency(self):
        print("\n评估模型大小和参数效率...")
        
        param_counts = {
            'Wxh': self.model.Wxh.size,
            'Whh': self.model.Whh.size,
            'Why': self.model.Why.size,
            'bh': self.model.bh.size,
            'by': self.model.by.size
        }
        
        total_params = sum(param_counts.values())
        memory_mb = total_params * 4 / (1024 * 1024)
        
        optimizer_params = 0
        optimizer_memory_mb = 0
        
        if hasattr(self.model, 'mWxh'):
            adam_params = (self.model.mWxh.size + self.model.vWxh.size +
                          self.model.mWhh.size + self.model.vWhh.size +
                          self.model.mWhy.size + self.model.vWhy.size +
                          self.model.mbh.size + self.model.vbh.size +
                          self.model.mby.size + self.model.vby.size)
            
            optimizer_params = adam_params
            optimizer_memory_mb = adam_params * 4 / (1024 * 1024)
        
        total_with_optimizer = total_params + optimizer_params
        total_memory_mb = memory_mb + optimizer_memory_mb
        
        test_loss = self.evaluate_loss_and_perplexity()['test_loss']
        test_perplexity = np.exp(test_loss)
        params_per_million = total_params / 1_000_000
        efficiency_score = test_perplexity / params_per_million if params_per_million > 0 else float('inf')
        
        results = {
            'total_parameters': total_params,
            'total_with_optimizer': total_with_optimizer,
            'memory_mb': memory_mb,
            'optimizer_memory_mb': optimizer_memory_mb,
            'total_memory_mb': total_memory_mb,
            'parameter_breakdown': param_counts,
            'parameters_per_million': params_per_million,
            'efficiency_score': efficiency_score
        }
        
        print(f"模型参数: {total_params:,}")
        print(f"优化器状态: {total_with_optimizer:,}")
        print(f"内存占用: {memory_mb:.2f} MB")
        print(f"优化器内存: {total_memory_mb:.2f} MB")
        print(f"每百万参数困惑度: {efficiency_score:.2f}")
        
        return results
    
    def evaluate_training_history(self, history_path):
        print("\n分析训练历史...")
        
        if not os.path.exists(history_path):
            print(f"训练历史文件不存在: {history_path}")
            return None
        
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"加载训练历史失败: {e}")
            return None
        
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        learning_rates = history.get('learning_rates', [])
        
        train_stability = self._calculate_training_stability(train_losses)
        val_stability = self._calculate_training_stability(val_losses)
        convergence_speed = self._calculate_convergence_speed(val_losses)
        
        metrics = {
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': history.get('best_val_loss'),
            'test_perplexity': history.get('test_perplexity'),
            'best_epoch': history.get('best_epoch'),
            'total_training_time': history.get('total_time'),
            'early_stopped': history.get('early_stopped', False),
            'epochs_trained': history.get('epochs_trained'),
            'train_stability': train_stability,
            'val_stability': val_stability,
            'convergence_speed': convergence_speed,
            'final_learning_rate': learning_rates[-1] if learning_rates else None
        }
        
        if metrics['total_training_time'] and metrics['epochs_trained']:
            metrics['seconds_per_epoch'] = metrics['total_training_time'] / metrics['epochs_trained']
            metrics['chars_per_second'] = len(self.text) * metrics['epochs_trained'] / metrics['total_training_time'] if metrics['total_training_time'] > 0 else 0
        
        print(f"最佳验证损失: {metrics['best_val_loss']:.4f}")
        print(f"最佳epoch: {metrics['best_epoch']}")
        print(f"最终测试困惑度: {metrics['test_perplexity']:.2f}")
        print(f"总训练时间: {metrics['total_training_time']/60:.1f} 分钟")
        print(f"训练稳定性: {train_stability:.3f}")
        print(f"验证稳定性: {val_stability:.3f}")
        print(f"收敛速度: {convergence_speed:.1f} epochs/0.1损失")
        print(f"早停触发: {'是' if metrics['early_stopped'] else '否'}")
        
        return metrics
    
    def _calculate_training_stability(self, losses):
        if len(losses) < 2:
            return 0
        
        diffs = np.diff(losses)
        variance = np.var(diffs)
        stability = 1.0 / (1.0 + variance)
        return stability
    
    def _calculate_convergence_speed(self, losses, target_improvement=0.1):
        if len(losses) < 2:
            return float('inf')
        
        initial_loss = losses[0]
        target_loss = initial_loss - target_improvement
        
        for i, loss in enumerate(losses):
            if loss <= target_loss:
                return i + 1
        
        return float('inf')
    
    def evaluate_inference_speed(self, num_runs=10, sequence_length=100):
        print(f"\n评估推理速度（{num_runs}次运行）...")
        
        times = []
        
        for i in range(num_runs):
            h = np.zeros((self.model.hidden_size, 1))
            seed_ix = np.random.randint(0, self.model.vocab_size)
            
            start_time = time.perf_counter()
            sample_ix = self.model.sample(h, seed_ix, sequence_length, temperature=0.7)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        chars_per_second = sequence_length / avg_time if avg_time > 0 else 0
        
        results = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'chars_per_second': chars_per_second,
            'inference_speed_std_percent': (std_time / avg_time * 100) if avg_time > 0 else 0
        }
        
        print(f"平均推理时间: {avg_time*1000:.2f} 毫秒/100字符")
        print(f"推理速度: {chars_per_second:.0f} 字符/秒")
        print(f"时间标准差: {std_time*1000:.2f} 毫秒 ({results['inference_speed_std_percent']:.1f}%)")
        
        return results
    
    def run_comprehensive_evaluation(self, history_path=None):
        print("="*70)
        print("SimpleRNN 综合评估报告")
        print("="*70)
        
        results = {
            'model_info': {
                'name': 'SimpleRNN',
                'vocab_size': self.model.vocab_size,
                'hidden_size': self.model.hidden_size,
                'seq_length': self.model.seq_length,
                'batch_size': self.model.batch_size,
                'model_path': self.model_path
            },
            'loss_and_perplexity': self.evaluate_loss_and_perplexity(),
            'generation_quality': self.evaluate_generation_quality(),
            'model_size_and_efficiency': self.evaluate_model_size_and_efficiency(),
            'inference_speed': self.evaluate_inference_speed()
        }
        
        if history_path:
            results['training_history'] = self.evaluate_training_history(history_path)
        
        results['comprehensive_score'] = self._calculate_comprehensive_score(results)
        self._generate_summary(results)
        
        return results
    
    def _calculate_comprehensive_score(self, results, weights=None):
        if weights is None:
            weights = {
                'loss_weight': 0.3,
                'quality_weight': 0.25,
                'efficiency_weight': 0.25,
                'speed_weight': 0.2
            }
        
        try:
            perplexity = results['loss_and_perplexity']['test_perplexity']
            perplexity_score = 1.0 / (1.0 + np.log(perplexity))
            
            generation_quality = results['generation_quality']['avg_diversity']
            quality_score = generation_quality
            
            efficiency = 1.0 / results['model_size_and_efficiency']['efficiency_score']
            efficiency_score = min(efficiency, 1.0)
            
            speed = results['inference_speed']['chars_per_second'] / 1000
            speed_score = min(speed, 1.0)
            
            comprehensive_score = (
                weights['loss_weight'] * perplexity_score +
                weights['quality_weight'] * quality_score +
                weights['efficiency_weight'] * efficiency_score +
                weights['speed_weight'] * speed_score
            )
            
            return comprehensive_score
        except:
            return 0.0
    
    def _generate_summary(self, results):
        print("\n" + "="*70)
        print("评估总结")
        print("="*70)
        
        loss_eval = results['loss_and_perplexity']
        gen_quality = results['generation_quality']
        model_size = results['model_size_and_efficiency']
        inference = results['inference_speed']
        
        print(f"性能指标:")
        print(f"测试集困惑度: {loss_eval['test_perplexity']:.2f}")
        print(f"评估速度: {loss_eval['samples_per_second']:.0f} 字符/秒")
        
        print(f"\n生成质量:")
        print(f"多样性: {gen_quality['avg_diversity']:.3f}")
        print(f"连贯性: {gen_quality['avg_coherence']:.3f}")
        print(f"重复率: {gen_quality['avg_repetition']:.3f}")
        print(f"词汇使用率: {gen_quality['avg_vocabulary_usage']:.3f}")
        print(f"生成速度: {gen_quality['avg_generation_speed']:.0f} 字符/秒")
        
        print(f"\n资源使用:")
        print(f"模型参数: {model_size['total_parameters']:,}")
        print(f"内存占用: {model_size['memory_mb']:.2f} MB")
        print(f"参数效率: {model_size['efficiency_score']:.2f}")
        
        print(f"\n推理速度:")
        print(f"推理速度: {inference['chars_per_second']:.0f} 字符/秒")
        print(f"时间稳定性: {inference['inference_speed_std_percent']:.1f}%")
        
        if 'training_history' in results:
            history = results['training_history']
            print(f"\n训练表现:")
            print(f"每epoch时间: {history.get('seconds_per_epoch', 0):.1f} 秒")
            print(f"总训练epoch: {history.get('epochs_trained', 0)}")
            print(f"训练稳定性: {history.get('train_stability', 0):.3f}")
            print(f"收敛速度: {history.get('convergence_speed', 0):.1f} epochs/0.1损失")
        
        print(f"\n综合评分: {results.get('comprehensive_score', 0):.3f}")
        print("="*70)


def main():
    print("SimpleRNN 模型综合评估")
    print("="*70)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    base_dir = os.path.join(project_root, 'results')
    
    model_paths = [
        os.path.join(base_dir, 'rnn_model_original.pkl'),
        os.path.join(base_dir, 'rnn_model.pkl'),
        os.path.join(project_root, 'results', 'rnn_model_original.pkl')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(f"找不到模型文件，尝试了以下路径:")
        for path in model_paths:
            print(f"- {path}")
        print("请先运行训练脚本: python train_rnn_original.py")
        return
    
    char_mapping_paths = [
        os.path.join(base_dir, 'char_mapping_original.json'),
        os.path.join(base_dir, 'char_mapping.json'),
        os.path.join(project_root, 'results', 'char_mapping_original.json')
    ]
    
    char_mapping_path = None
    for path in char_mapping_paths:
        if os.path.exists(path):
            char_mapping_path = path
            break
    
    history_paths = [
        os.path.join(base_dir, 'training_history_original.json'),
        os.path.join(base_dir, 'training_history.json'),
        os.path.join(project_root, 'results', 'training_history_original.json')
    ]
    
    history_path = None
    for path in history_paths:
        if os.path.exists(path):
            history_path = path
            break
    
    config_path = os.path.join(project_root, 'config', 'rnn_original_config.yaml')
    
    print(f"项目根目录: {project_root}")
    print(f"使用模型文件: {model_path}")
    print(f"使用字符映射: {char_mapping_path if char_mapping_path else '从数据重新生成'}")
    print(f"使用训练历史: {history_path if history_path else '未找到'}")
    print(f"使用配置文件: {config_path if os.path.exists(config_path) else '使用默认配置'}")
    
    evaluator = SimpleRNNEvaluator(model_path, char_mapping_path, config_path)
    results = evaluator.run_comprehensive_evaluation(history_path)
    
    output_path = os.path.join(base_dir, 'evaluation_original_comprehensive.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n评估结果已保存到: {output_path}")
    
    brief_report_path = os.path.join(base_dir, 'evaluation_original_brief.txt')
    with open(brief_report_path, 'w', encoding='utf-8') as f:
        f.write("SimpleRNN 评估简要报告\n")
        f.write("="*50 + "\n")
        f.write(f"模型: SimpleRNN\n")
        f.write(f"测试困惑度: {results['loss_and_perplexity']['test_perplexity']:.2f}\n")
        f.write(f"生成多样性: {results['generation_quality']['avg_diversity']:.3f}\n")
        f.write(f"模型参数: {results['model_size_and_efficiency']['total_parameters']:,}\n")
        f.write(f"推理速度: {results['inference_speed']['chars_per_second']:.0f} 字符/秒\n")
        f.write(f"综合评分: {results.get('comprehensive_score', 0):.3f}\n")
    
    print(f"简要报告已保存到: {brief_report_path}")


if __name__ == '__main__':
    main()