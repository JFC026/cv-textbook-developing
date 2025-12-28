# evaluate_improved.py - ImprovedRNN模型评估脚本

import numpy as np
import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.rnn_improved import ImprovedRNN
from src.rnn_original import create_batches
from src.text_loader import load_text_data

class ImprovedRNNEvaluator:
    """ImprovedRNN模型评估器"""
    
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
        
        print(f"ImprovedRNN评估器初始化完成")
        print(f"模型类型: ImprovedRNN")
        print(f"词汇表大小: {self.model.vocab_size}")
        print(f"隐藏层大小: {self.model.hidden_size}")
        print(f"Dropout率: {self.model.dropout_rate}")
    
    def _load_config(self):
        import yaml
        
        config_paths = []
        if self.config_path and os.path.exists(self.config_path):
            config_paths.append(self.config_path)
        
        default_paths = [
            os.path.join(self.project_root, 'config', 'rnn_improved_config.yaml'),
            os.path.join(self.project_root, 'config', 'rnn_config.yaml'),
            os.path.join(self.project_root, 'rnn_improved_config.yaml')
        ]
        
        for path in default_paths:
            if os.path.exists(path) and path not in config_paths:
                config_paths.append(path)
        
        config = None
        for config_path in config_paths:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
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
                },
                'improved': {
                    'dropout_rate': 0.2,
                    'top_k_sampling': 5
                }
            }
        
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
        
        return char_to_ix, ix_to_char
    
    def _load_model(self):
        print(f"加载改进模型文件: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        model = ImprovedRNN(
            vocab_size=len(self.char_to_ix),
            hidden_size=self.config['model']['hidden_size'],
            seq_length=self.config['data']['seq_length'],
            batch_size=self.config['training'].get('batch_size', 32),
            learning_rate=0.001,
            dropout_rate=self.config['improved'].get('dropout_rate', 0.2)
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
        batch_gen = create_batches(self.test_data, self.model.seq_length, eval_batch_size)
        
        h_eval = np.zeros((self.model.hidden_size, eval_batch_size))
        start_time = time.time()
        
        for inputs_batch, targets_batch in batch_gen:
            xs, hs, ys, ps, _ = self.model.forward(inputs_batch, h_eval, training=False)
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
            'evaluation_mode': 'without_dropout'
        }
        
        print(f"测试集损失: {avg_loss:.4f}")
        print(f"测试集困惑度: {perplexity:.2f}")
        print(f"评估时间: {eval_time:.2f}秒")
        
        return results
    
    def evaluate_sampling_strategies(self, num_samples=3, sample_length=200):
        print(f"\n评估不同采样策略（{num_samples}个样本）...")
        
        strategies = [
            {'name': '贪婪采样', 'temperature': 0.1, 'top_k': None},
            {'name': '标准采样', 'temperature': 0.7, 'top_k': None},
            {'name': '高随机性', 'temperature': 1.2, 'top_k': None},
            {'name': 'Top-k (k=5)', 'temperature': 0.7, 'top_k': 5},
            {'name': 'Top-k (k=10)', 'temperature': 0.7, 'top_k': 10}
        ]
        
        metrics = {'strategies': {}}
        
        for strategy in strategies:
            print(f"\n策略: {strategy['name']}")
            strategy_metrics = {
                'samples': [],
                'diversity_scores': [],
                'coherence_scores': [],
                'repetition_rates': []
            }
            
            for i in range(num_samples):
                h = np.zeros((self.model.hidden_size, 1))
                seed_ix = np.random.randint(0, self.model.vocab_size)
                
                start_time = time.time()
                sample_ix = self.model.sample(h, seed_ix, sample_length, 
                                             temperature=strategy['temperature'], 
                                             top_k=strategy['top_k'])
                gen_time = time.time() - start_time
                
                sample_text = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                
                diversity = self._calculate_diversity(sample_text)
                coherence = self._calculate_coherence(sample_text)
                repetition = self._calculate_repetition_rate(sample_text)
                
                strategy_metrics['samples'].append({
                    'text': sample_text,
                    'generation_time': gen_time,
                    'chars_per_second': sample_length / gen_time if gen_time > 0 else 0
                })
                strategy_metrics['diversity_scores'].append(diversity)
                strategy_metrics['coherence_scores'].append(coherence)
                strategy_metrics['repetition_rates'].append(repetition)
                
                if i == 0:
                    print(f"样本 1: {sample_text[:60]}...")
                    print(f"多样性: {diversity:.3f}, 连贯性: {coherence:.3f}, 重复率: {repetition:.3f}")
            
            strategy_metrics['avg_diversity'] = np.mean(strategy_metrics['diversity_scores'])
            strategy_metrics['avg_coherence'] = np.mean(strategy_metrics['coherence_scores'])
            strategy_metrics['avg_repetition'] = np.mean(strategy_metrics['repetition_rates'])
            
            metrics['strategies'][strategy['name']] = strategy_metrics
        
        return metrics
    
    def evaluate_dropout_effect(self):
        print("\n评估Dropout效果...")
        
        eval_batch_size = min(self.model.batch_size, 8)
        batch_gen = create_batches(self.test_data[:1000], self.model.seq_length, eval_batch_size)
        
        results = {'with_dropout': None, 'without_dropout': None}
        
        print("评估无Dropout模式...")
        h_eval = np.zeros((self.model.hidden_size, eval_batch_size))
        total_loss_no_dropout = 0
        batches = 0
        
        for inputs_batch, targets_batch in batch_gen:
            xs, hs, ys, ps, _ = self.model.forward(inputs_batch, h_eval, training=False)
            loss = self.model.compute_loss(ps, targets_batch)
            total_loss_no_dropout += loss
            batches += 1
            h_eval = hs[-1]
        
        avg_loss_no_dropout = total_loss_no_dropout / batches if batches > 0 else 0
        results['without_dropout'] = {
            'loss': float(avg_loss_no_dropout),
            'perplexity': float(np.exp(avg_loss_no_dropout))
        }
        
        print("评估有Dropout模式...")
        h_eval = np.zeros((self.model.hidden_size, eval_batch_size))
        total_loss_with_dropout = 0
        batches = 0
        
        for inputs_batch, targets_batch in batch_gen:
            xs, hs, ys, ps, _ = self.model.forward(inputs_batch, h_eval, training=True)
            loss = self.model.compute_loss(ps, targets_batch)
            total_loss_with_dropout += loss
            batches += 1
            h_eval = hs[-1]
        
        avg_loss_with_dropout = total_loss_with_dropout / batches if batches > 0 else 0
        results['with_dropout'] = {
            'loss': float(avg_loss_with_dropout),
            'perplexity': float(np.exp(avg_loss_with_dropout))
        }
        
        print(f"无Dropout - 损失: {avg_loss_no_dropout:.4f}, 困惑度: {np.exp(avg_loss_no_dropout):.2f}")
        print(f"有Dropout - 损失: {avg_loss_with_dropout:.4f}, 困惑度: {np.exp(avg_loss_with_dropout):.2f}")
        print(f"差异: {(avg_loss_with_dropout - avg_loss_no_dropout):.4f}")
        
        return results
    
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
    
    def run_comprehensive_evaluation(self, history_path=None):
        print("="*70)
        print("ImprovedRNN 综合评估报告")
        print("="*70)
        
        results = {
            'model_info': {
                'name': 'ImprovedRNN',
                'vocab_size': self.model.vocab_size,
                'hidden_size': self.model.hidden_size,
                'dropout_rate': self.model.dropout_rate,
                'model_path': self.model_path
            },
            'loss_and_perplexity': self.evaluate_loss_and_perplexity(),
            'sampling_strategies': self.evaluate_sampling_strategies(),
            'dropout_effect': self.evaluate_dropout_effect()
        }
        
        if history_path and os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                results['training_history'] = {
                    'best_val_loss': history.get('best_val_loss'),
                    'test_perplexity': history.get('test_perplexity'),
                    'best_epoch': history.get('best_epoch'),
                    'improved_params': history.get('improved_params', {})
                }
            except Exception as e:
                print(f"加载训练历史失败: {e}")
        
        self._generate_summary(results)
        return results
    
    def _generate_summary(self, results):
        print("\n" + "="*70)
        print("改进版RNN评估总结")
        print("="*70)
        
        loss_eval = results['loss_and_perplexity']
        sampling = results['sampling_strategies']
        dropout = results['dropout_effect']
        
        print(f"性能指标:")
        print(f"测试集困惑度: {loss_eval['test_perplexity']:.2f}")
        print(f"测试集损失: {loss_eval['test_loss']:.4f}")
        
        print(f"\n采样策略对比:")
        for strategy_name, metrics in sampling['strategies'].items():
            print(f"{strategy_name:15} - 多样性: {metrics['avg_diversity']:.3f}, "
                  f"连贯性: {metrics['avg_coherence']:.3f}, "
                  f"重复率: {metrics['avg_repetition']:.3f}")
        
        print(f"\nDropout效果:")
        print(f"无Dropout困惑度: {dropout['without_dropout']['perplexity']:.2f}")
        print(f"有Dropout困惑度: {dropout['with_dropout']['perplexity']:.2f}")
        print(f"差异: {dropout['with_dropout']['perplexity'] - dropout['without_dropout']['perplexity']:.2f}")
        
        if 'training_history' in results:
            history = results['training_history']
            print(f"\n训练信息:")
            print(f"最佳验证损失: {history['best_val_loss']:.4f}")
            print(f"最佳epoch: {history['best_epoch']}")
            if 'improved_params' in history:
                improved = history['improved_params']
                print(f"改进参数: Dropout率={improved.get('dropout_rate', 'N/A')}, "
                      f"Top-k={improved.get('top_k_sampling', 'N/A')}")
        
        print("\n改进功能评估完成")
        print("="*70)


def main():
    print("ImprovedRNN 模型综合评估")
    print("="*70)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    base_dir = os.path.join(project_root, 'results')
    
    model_paths = [
        os.path.join(base_dir, 'rnn_model_improved.pkl'),
        os.path.join(base_dir, 'rnn_model.pkl'),
        os.path.join(project_root, 'results', 'rnn_model_improved.pkl')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(f"找不到改进模型文件")
        for path in model_paths:
            print(f"- {path}")
        print("请先运行训练脚本: python train_rnn_improved.py")
        return
    
    char_mapping_paths = [
        os.path.join(base_dir, 'char_mapping_improved.json'),
        os.path.join(base_dir, 'char_mapping.json')
    ]
    
    char_mapping_path = None
    for path in char_mapping_paths:
        if os.path.exists(path):
            char_mapping_path = path
            break
    
    history_paths = [
        os.path.join(base_dir, 'training_history_improved.json'),
        os.path.join(base_dir, 'training_history.json')
    ]
    
    history_path = None
    for path in history_paths:
        if os.path.exists(path):
            history_path = path
            break
    
    config_path = os.path.join(project_root, 'config', 'rnn_improved_config.yaml')
    
    print(f"项目根目录: {project_root}")
    print(f"使用模型文件: {model_path}")
    print(f"使用字符映射: {char_mapping_path if char_mapping_path else '从数据重新生成'}")
    
    evaluator = ImprovedRNNEvaluator(model_path, char_mapping_path, config_path)
    results = evaluator.run_comprehensive_evaluation(history_path)
    
    output_path = os.path.join(base_dir, 'evaluation_improved_comprehensive.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n评估结果已保存到: {output_path}")

if __name__ == '__main__':
    main()