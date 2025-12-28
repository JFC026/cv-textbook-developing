#!/usr/bin/env python3
"""
main_improved_simple.py - ImprovedRNN 运行脚本
不修改配置文件，不生成备份文件
"""

import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    
    script_path = os.path.join(current_dir, 'src', script_name)
    
    if not os.path.exists(script_path):
        print(f"脚本不存在: {script_path}")
        return False
    
    try:
        original_modules = set(sys.modules.keys())
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_code = f.read()
        
        exec_env = {
            '__file__': script_path,
            '__name__': '__main__',
            'sys': sys,
            'os': os,
            'time': time
        }
        
        print(f"开始执行: {script_name}")
        start_time = time.time()
        
        exec(script_code, exec_env)
        
        execution_time = time.time() - start_time
        print(f"{description}完成，耗时: {execution_time:.1f}秒")
        
        new_modules = set(sys.modules.keys()) - original_modules
        for module in new_modules:
            if module not in ['sys', 'os', 'time']:
                del sys.modules[module]
        
        return True
        
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("ImprovedRNN 运行脚本")
    print("="*60)
    print("不修改配置文件")
    print("使用现有配置: rnn_improved_config.yaml")
    print("="*60)
    
    required_files = [
        ('data/input.txt', '训练数据'),
        ('config/rnn_improved_config.yaml', '配置文件'),
        ('src/train_rnn_improved.py', '训练脚本'),
        ('src/evaluate_improved.py', '评估脚本')
    ]
    
    print("检查必要文件...")
    all_exists = True
    
    for file_path, description in required_files:
        full_path = os.path.join(current_dir, file_path)
        if os.path.exists(full_path):
            print(f"{description}: 存在")
        else:
            print(f"{description}: 缺失")
            all_exists = False
    
    if not all_exists:
        print("\n缺少必要文件，请检查后重试")
        return
    
    print("\n" + "="*60)
    print("1. 运行 ImprovedRNN 训练...")
    print("="*60)
    
    response = input("是否跳过训练，直接评估现有模型？(y/N): ").strip().lower()
    
    if response != 'y':
        if not run_script('train_rnn_improved.py', '训练 ImprovedRNN'):
            print("训练失败")
            return
    
    print("\n" + "="*60)
    print("2. 运行 ImprovedRNN 评估...")
    print("="*60)
    
    if not run_script('evaluate_improved.py', '评估 ImprovedRNN'):
        print("评估失败")
    
    print("\n" + "="*60)
    print("ImprovedRNN 流程完成!")
    print("="*60)
    
    results_dir = os.path.join(current_dir, 'results')
    if os.path.exists(results_dir):
        print("\n生成的结果文件:")
        for file in os.listdir(results_dir):
            if 'improved' in file.lower() and file.endswith(('.pkl', '.json', '.txt')):
                print(f"results/{file}")

if __name__ == "__main__":
    main()