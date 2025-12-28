#!/usr/bin/env python3
"""
main_original.py - SimpleRNN训练和评估脚本
运行方式：python main_original.py
支持直接评估现有模型
"""

import os
import sys
import time
import argparse
import traceback

def parse_arguments():
    parser = argparse.ArgumentParser(description='SimpleRNN训练脚本')
    parser.add_argument('--skip-train', action='store_true', 
                       help='跳过训练，直接评估')
    parser.add_argument('--train-only', action='store_true',
                       help='只训练')
    parser.add_argument('--eval-only', action='store_true',
                       help='只评估')
    parser.add_argument('--force-train', action='store_true',
                       help='强制重新训练')
    parser.add_argument('--skip-checks', action='store_true',
                       help='跳过文件检查')
    return parser.parse_args()

def check_dependencies(skip_checks=False):
    if skip_checks:
        print("跳过文件检查")
        return True
    
    print("检查项目依赖...")
    
    required_files = [
        'data/input.txt',
        'src/rnn_original.py',
        'src/train_rnn_original.py', 
        'src/evaluate_original.py',
        'src/text_loader.py',
        'config/rnn_original_config.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("缺少以下文件:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("所有依赖文件都存在")
    return True

def check_existing_model():
    model_paths = [
        'results/rnn_model_original.pkl',
        'results/rnn_model.pkl'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"发现现有模型: {model_path}")
            try:
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                print(f"文件大小: {file_size:.2f} MB")
                return model_path
            except:
                return model_path
    
    print("未找到现有模型")
    return None

def run_training():
    print("\n" + "="*70)
    print("训练 SimpleRNN 模型")
    print("="*70)
    
    try:
        sys.path.append('src')
        from src.train_rnn_original import main as train_main
        
        print("开始训练...")
        print("训练时间取决于配置")
        
        start_time = time.time()
        train_main()
        training_time = time.time() - start_time
        
        print(f"训练完成，耗时: {training_time/60:.1f} 分钟")
        return True
        
    except Exception as e:
        print(f"训练失败: {e}")
        traceback.print_exc()
        return False

def run_evaluation():
    print("\n" + "="*70)
    print("评估 SimpleRNN 模型")
    print("="*70)
    
    try:
        sys.path.append('src')
        from src.evaluate_original import main as eval_main
        
        print("开始评估...")
        
        start_time = time.time()
        eval_main()
        eval_time = time.time() - start_time
        
        print(f"评估完成，耗时: {eval_time:.1f} 秒")
        return True
        
    except Exception as e:
        print(f"评估失败: {e}")
        traceback.print_exc()
        return False

def show_training_summary():
    history_path = 'results/training_history_original.json'
    if os.path.exists(history_path):
        try:
            import json
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            print("\n训练历史摘要:")
            print(f"最佳验证损失: {history.get('best_val_loss', 'N/A'):.4f}")
            print(f"测试困惑度: {history.get('test_perplexity', 'N/A'):.2f}")
            print(f"最佳epoch: {history.get('best_epoch', 'N/A')}")
            print(f"总训练epoch: {history.get('epochs_trained', 'N/A')}")
            print(f"早停触发: {'是' if history.get('early_stopped') else '否'}")
        except:
            pass

def show_evaluation_summary():
    eval_path = 'results/evaluation_original_comprehensive.json'
    if os.path.exists(eval_path):
        try:
            import json
            with open(eval_path, 'r') as f:
                eval_results = json.load(f)
            
            if 'loss_and_perplexity' in eval_results:
                loss_eval = eval_results['loss_and_perplexity']
                print("\n评估结果摘要:")
                print(f"测试困惑度: {loss_eval.get('test_perplexity', 'N/A'):.2f}")
                print(f"测试损失: {loss_eval.get('test_loss', 'N/A'):.4f}")
                print(f"评估速度: {loss_eval.get('samples_per_second', 0):.0f} 字符/秒")
        except:
            pass

def check_results():
    print("\n" + "="*70)
    print("检查输出结果")
    print("="*70)
    
    result_files = [
        ('results/rnn_model_original.pkl', '模型文件'),
        ('results/training_history_original.json', '训练历史'),
        ('results/evaluation_original_comprehensive.json', '评估结果'),
        ('results/plots/loss_curve_original.png', '损失曲线')
    ]
    
    for file_path, description in result_files:
        if os.path.exists(file_path):
            print(f"{description}: {file_path}")
        else:
            print(f"{description}: 未找到")

def main():
    args = parse_arguments()
    
    print("="*70)
    print("SimpleRNN 训练和评估脚本")
    print("="*70)
    
    train_model = True
    eval_model = True
    
    if args.eval_only:
        train_model = False
        eval_model = True
        print("模式: 仅评估")
    elif args.train_only:
        train_model = True
        eval_model = False
        print("模式: 仅训练")
    elif args.skip_train:
        train_model = False
        eval_model = True
        print("模式: 跳过训练，直接评估")
    else:
        print("模式: 完整流程")
    
    if not check_dependencies(args.skip_checks):
        print("\n请先解决依赖问题")
        return
    
    existing_model = check_existing_model()
    
    if train_model:
        if existing_model and not args.force_train:
            print(f"\n发现现有模型: {existing_model}")
            response = input("是否跳过训练，使用现有模型？(y/N): ").strip().lower()
            if response == 'y':
                print("跳过训练，使用现有模型")
                train_model = False
    
    if train_model:
        if not run_training():
            print("\n训练失败")
            if eval_model and existing_model:
                print("尝试使用现有模型进行评估")
            else:
                return
    elif eval_model:
        print("\n跳过训练阶段")
    
    if eval_model:
        if not check_existing_model():
            print("\n没有找到模型文件，无法评估")
            return
        
        if not run_evaluation():
            print("\n评估失败")
    
    if train_model:
        show_training_summary()
    
    if eval_model:
        show_evaluation_summary()
    
    check_results()
    
    print("\n" + "="*70)
    print("SimpleRNN 流程完成!")
    print("="*70)
    
    if existing_model:
        print("\n使用说明:")
        print("1. 直接评估现有模型: python main_original.py --eval-only")
        print("2. 重新训练并评估: python main_original.py --force-train")
        print("3. 仅训练: python main_original.py --train-only")
    
    print("\n生成的文件:")
    print("   - 模型: results/rnn_model_original.pkl")
    print("   - 训练历史: results/training_history_original.json")
    print("   - 评估结果: results/evaluation_original_comprehensive.json")
    print("   - 损失曲线: results/plots/loss_curve_original.png")
    print("\n快速命令:")
    print("   python src/evaluate_original.py")
    print("   python src/diagnose.py")
    print("="*70)

if __name__ == "__main__":
    main()