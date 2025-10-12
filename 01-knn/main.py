#!/usr/bin/env python3
"""
K-NN 分类器一键运行脚本
一键执行数据下载、训练、评估和演示
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """打印标题"""
    print("\n" + "="*70)
    print(f"🚀 {title}")
    print("="*70)

def run_script(script_name, description):
    """运行指定的Python脚本"""
    print_header(description)
    print(f"📁 执行脚本: {script_name}")
    
    start_time = time.time()
    
    try:
        # 运行脚本并实时输出
        result = subprocess.run([
            sys.executable, 
            os.path.join('src', script_name)
        ], check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        elapsed_time = time.time() - start_time
        print(f"✅ {description} 完成! 耗时: {elapsed_time:.2f}秒")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败! 错误代码: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ 脚本文件未找到: {script_name}")
        return False

def check_requirements():
    """检查依赖是否安装"""
    print_header("检查环境依赖")
    
    required_packages = ['numpy', 'sklearn', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n⚠️  缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖检查通过!")
    return True

def main():
    """主函数"""
    print_header("K-NN 分类器完整工作流")
    print("本脚本将按顺序执行以下步骤:")
    print("1. 📥 下载MNIST数据")
    print("2. 🏋️ 训练K-NN模型")
    print("3. 📊 评估模型性能") 
    print("4. 🎯 运行演示示例")
    print("5. 📈 查看结果汇总")
    
    # 检查当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n📂 工作目录: {current_dir}")
    
    # 确认执行
    input("\n按 Enter 键开始执行，或 Ctrl+C 取消...")
    
    # 检查依赖
    if not check_requirements():
        return
    
    # 执行步骤
    steps = [
        ('download_mnist.py', '下载MNIST数据集'),
        ('train_knn.py', '训练K-NN模型'),
        ('evaluate.py', '评估模型性能'),
        ('demo.py', '运行演示示例')
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for script, description in steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n⚠️ 步骤失败，停止执行")
            break
        
        # 步骤间暂停
        if script != steps[-1][0]:  # 不是最后一步
            time.sleep(1)
    
    # 汇总结果
    print_header("执行结果汇总")
    print(f"✅ 成功步骤: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 所有步骤执行完成!")
        print("\n📊 生成的结果文件:")
        results_dir = os.path.join(current_dir, 'results')
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.endswith(('.png', '.json', '.txt')):
                    print(f"   📄 {file}")
        
        print("\n🎯 下一步:")
        print("   - 查看 results/ 目录中的图表和结果")
        print("   - 修改 src/configs/knn_config.yaml 调整参数")
        print("   - 单独运行某个脚本进行特定测试")
    else:
        print("❌ 部分步骤执行失败，请检查错误信息")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户取消执行")
    except Exception as e:
        print(f"\n\n❌ 发生未知错误: {e}")