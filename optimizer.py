import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import threading
from collections import deque

class AutoOptimizer:
    """自动优化和策略迭代系统"""
    
    def __init__(self):
        self.performance_history = []
        self.strategy_configs = []
        self.current_config = self.load_config()
        self.optimization_interval = 3600  # 每小时优化一次
        
    def load_config(self):
        """加载策略配置"""
        default_config = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'confidence_threshold': 0.55,
            'entry_threshold': 0.7,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }
        
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except:
            return default_config
            
    def save_config(self):
        """保存策略配置"""
        with open('config.json', 'w') as f:
            json.dump(self.current_config, f, indent=4)
            
    def evaluate_performance(self):
        """评估当前策略性能"""
        response = requests.get('http://localhost:5000/api/stats')
        stats = response.json()
        
        # 计算综合得分
        btc_score = (stats['win_rate_10m']['BTC'] + stats['win_rate_30m']['BTC']) / 2
        eth_score = (stats['win_rate_10m']['ETH'] + stats['win_rate_30m']['ETH']) / 2
        overall_score = (btc_score + eth_score) / 2
        
        performance = {
            'timestamp': datetime.now().isoformat(),
            'config': self.current_config.copy(),
            'score': overall_score,
            'btc_10m': stats['win_rate_10m']['BTC'],
            'btc_30m': stats['win_rate_30m']['BTC'],
            'eth_10m': stats['win_rate_10m']['ETH'],
            'eth_30m': stats['win_rate_30m']['ETH'],
            'total_predictions': stats['total_predictions']
        }
        
        self.performance_history.append(performance)
        return overall_score
        
    def optimize_parameters(self):
        """优化策略参数"""
        current_score = self.evaluate_performance()
        
        if current_score < 60:  # 如果胜率低于60%，进行优化
            print(f"Current win rate: {current_score:.2f}%, optimizing...")
            
            # 参数优化范围
            param_ranges = {
                'rsi_oversold': (20, 40),
                'rsi_overbought': (60, 80),
                'bb_period': (10, 30),
                'bb_std': (1.5, 3),
                'confidence_threshold': (0.5, 0.7),
                'entry_threshold': (0.6, 0.8)
            }
            
            # 随机搜索优化
            best_config = self.current_config.copy()
            best_score = current_score
            
            for _ in range(10):  # 测试10个不同的配置
                test_config = self.current_config.copy()
                
                # 随机调整参数
                for param, (min_val, max_val) in param_ranges.items():
                    if np.random.random() > 0.5:  # 50%概率调整每个参数
                        test_config[param] = np.random.uniform(min_val, max_val)
                        
                # 测试新配置
                self.current_config = test_config
                self.save_config()
                
                # 等待一段时间收集数据
                time.sleep(300)  # 5分钟
                
                # 评估新配置
                test_score = self.evaluate_performance()
                
                if test_score > best_score:
                    best_config = test_config.copy()
                    best_score = test_score
                    print(f"Found better config with score: {best_score:.2f}%")
                    
            # 应用最佳配置
            self.current_config = best_config
            self.save_config()
            print(f"Optimization complete. Best score: {best_score:.2f}%")
            
        else:
            print(f"Current win rate: {current_score:.2f}%, no optimization needed")
            
    def continuous_optimization(self):
        """持续优化循环"""
        while True:
            try:
                # 等待足够的数据
                time.sleep(self.optimization_interval)
                
                # 执行优化
                self.optimize_parameters()
                
                # 保存性能历史
                with open('performance_history.json', 'w') as f:
                    json.dump(self.performance_history, f, indent=4)
                    
            except Exception as e:
                print(f"Optimization error: {e}")
                time.sleep(60)
                
    def generate_report(self):
        """生成性能报告"""
        if len(self.performance_history) < 2:
            return "Insufficient data for report"
            
        df = pd.DataFrame(self.performance_history)
        
        report = {
            'total_evaluations': len(df),
            'average_score': df['score'].mean(),
            'best_score': df['score'].max(),
            'worst_score': df['score'].min(),
            'score_trend': 'improving' if df['score'].iloc[-1] > df['score'].iloc[0] else 'declining',
            'best_config': self.performance_history[df['score'].idxmax()]['config'],
            'current_config': self.current_config,
            'improvement_rate': (df['score'].iloc[-1] - df['score'].iloc[0]) / len(df) if len(df) > 1 else 0
        }
        
        return report

def main():
    optimizer = AutoOptimizer()
    
    # 启动持续优化线程
    optimization_thread = threading.Thread(target=optimizer.continuous_optimization, daemon=True)
    optimization_thread.start()
    
    # 定期生成报告
    while True:
        time.sleep(1800)  # 每30分钟生成一次报告
        report = optimizer.generate_report()
        print("\n=== Performance Report ===")
        print(json.dumps(report, indent=2))
        print("==========================\n")

if __name__ == "__main__":
    main()