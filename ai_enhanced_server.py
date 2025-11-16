#!/usr/bin/env python3
"""
AI增强的加密货币交易预测服务器
集成了DeepSeek AI和全面的技术分析引擎
"""

import json
import time
import os
from datetime import datetime, timedelta
import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import websocket
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv

# 导入自定义模块
from technical_analysis import ComprehensiveTechnicalAnalysis
from deepseek_ai import DeepSeekAIAnalyzer

# 加载环境变量
load_dotenv()

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 配置参数
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-80849bf92e2b43f992b77a319910765d')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
MIN_CONFIDENCE = int(os.getenv('MIN_CONFIDENCE', 85))


class EnhancedDataCollector:
    """增强的实时数据采集器，支持多时间周期"""

    def __init__(self):
        # 数据存储 - 使用deque高效管理历史数据
        self.data_storage = {
            'BTCUSDT': {
                '1m': deque(maxlen=1440),  # 24小时
                '3m': deque(maxlen=480),   # 24小时
                '5m': deque(maxlen=288),   # 24小时
                '15m': deque(maxlen=96),   # 24小时
                '1h': deque(maxlen=168),   # 7天
                '4h': deque(maxlen=180),   # 30天
            },
            'ETHUSDT': {
                '1m': deque(maxlen=1440),
                '3m': deque(maxlen=480),
                '5m': deque(maxlen=288),
                '15m': deque(maxlen=96),
                '1h': deque(maxlen=168),
                '4h': deque(maxlen=180),
            }
        }

        # 指标缓存
        self.indicators_cache = {}

        # WebSocket连接
        self.ws_connections = {}

        # 技术分析引擎
        self.tech_analyzer = ComprehensiveTechnicalAnalysis()

        # 最新价格
        self.latest_prices = {'BTCUSDT': 0, 'ETHUSDT': 0}

        # 统计信息
        self.stats = {
            'data_points_collected': 0,
            'websocket_errors': 0,
            'last_update_time': None
        }

    def fetch_historical_data(self, symbol: str, interval: str = '1m', limit: int = 500) -> pd.DataFrame:
        """从Binance获取历史K线数据"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            print(f"Fetching historical data for {symbol} ({interval})...", flush=True)
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            print(f"✓ Fetched {len(df)} candles for {symbol} ({interval})", flush=True)
            return df

        except Exception as e:
            print(f"✗ Error fetching historical data for {symbol}: {e}", flush=True)
            return pd.DataFrame()

    def initialize_historical_data(self):
        """初始化所有时间周期的历史数据"""
        intervals = ['1m', '3m', '5m', '15m', '1h', '4h']
        symbols = ['BTCUSDT', 'ETHUSDT']

        for symbol in symbols:
            for interval in intervals:
                df = self.fetch_historical_data(symbol, interval, limit=500)
                if not df.empty:
                    for _, row in df.iterrows():
                        self.data_storage[symbol][interval].append({
                            'time': row['timestamp'],
                            'open': row['open'],
                            'high': row['high'],
                            'low': row['low'],
                            'close': row['close'],
                            'volume': row['volume']
                        })

                    # 更新最新价格
                    self.latest_prices[symbol] = df.iloc[-1]['close']

                time.sleep(0.2)  # 避免API限制

        print("✓ Historical data initialization complete", flush=True)

    def start_websocket_streams(self):
        """启动WebSocket实时数据流"""
        symbols = ['BTCUSDT', 'ETHUSDT']

        for symbol in symbols:
            # 1分钟K线流
            ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_1m"
            self._start_single_websocket(symbol, '1m', ws_url)
            time.sleep(0.5)

    def _start_single_websocket(self, symbol: str, interval: str, ws_url: str):
        """启动单个WebSocket连接"""

        def on_message(ws, message):
            try:
                data = json.loads(message)
                kline = data['k']

                # 只处理已关闭的K线
                if kline['x']:  # x=True表示K线已关闭
                    candle = {
                        'time': datetime.fromtimestamp(kline['t'] / 1000),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    }

                    self.data_storage[symbol][interval].append(candle)
                    self.latest_prices[symbol] = candle['close']
                    self.stats['data_points_collected'] += 1
                    self.stats['last_update_time'] = datetime.now().isoformat()

                    # 实时计算技术指标
                    self.calculate_comprehensive_indicators(symbol)

                    print(f"{symbol} @ {candle['close']:.2f} USDT", flush=True)

            except Exception as e:
                print(f"WebSocket message error ({symbol}): {e}", flush=True)
                self.stats['websocket_errors'] += 1

        def on_error(ws, error):
            print(f"WebSocket error ({symbol}): {error}", flush=True)
            self.stats['websocket_errors'] += 1

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed ({symbol}): {close_status_code}", flush=True)
            # 自动重连
            time.sleep(5)
            self._start_single_websocket(symbol, interval, ws_url)

        def on_open(ws):
            print(f"✓ WebSocket connected: {symbol} {interval}", flush=True)

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # 在后台线程运行
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()

        self.ws_connections[f"{symbol}_{interval}"] = ws

    def calculate_comprehensive_indicators(self, symbol: str):
        """计算完整的技术指标集"""
        try:
            # 获取1小时数据用于全面分析
            data_1h = list(self.data_storage[symbol]['1h'])

            if len(data_1h) < 100:
                return None

            df = pd.DataFrame(data_1h)

            # 使用技术分析引擎计算所有指标
            indicators = self.tech_analyzer.calculate_all_indicators(df, symbol)

            # 缓存结果
            cache_key = f"{symbol}_comprehensive"
            self.indicators_cache[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'indicators': indicators,
                'current_price': self.latest_prices[symbol]
            }

            return indicators

        except Exception as e:
            print(f"Error calculating comprehensive indicators for {symbol}: {e}", flush=True)
            return None

    def get_multi_timeframe_data(self, symbol: str) -> Dict:
        """获取多时间周期数据用于AI分析"""
        result = {}

        for interval in ['3m', '15m', '1h', '4h']:
            data_list = list(self.data_storage[symbol][interval])
            if len(data_list) >= 50:
                df = pd.DataFrame(data_list[-100:])  # 取最近100根K线
                result[interval] = {
                    'data': df.to_dict('records'),
                    'close_prices': df['close'].tolist(),
                    'latest_close': float(df.iloc[-1]['close'])
                }

        return result

    def get_latest_price(self, symbol: str) -> Dict:
        """获取最新价格信息"""
        return {
            'symbol': symbol,
            'price': float(self.latest_prices.get(symbol, 0)),
            'time': datetime.now().isoformat()
        }


class AITradingAnalyzer:
    """AI交易分析器 - 集成DeepSeek AI"""

    def __init__(self, api_key: str = None):
        self.ai_analyzer = DeepSeekAIAnalyzer(api_key or DEEPSEEK_API_KEY)
        self.prediction_history = deque(maxlen=1000)

        # 性能统计
        self.performance_stats = {
            'total_predictions': 0,
            'predictions_10m': {'correct': 0, 'total': 0, 'win_rate': 0},
            'predictions_30m': {'correct': 0, 'total': 0, 'win_rate': 0},
            'by_symbol': {
                'BTCUSDT': {'correct': 0, 'total': 0, 'win_rate': 0},
                'ETHUSDT': {'correct': 0, 'total': 0, 'win_rate': 0}
            },
            'high_confidence_win_rate': 0,  # 信心度>=90的胜率
            'avg_confidence': 0
        }

    def generate_ai_prediction(
        self,
        symbol: str,
        data_collector: EnhancedDataCollector
    ) -> Optional[Dict]:
        """生成AI增强的交易预测"""

        try:
            # 获取当前价格
            current_price = data_collector.latest_prices.get(symbol, 0)
            if current_price == 0:
                return None

            # 获取全面的技术指标
            cache_key = f"{symbol}_comprehensive"
            cached_indicators = data_collector.indicators_cache.get(cache_key)

            if not cached_indicators:
                indicators = data_collector.calculate_comprehensive_indicators(symbol)
            else:
                indicators = cached_indicators['indicators']

            if not indicators:
                return {'error': 'Insufficient data for analysis'}

            # 获取多时间周期数据
            multi_timeframe_data = data_collector.get_multi_timeframe_data(symbol)

            if len(multi_timeframe_data) < 3:
                return {'error': 'Insufficient multi-timeframe data'}

            # 获取历史表现数据
            historical_performance = {
                'recent_win_rate': self.performance_stats['by_symbol'][symbol]['win_rate'],
                'total_predictions': self.performance_stats['by_symbol'][symbol]['total'],
                'avg_confidence': self.performance_stats['avg_confidence']
            }

            # 调用DeepSeek AI进行分析
            ai_analysis = self.ai_analyzer.analyze_market(
                symbol=symbol,
                current_price=current_price,
                indicators=indicators,
                multi_timeframe_data=multi_timeframe_data,
                historical_performance=historical_performance
            )

            # 构建完整的预测结果
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'ai_analysis': ai_analysis,
                'indicators_summary': self._summarize_indicators(indicators),
                'verification_pending': True
            }

            # 保存到历史记录
            self.prediction_history.append(prediction)
            self.performance_stats['total_predictions'] += 1

            return prediction

        except Exception as e:
            print(f"Error generating AI prediction for {symbol}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _summarize_indicators(self, indicators: Dict) -> Dict:
        """总结关键技术指标"""
        summary = {}

        # 趋势指标
        if 'EMA_5' in indicators and len(indicators['EMA_5']) > 0:
            summary['ema_5'] = float(indicators['EMA_5'][-1])
        if 'EMA_20' in indicators and len(indicators['EMA_20']) > 0:
            summary['ema_20'] = float(indicators['EMA_20'][-1])

        # 动量指标
        if 'RSI_14' in indicators and len(indicators['RSI_14']) > 0:
            summary['rsi_14'] = float(indicators['RSI_14'][-1])

        # 波动性指标
        if 'BB_upper' in indicators and len(indicators['BB_upper']) > 0:
            summary['bb_upper'] = float(indicators['BB_upper'][-1])
        if 'BB_lower' in indicators and len(indicators['BB_lower']) > 0:
            summary['bb_lower'] = float(indicators['BB_lower'][-1])

        # 理论分析
        if 'DOW_primary_trend' in indicators:
            summary['dow_primary'] = indicators['DOW_primary_trend']
        if 'ELLIOTT_direction' in indicators:
            summary['elliott_direction'] = indicators['ELLIOTT_direction']

        return summary

    def verify_predictions(self, data_collector: EnhancedDataCollector):
        """验证历史预测的准确性"""
        current_time = datetime.now()

        for prediction in self.prediction_history:
            if not prediction.get('verification_pending'):
                continue

            pred_time = datetime.fromisoformat(prediction['timestamp'])
            time_diff = (current_time - pred_time).total_seconds()

            symbol = prediction['symbol']
            original_price = prediction['current_price']
            current_price = data_collector.latest_prices.get(symbol, 0)

            if current_price == 0:
                continue

            ai_analysis = prediction.get('ai_analysis', {})

            # 验证10分钟预测
            if time_diff >= 600 and not prediction.get('verified_10m'):
                pred_10m = ai_analysis.get('timeframe_10m', {})
                if pred_10m.get('direction'):
                    is_correct = self._check_prediction_accuracy(
                        pred_10m['direction'],
                        original_price,
                        current_price
                    )

                    prediction['verified_10m'] = True
                    prediction['verified_10m_correct'] = is_correct

                    if is_correct:
                        self.performance_stats['predictions_10m']['correct'] += 1
                        self.performance_stats['by_symbol'][symbol]['correct'] += 1

                    self.performance_stats['predictions_10m']['total'] += 1
                    self.performance_stats['by_symbol'][symbol]['total'] += 1

            # 验证30分钟预测
            if time_diff >= 1800 and not prediction.get('verified_30m'):
                pred_30m = ai_analysis.get('timeframe_30m', {})
                if pred_30m.get('direction'):
                    is_correct = self._check_prediction_accuracy(
                        pred_30m['direction'],
                        original_price,
                        current_price
                    )

                    prediction['verified_30m'] = True
                    prediction['verified_30m_correct'] = is_correct
                    prediction['verification_pending'] = False

                    if is_correct:
                        self.performance_stats['predictions_30m']['correct'] += 1

                    self.performance_stats['predictions_30m']['total'] += 1

            # 更新胜率
            self._update_win_rates()

    def _check_prediction_accuracy(
        self,
        predicted_direction: str,
        original_price: float,
        actual_price: float
    ) -> bool:
        """检查预测是否准确"""
        price_change = actual_price - original_price

        if predicted_direction == 'up' and price_change > 0:
            return True
        elif predicted_direction == 'down' and price_change < 0:
            return True
        elif predicted_direction == 'wait':
            # 观望被视为中性，不计入胜率
            return None

        return False

    def _update_win_rates(self):
        """更新胜率统计"""
        # 10分钟胜率
        if self.performance_stats['predictions_10m']['total'] > 0:
            self.performance_stats['predictions_10m']['win_rate'] = (
                self.performance_stats['predictions_10m']['correct'] /
                self.performance_stats['predictions_10m']['total'] * 100
            )

        # 30分钟胜率
        if self.performance_stats['predictions_30m']['total'] > 0:
            self.performance_stats['predictions_30m']['win_rate'] = (
                self.performance_stats['predictions_30m']['correct'] /
                self.performance_stats['predictions_30m']['total'] * 100
            )

        # 按币种胜率
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            if self.performance_stats['by_symbol'][symbol]['total'] > 0:
                self.performance_stats['by_symbol'][symbol]['win_rate'] = (
                    self.performance_stats['by_symbol'][symbol]['correct'] /
                    self.performance_stats['by_symbol'][symbol]['total'] * 100
                )


# 全局实例
data_collector = EnhancedDataCollector()
ai_analyzer = AITradingAnalyzer()


# ==================== Flask路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/prices')
def get_prices():
    """获取最新价格"""
    prices = {}
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        prices[symbol] = data_collector.get_latest_price(symbol)

    return jsonify(prices)


@app.route('/api/ai_prediction/<symbol>')
def get_ai_prediction(symbol):
    """获取AI增强预测"""
    if symbol not in ['BTCUSDT', 'ETHUSDT']:
        return jsonify({'error': 'Invalid symbol'}), 400

    prediction = ai_analyzer.generate_ai_prediction(symbol, data_collector)

    if prediction and 'error' not in prediction:
        return jsonify(prediction)
    else:
        return jsonify(prediction or {'error': 'Prediction failed'}), 500


@app.route('/api/stats')
def get_stats():
    """获取统计信息"""
    stats = {
        'performance': ai_analyzer.performance_stats,
        'data_collection': data_collector.stats,
        'predictions_history': [
            {
                'symbol': p['symbol'],
                'timestamp': p['timestamp'],
                'current_price': p['current_price'],
                'ai_decision': p.get('ai_analysis', {}).get('action'),
                'confidence': p.get('ai_analysis', {}).get('overall_confidence', 0),
                'verified_10m': p.get('verified_10m', False),
                'verified_10m_correct': p.get('verified_10m_correct', None),
                'verified_30m': p.get('verified_30m', False),
                'verified_30m_correct': p.get('verified_30m_correct', None)
            }
            for p in list(ai_analyzer.prediction_history)[-50:]
        ]
    }

    return jsonify(stats)


@app.route('/api/indicators/<symbol>')
def get_indicators(symbol):
    """获取技术指标"""
    if symbol not in ['BTCUSDT', 'ETHUSDT']:
        return jsonify({'error': 'Invalid symbol'}), 400

    cache_key = f"{symbol}_comprehensive"
    cached = data_collector.indicators_cache.get(cache_key)

    if cached:
        return jsonify({
            'symbol': symbol,
            'timestamp': cached['timestamp'],
            'current_price': cached['current_price'],
            'indicators': cached['indicators']
        })
    else:
        return jsonify({'error': 'No indicators available'}), 404


@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'websocket_connections': len(data_collector.ws_connections),
        'data_points_collected': data_collector.stats['data_points_collected'],
        'total_predictions': ai_analyzer.performance_stats['total_predictions']
    })


# ==================== SocketIO事件 ====================

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print('Client connected', flush=True)
    emit('connection_status', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    print('Client disconnected', flush=True)


@socketio.on('request_prediction')
def handle_prediction_request(data):
    """处理预测请求"""
    symbol = data.get('symbol', 'BTCUSDT')
    prediction = ai_analyzer.generate_ai_prediction(symbol, data_collector)

    if prediction:
        emit('prediction_update', prediction)


# ==================== 后台任务 ====================

def continuous_prediction_loop():
    """持续预测循环"""
    print("Starting continuous prediction loop...", flush=True)

    while True:
        try:
            # 每60秒生成一次预测
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                prediction = ai_analyzer.generate_ai_prediction(symbol, data_collector)

                if prediction and 'error' not in prediction:
                    # 通过WebSocket广播预测
                    socketio.emit('prediction_update', prediction)
                    print(f"✓ Generated prediction for {symbol}", flush=True)

            # 验证历史预测
            ai_analyzer.verify_predictions(data_collector)

            time.sleep(60)  # 每60秒一次

        except Exception as e:
            print(f"Prediction loop error: {e}", flush=True)
            time.sleep(10)


def periodic_indicator_calculation():
    """定期计算技术指标"""
    print("Starting periodic indicator calculation...", flush=True)

    while True:
        try:
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                data_collector.calculate_comprehensive_indicators(symbol)

            time.sleep(30)  # 每30秒更新一次指标

        except Exception as e:
            print(f"Indicator calculation error: {e}", flush=True)
            time.sleep(10)


# ==================== 主程序入口 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("AI-Enhanced Crypto Trading Prediction System")
    print("=" * 60)
    print(f"DeepSeek API: {'✓ Configured' if DEEPSEEK_API_KEY else '✗ Missing'}")
    print(f"Min Confidence: {MIN_CONFIDENCE}%")
    print(f"Server Port: {FLASK_PORT}")
    print("=" * 60)

    # 初始化历史数据
    print("\n[1/4] Initializing historical data...")
    data_collector.initialize_historical_data()

    # 计算初始技术指标
    print("\n[2/4] Calculating initial technical indicators...")
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        data_collector.calculate_comprehensive_indicators(symbol)

    # 启动WebSocket数据流
    print("\n[3/4] Starting WebSocket streams...")
    data_collector.start_websocket_streams()

    # 启动后台任务
    print("\n[4/4] Starting background tasks...")
    threading.Thread(target=continuous_prediction_loop, daemon=True).start()
    threading.Thread(target=periodic_indicator_calculation, daemon=True).start()

    # 启动Flask服务器
    print("\n" + "=" * 60)
    print(f"✓ Server is running at http://localhost:{FLASK_PORT}")
    print(f"✓ Open your browser and navigate to the URL above")
    print("=" * 60 + "\n")

    socketio.run(app, host='0.0.0.0', port=FLASK_PORT, debug=DEBUG_MODE)
