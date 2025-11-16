"""
加密货币 AI 交易系统 V2.0 - 集成 DeepSeek API
完整实现道氏理论、波浪理论、江恩理论和现代技术指标
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import websocket
import threading
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests
import talib
from collections import deque
import warnings
from openai import OpenAI
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'crypto-ai-trader-v2'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-80849bf92e2b43f992b77a319910765d"
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

class AdvancedTechnicalAnalysis:
    """高级技术分析 - 道氏、波浪、江恩理论"""

    @staticmethod
    def identify_dow_trend(df, period='4h'):
        """道氏理论趋势识别"""
        if len(df) < 20:
            return {'trend': 'UNKNOWN', 'strength': 0}

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 识别高点和低点
        highs = []
        lows = []

        for i in range(5, len(close)-5):
            if high[i] == max(high[i-5:i+5]):
                highs.append((i, high[i]))
            if low[i] == min(low[i-5:i+5]):
                lows.append((i, low[i]))

        if len(highs) < 2 or len(lows) < 2:
            return {'trend': 'UNKNOWN', 'strength': 0}

        # 判断趋势
        recent_highs = [h[1] for h in highs[-3:]]
        recent_lows = [l[1] for l in lows[-3:]]

        # 上升趋势：高点抬高，低点抬高
        higher_highs = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
        higher_lows = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))

        # 下降趋势：高点降低，低点降低
        lower_highs = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
        lower_lows = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))

        if higher_highs and higher_lows:
            strength = (recent_highs[-1] - recent_highs[0]) / recent_highs[0] * 100
            return {'trend': 'BULLISH', 'strength': strength, 'confidence': 0.9}
        elif lower_highs and lower_lows:
            strength = (recent_highs[0] - recent_highs[-1]) / recent_highs[0] * 100
            return {'trend': 'BEARISH', 'strength': strength, 'confidence': 0.9}
        else:
            return {'trend': 'SIDEWAYS', 'strength': 0, 'confidence': 0.5}

    @staticmethod
    def identify_elliott_wave(df):
        """艾略特波浪理论识别"""
        if len(df) < 50:
            return {'wave': 'UNKNOWN', 'position': None}

        close = df['close'].values

        # 寻找波峰和波谷
        peaks = []
        troughs = []

        for i in range(10, len(close)-10):
            if close[i] == max(close[i-10:i+10]):
                peaks.append((i, close[i]))
            if close[i] == min(close[i-10:i+10]):
                troughs.append((i, close[i]))

        if len(peaks) < 3 or len(troughs) < 3:
            return {'wave': 'UNKNOWN', 'position': None}

        # 简化的波浪识别
        # 浪1: 上涨
        # 浪2: 回调 (23.6%-61.8%)
        # 浪3: 主升浪 (通常最强)
        # 浪4: 回调
        # 浪5: 最后冲刺

        recent_moves = []
        all_points = sorted(peaks + troughs, key=lambda x: x[0])

        for i in range(len(all_points)-1):
            price_change = (all_points[i+1][1] - all_points[i][1]) / all_points[i][1]
            recent_moves.append(price_change)

        # 简单模式匹配
        if len(recent_moves) >= 5:
            # 检查是否符合 5 浪结构
            if (recent_moves[-5] > 0 and  # 浪1 上涨
                recent_moves[-4] < 0 and  # 浪2 回调
                recent_moves[-3] > recent_moves[-5] and  # 浪3 最强
                recent_moves[-2] < 0 and  # 浪4 回调
                recent_moves[-1] > 0):     # 浪5 上涨

                # 判断当前位置
                if abs(recent_moves[-1]) < abs(recent_moves[-3]) * 0.5:
                    return {
                        'wave': 'IMPULSE',
                        'position': 'Wave_5_in_progress',
                        'next_move': 'CORRECTIVE',
                        'confidence': 0.7
                    }
                else:
                    return {
                        'wave': 'IMPULSE',
                        'position': 'Wave_5_ending',
                        'next_move': 'CORRECTIVE_A',
                        'confidence': 0.8
                    }

        return {'wave': 'UNKNOWN', 'position': None, 'confidence': 0}

    @staticmethod
    def calculate_fibonacci_levels(df, lookback=100):
        """计算斐波那契回撤位和扩展位"""
        if len(df) < lookback:
            lookback = len(df)

        recent = df.tail(lookback)
        high = recent['high'].max()
        low = recent['low'].min()
        diff = high - low

        # 回撤位
        retracement = {
            'level_0': high,
            'level_236': high - 0.236 * diff,
            'level_382': high - 0.382 * diff,
            'level_500': high - 0.500 * diff,
            'level_618': high - 0.618 * diff,
            'level_786': high - 0.786 * diff,
            'level_100': low
        }

        # 扩展位
        extension = {
            'level_1272': high + 0.272 * diff,
            'level_1618': high + 0.618 * diff,
            'level_2618': high + 1.618 * diff
        }

        return {
            'retracement': retracement,
            'extension': extension,
            'range_high': high,
            'range_low': low
        }

    @staticmethod
    def gann_angles(current_price, timeframe_bars=100):
        """江恩角度线计算"""
        # 1x1 线 (45度) - 最重要的支撑/阻力
        gann_1x1_support = current_price * 0.98
        gann_1x1_resistance = current_price * 1.02

        # 其他江恩角度
        angles = {
            '1x1_support': gann_1x1_support,
            '1x1_resistance': gann_1x1_resistance,
            '1x2_support': current_price * 0.97,
            '2x1_resistance': current_price * 1.03,
            '1x4_support': current_price * 0.96,
            '4x1_resistance': current_price * 1.04
        }

        # 江恩时间周期
        time_cycles = {
            'minor': 7,   # 7天周期
            'intermediate': 30,  # 30天周期
            'major': 90   # 90天周期
        }

        return {
            'angles': angles,
            'time_cycles': time_cycles,
            'current_price': current_price
        }

class MultiTimeframeAnalyzer:
    """多周期分析器"""

    def __init__(self):
        self.cache = {}

    def fetch_klines(self, symbol, interval, limit=200):
        """获取 K 线数据"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_volume',
                'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return pd.DataFrame()

    def calculate_comprehensive_indicators(self, df):
        """计算全部技术指标"""
        if len(df) < 50:
            return {}

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        indicators = {}

        # 移动平均线
        indicators['EMA_9'] = talib.EMA(close, timeperiod=9)
        indicators['EMA_20'] = talib.EMA(close, timeperiod=20)
        indicators['EMA_50'] = talib.EMA(close, timeperiod=50)
        indicators['EMA_200'] = talib.EMA(close, timeperiod=200)

        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)

        # 布林带
        indicators['BB_upper'], indicators['BB_middle'], indicators['BB_lower'] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )

        # 布林带宽度
        bb_width = (indicators['BB_upper'] - indicators['BB_lower']) / indicators['BB_middle']
        indicators['BB_width'] = bb_width

        # MACD
        indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # RSI
        indicators['RSI_7'] = talib.RSI(close, timeperiod=7)
        indicators['RSI_14'] = talib.RSI(close, timeperiod=14)
        indicators['RSI_21'] = talib.RSI(close, timeperiod=21)

        # Stochastic
        indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(
            high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
        )

        # ATR
        indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)

        # ADX
        indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)

        # CCI
        indicators['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        # MFI
        indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)

        # OBV
        indicators['OBV'] = talib.OBV(close, volume)

        # SAR
        indicators['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

        # Williams %R
        indicators['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        # 成交量均线
        indicators['Volume_MA'] = talib.SMA(volume, timeperiod=20)
        indicators['Volume_ratio'] = volume / indicators['Volume_MA']

        return indicators

    def analyze_multi_timeframe(self, symbol):
        """多周期综合分析"""
        timeframes = {
            '3m': '3m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h'
        }

        analysis = {}

        for tf_name, tf_interval in timeframes.items():
            df = self.fetch_klines(symbol, tf_interval, limit=200)

            if df.empty:
                continue

            # 计算技术指标
            indicators = self.calculate_comprehensive_indicators(df)

            # 道氏理论分析
            dow_trend = AdvancedTechnicalAnalysis.identify_dow_trend(df, tf_name)

            # 波浪理论分析
            elliott_wave = AdvancedTechnicalAnalysis.identify_elliott_wave(df)

            # 斐波那契水平
            fib_levels = AdvancedTechnicalAnalysis.calculate_fibonacci_levels(df)

            # 当前价格
            current_price = df['close'].iloc[-1]

            # 江恩角度
            gann = AdvancedTechnicalAnalysis.gann_angles(current_price)

            # 提取最新指标值
            latest_indicators = {}
            for key, value in indicators.items():
                if isinstance(value, np.ndarray) and len(value) > 0:
                    latest_indicators[key] = float(value[-1]) if not np.isnan(value[-1]) else None

            analysis[tf_name] = {
                'current_price': float(current_price),
                'indicators': latest_indicators,
                'dow_trend': dow_trend,
                'elliott_wave': elliott_wave,
                'fibonacci': fib_levels,
                'gann': gann,
                'klines': df.tail(50).to_dict('records')  # 最近50根K线
            }

        return analysis

class DeepSeekTradingAI:
    """DeepSeek AI 交易决策引擎"""

    def __init__(self):
        self.client = deepseek_client
        self.trade_history = []
        self.win_rate = {'10m': 0, '30m': 0}
        self.total_trades = 0
        self.correct_predictions = {'10m': 0, '30m': 0}

        # 加载 prompt
        with open('ai_trading_prompt.md', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()

    def generate_trading_signal(self, symbol, multi_tf_analysis):
        """使用 DeepSeek API 生成交易信号"""
        try:
            # 准备市场数据
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'multi_timeframe_analysis': multi_tf_analysis
            }

            # 调用 DeepSeek API
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",  # 使用思考模式
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt[:4000]  # 限制长度
                    },
                    {
                        "role": "user",
                        "content": f"""
请分析以下市场数据，给出 {symbol} 的 10 分钟和 30 分钟精确预测：

市场数据:
{json.dumps(market_data, indent=2, ensure_ascii=False)}

要求:
1. 使用道氏理论、波浪理论、江恩理论分析趋势
2. 检查多周期一致性 (3m/15m/1h/4h)
3. 计算信心度评分（需 ≥85）
4. 检查多空确认清单（需 ≥6/8）
5. 确认不是追涨追跌
6. 输出完整的 Markdown 思维链分析
7. 输出标准 JSON 信号格式

JSON 格式示例:
{{
  "signal": {{
    "direction": "LONG" 或 "SHORT" 或 "WAIT",
    "confidence": 85-100,
    "entry_price": 98500,
    "stop_loss": 97500,
    "take_profit": 101500,
    "risk_reward_ratio": "1:3"
  }},
  "predictions": {{
    "10m": {{
      "direction": "UP" 或 "DOWN",
      "target_price": 98800,
      "probability": 0.85,
      "signal_type": "B10" 或 "S10"
    }},
    "30m": {{
      "direction": "UP" 或 "DOWN",
      "target_price": 99500,
      "probability": 0.80,
      "signal_type": "B30" 或 "S30"
    }}
  }},
  "reasoning": ["原因1", "原因2", ...],
  "action": "open_long" 或 "open_short" 或 "wait"
}}
"""
                    }
                ],
                temperature=0.3,  # 降低随机性
                max_tokens=4000
            )

            # 解析响应
            ai_response = response.choices[0].message.content

            # 提取 JSON 部分
            try:
                # 尝试提取 JSON
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = ai_response[json_start:json_end]
                    signal_data = json.loads(json_str)
                else:
                    # 如果没有找到 JSON，创建默认信号
                    signal_data = {
                        'signal': {'direction': 'WAIT', 'confidence': 0},
                        'predictions': {
                            '10m': {'direction': 'NEUTRAL', 'signal_type': 'WAIT'},
                            '30m': {'direction': 'NEUTRAL', 'signal_type': 'WAIT'}
                        },
                        'action': 'wait'
                    }
            except json.JSONDecodeError:
                signal_data = {
                    'signal': {'direction': 'WAIT', 'confidence': 0},
                    'predictions': {
                        '10m': {'direction': 'NEUTRAL', 'signal_type': 'WAIT'},
                        '30m': {'direction': 'NEUTRAL', 'signal_type': 'WAIT'}
                    },
                    'action': 'wait'
                }

            # 添加思维链分析
            signal_data['reasoning_chain'] = ai_response
            signal_data['timestamp'] = datetime.now().isoformat()
            signal_data['symbol'] = symbol

            # 记录交易
            self.trade_history.append(signal_data)

            return signal_data

        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return {
                'error': str(e),
                'signal': {'direction': 'WAIT', 'confidence': 0},
                'predictions': {
                    '10m': {'direction': 'NEUTRAL', 'signal_type': 'WAIT'},
                    '30m': {'direction': 'NEUTRAL', 'signal_type': 'WAIT'}
                },
                'action': 'wait'
            }

    def verify_prediction(self, trade_id, actual_price_10m, actual_price_30m):
        """验证预测准确性"""
        if trade_id >= len(self.trade_history):
            return

        trade = self.trade_history[trade_id]

        if 'predictions' not in trade:
            return

        current_price = trade.get('current_price', 0)

        # 验证 10 分钟
        if '10m' in trade['predictions']:
            pred_10m = trade['predictions']['10m']
            if pred_10m['direction'] == 'UP' and actual_price_10m > current_price:
                self.correct_predictions['10m'] += 1
            elif pred_10m['direction'] == 'DOWN' and actual_price_10m < current_price:
                self.correct_predictions['10m'] += 1

        # 验证 30 分钟
        if '30m' in trade['predictions']:
            pred_30m = trade['predictions']['30m']
            if pred_30m['direction'] == 'UP' and actual_price_30m > current_price:
                self.correct_predictions['30m'] += 1
            elif pred_30m['direction'] == 'DOWN' and actual_price_30m < current_price:
                self.correct_predictions['30m'] += 1

        self.total_trades += 1

        # 更新胜率
        if self.total_trades > 0:
            self.win_rate['10m'] = (self.correct_predictions['10m'] / self.total_trades) * 100
            self.win_rate['30m'] = (self.correct_predictions['30m'] / self.total_trades) * 100

# 全局实例
multi_tf_analyzer = MultiTimeframeAnalyzer()
deepseek_ai = DeepSeekTradingAI()

# Flask 路由
@app.route('/')
def index():
    return render_template('index_v2.html')

@app.route('/api/analyze/<symbol>')
def analyze_symbol(symbol):
    """分析指定币种"""
    try:
        # 多周期分析
        analysis = multi_tf_analyzer.analyze_multi_timeframe(symbol)

        # DeepSeek AI 生成信号
        signal = deepseek_ai.generate_trading_signal(symbol, analysis)

        return jsonify({
            'success': True,
            'analysis': analysis,
            'signal': signal
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def get_stats():
    """获取统计数据"""
    return jsonify({
        'total_trades': deepseek_ai.total_trades,
        'win_rate': deepseek_ai.win_rate,
        'trade_history': deepseek_ai.trade_history[-20:]  # 最近20笔
    })

@app.route('/api/price/<symbol>')
def get_current_price(symbol):
    """获取当前价格"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return jsonify({
            'symbol': symbol,
            'price': float(data['price']),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket 事件
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to trading server'})

@socketio.on('request_analysis')
def handle_analysis_request(data):
    """处理分析请求"""
    symbol = data.get('symbol', 'BTCUSDT')

    try:
        # 多周期分析
        analysis = multi_tf_analyzer.analyze_multi_timeframe(symbol)

        # DeepSeek AI 生成信号
        signal = deepseek_ai.generate_trading_signal(symbol, analysis)

        emit('analysis_result', {
            'symbol': symbol,
            'analysis': analysis,
            'signal': signal
        })
    except Exception as e:
        emit('error', {'message': str(e)})

def continuous_monitoring():
    """持续监控和信号生成"""
    while True:
        try:
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                # 生成分析
                analysis = multi_tf_analyzer.analyze_multi_timeframe(symbol)
                signal = deepseek_ai.generate_trading_signal(symbol, analysis)

                # 广播信号
                socketio.emit('signal_update', {
                    'symbol': symbol,
                    'signal': signal,
                    'timestamp': datetime.now().isoformat()
                })

            # 每 3 分钟更新一次
            time.sleep(180)
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(60)

if __name__ == '__main__':
    print("🚀 启动加密货币 AI 交易系统 V2.0")
    print("=" * 60)
    print("✅ 集成 DeepSeek AI 推理引擎")
    print("✅ 道氏理论、波浪理论、江恩理论分析")
    print("✅ 多周期技术指标 (3m/15m/1h/4h)")
    print("✅ 实时信号生成和可视化")
    print("=" * 60)

    # 启动持续监控线程
    threading.Thread(target=continuous_monitoring, daemon=True).start()

    # 启动服务器
    print("🌐 服务器启动: http://localhost:5000")
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')
