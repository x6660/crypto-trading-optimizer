import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import websocket
import threading
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import requests
import talib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class TradingDataCollector:
    """实时数据采集和技术指标计算"""
    
    def __init__(self):
        self.btc_data = deque(maxlen=1000)
        self.eth_data = deque(maxlen=1000)
        self.indicators_cache = {}
        self.ws_btc = None
        self.ws_eth = None
        
    def start_websocket(self):
        """启动WebSocket连接获取实时数据"""
        def on_message_btc(ws, message):
            try:
                data = json.loads(message)
                kline = data['k']
                self.btc_data.append({
                    'time': datetime.fromtimestamp(kline['t']/1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                })
                self.calculate_indicators('BTCUSDT')
            except Exception as e:
                print(f"BTC WebSocket message error: {e}")

        def on_message_eth(ws, message):
            try:
                data = json.loads(message)
                kline = data['k']
                self.eth_data.append({
                    'time': datetime.fromtimestamp(kline['t']/1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                })
                self.calculate_indicators('ETHUSDT')
            except Exception as e:
                print(f"ETH WebSocket message error: {e}")
            
        # BTC WebSocket
        self.ws_btc = websocket.WebSocketApp(
            "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
            on_message=on_message_btc
        )
        
        # ETH WebSocket
        self.ws_eth = websocket.WebSocketApp(
            "wss://stream.binance.com:9443/ws/ethusdt@kline_1m",
            on_message=on_message_eth
        )
        
        # 启动WebSocket线程
        threading.Thread(target=self.ws_btc.run_forever, daemon=True).start()
        threading.Thread(target=self.ws_eth.run_forever, daemon=True).start()
        
    def fetch_historical_data(self, symbol, interval='1m', limit=500):
        """获取历史K线数据"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close',
                                             'volume', 'close_time', 'quote_volume',
                                             'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, symbol):
        """计算全部技术指标"""
        if symbol == 'BTCUSDT':
            data = list(self.btc_data)
        else:
            data = list(self.eth_data)
            
        if len(data) < 100:
            return None
            
        df = pd.DataFrame(data)
        
        # 价格数据
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # 移动平均线
        indicators['SMA_5'] = talib.SMA(close, timeperiod=5)
        indicators['SMA_10'] = talib.SMA(close, timeperiod=10)
        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
        indicators['EMA_9'] = talib.EMA(close, timeperiod=9)
        indicators['EMA_21'] = talib.EMA(close, timeperiod=21)
        indicators['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # 布林带
        indicators['BB_upper'], indicators['BB_middle'], indicators['BB_lower'] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
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
        
        # Ichimoku
        period9_high = pd.Series(high).rolling(window=9).max()
        period9_low = pd.Series(low).rolling(window=9).min()
        indicators['ICHIMOKU_tenkan'] = (period9_high + period9_low) / 2
        
        period26_high = pd.Series(high).rolling(window=26).max()
        period26_low = pd.Series(low).rolling(window=26).min()
        indicators['ICHIMOKU_kijun'] = (period26_high + period26_low) / 2
        
        # 成交量指标
        indicators['VWAP'] = (close * volume).cumsum() / volume.cumsum()
        
        self.indicators_cache[symbol] = indicators
        return indicators

class DeepLearningModel(nn.Module):
    """深度学习预测模型 - LSTM + Transformer混合架构"""
    
    def __init__(self, input_dim=50, hidden_dim=256, num_layers=4, num_heads=8):
        super(DeepLearningModel, self).__init__()
        
        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim*2, nhead=num_heads, dim_feedforward=512, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads, dropout=0.2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)  # 4个输出: 10分钟涨跌概率, 30分钟涨跌概率
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # Transformer处理
        transformer_out = self.transformer(lstm_out)
        
        # 注意力处理
        attn_out, _ = self.attention(transformer_out, transformer_out, transformer_out)
        
        # 取最后一个时间步
        out = attn_out[:, -1, :]
        
        # 全连接层
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.dropout(self.relu(self.fc3(out)))
        out = self.sigmoid(self.fc4(out))
        
        return out

class TradingPredictor:
    """交易预测和信号生成系统"""
    
    def __init__(self):
        self.model_btc = DeepLearningModel()
        self.model_eth = DeepLearningModel()
        self.scaler = StandardScaler()
        self.predictions_history = []
        self.win_rate_10m = {'BTC': 0, 'ETH': 0}
        self.win_rate_30m = {'BTC': 0, 'ETH': 0}
        self.total_predictions = 0
        self.correct_predictions_10m = {'BTC': 0, 'ETH': 0}
        self.correct_predictions_30m = {'BTC': 0, 'ETH': 0}
        
    def prepare_features(self, indicators, price_data):
        """准备模型输入特征"""
        if not indicators:
            return None
            
        features = []
        
        # 提取最新的指标值
        for key, value in indicators.items():
            if isinstance(value, np.ndarray) and len(value) > 0:
                features.append(value[-1] if not np.isnan(value[-1]) else 0)
                
        # 添加价格变化率
        if len(price_data) > 1:
            price_change_1m = (price_data[-1]['close'] - price_data[-2]['close']) / price_data[-2]['close']
            features.append(price_change_1m)
            
        if len(price_data) > 5:
            price_change_5m = (price_data[-1]['close'] - price_data[-5]['close']) / price_data[-5]['close']
            features.append(price_change_5m)
            
        if len(price_data) > 15:
            price_change_15m = (price_data[-1]['close'] - price_data[-15]['close']) / price_data[-15]['close']
            features.append(price_change_15m)
            
        # 添加成交量变化
        if len(price_data) > 1:
            volume_ratio = price_data[-1]['volume'] / (sum([p['volume'] for p in price_data[-10:]]) / 10)
            features.append(volume_ratio)
            
        return np.array(features)
    
    def check_entry_conditions(self, indicators, current_price):
        """检查入场条件 - 基于布林带和多指标共振"""
        if not indicators:
            return False, None
            
        signals = []
        
        # 布林带策略
        bb_upper = indicators.get('BB_upper', [])
        bb_lower = indicators.get('BB_lower', [])
        bb_middle = indicators.get('BB_middle', [])
        
        if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]):
            # 价格触及下轨且RSI超卖 - 做多信号
            if current_price <= bb_lower[-1] * 1.001:
                rsi = indicators.get('RSI_14', [])
                if len(rsi) > 0 and rsi[-1] < 35:
                    signals.append(('LONG', 0.8))
                    
            # 价格触及上轨且RSI超买 - 做空信号
            if current_price >= bb_upper[-1] * 0.999:
                rsi = indicators.get('RSI_14', [])
                if len(rsi) > 0 and rsi[-1] > 65:
                    signals.append(('SHORT', 0.8))
                    
        # MACD策略
        macd = indicators.get('MACD', [])
        macd_signal = indicators.get('MACD_signal', [])
        if len(macd) > 1 and len(macd_signal) > 1:
            if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
                signals.append(('LONG', 0.7))
            elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                signals.append(('SHORT', 0.7))
                
        # 多重共振检查
        if len(signals) >= 2:
            long_signals = [s for s in signals if s[0] == 'LONG']
            short_signals = [s for s in signals if s[0] == 'SHORT']
            
            if len(long_signals) >= 2:
                return True, 'LONG'
            elif len(short_signals) >= 2:
                return True, 'SHORT'
                
        return False, None
    
    def generate_prediction(self, symbol, data_collector):
        """生成预测信号"""
        if symbol == 'BTCUSDT':
            price_data = list(data_collector.btc_data)
            model = self.model_btc
        else:
            price_data = list(data_collector.eth_data)
            model = self.model_eth
            
        if len(price_data) < 100:
            return None
            
        indicators = data_collector.indicators_cache.get(symbol)
        if not indicators:
            return None
            
        # 准备特征
        features = self.prepare_features(indicators, price_data)
        if features is None or len(features) < 50:
            return None
            
        # 补齐特征维度
        if len(features) < 50:
            features = np.pad(features, (0, 50 - len(features)), 'constant')
        elif len(features) > 50:
            features = features[:50]
            
        # 准备序列数据
        sequence_length = 20
        sequences = []
        for i in range(min(sequence_length, len(price_data))):
            idx = -(sequence_length - i)
            seq_features = self.prepare_features(indicators, price_data[:idx])
            if seq_features is not None:
                if len(seq_features) < 50:
                    seq_features = np.pad(seq_features, (0, 50 - len(seq_features)), 'constant')
                elif len(seq_features) > 50:
                    seq_features = seq_features[:50]
                sequences.append(seq_features)
                
        if len(sequences) < sequence_length:
            # 补齐序列
            for _ in range(sequence_length - len(sequences)):
                sequences.insert(0, np.zeros(50))
                
        # 转换为张量
        x = torch.FloatTensor(sequences).unsqueeze(0)
        
        # 预测
        model.eval()
        with torch.no_grad():
            predictions = model(x).numpy()[0]
            
        current_price = price_data[-1]['close']
        current_time = datetime.now()
        
        # 检查入场条件
        should_enter, direction = self.check_entry_conditions(indicators, current_price)
        
        prediction = {
            'symbol': symbol,
            'current_price': current_price,
            'current_time': current_time.isoformat(),
            'predictions': {
                '10m': {
                    'up_probability': float(predictions[0]),
                    'down_probability': float(predictions[1]),
                    'direction': 'UP' if predictions[0] > 0.55 else 'DOWN' if predictions[1] > 0.55 else 'NEUTRAL'
                },
                '30m': {
                    'up_probability': float(predictions[2]),
                    'down_probability': float(predictions[3]),
                    'direction': 'UP' if predictions[2] > 0.55 else 'DOWN' if predictions[3] > 0.55 else 'NEUTRAL'
                }
            },
            'entry_signal': direction if should_enter else None,
            'confidence': max(predictions) if should_enter else 0
        }
        
        # 保存预测记录
        self.predictions_history.append(prediction)
        
        return prediction
    
    def verify_prediction(self, prediction, actual_price_10m, actual_price_30m):
        """验证预测结果并更新胜率"""
        symbol = 'BTC' if prediction['symbol'] == 'BTCUSDT' else 'ETH'
        current_price = prediction['current_price']
        
        # 验证10分钟预测
        if prediction['predictions']['10m']['direction'] == 'UP':
            if actual_price_10m > current_price:
                self.correct_predictions_10m[symbol] += 1
        elif prediction['predictions']['10m']['direction'] == 'DOWN':
            if actual_price_10m < current_price:
                self.correct_predictions_10m[symbol] += 1
                
        # 验证30分钟预测
        if prediction['predictions']['30m']['direction'] == 'UP':
            if actual_price_30m > current_price:
                self.correct_predictions_30m[symbol] += 1
        elif prediction['predictions']['30m']['direction'] == 'DOWN':
            if actual_price_30m < current_price:
                self.correct_predictions_30m[symbol] += 1
                
        self.total_predictions += 1
        
        # 更新胜率
        if self.total_predictions > 0:
            self.win_rate_10m[symbol] = self.correct_predictions_10m[symbol] / self.total_predictions * 100
            self.win_rate_30m[symbol] = self.correct_predictions_30m[symbol] / self.total_predictions * 100
            
    def optimize_model(self, symbol, training_data, labels):
        """优化模型参数"""
        if symbol == 'BTCUSDT':
            model = self.model_btc
        else:
            model = self.model_eth
            
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(training_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # 保存模型
        torch.save(model.state_dict(), f'model_{symbol}.pth')

# 全局实例
data_collector = TradingDataCollector()
predictor = TradingPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/prediction/<symbol>')
def get_prediction(symbol):
    prediction = predictor.generate_prediction(symbol, data_collector)
    if prediction:
        return jsonify(prediction)
    return jsonify({'error': 'Insufficient data'}), 400

@app.route('/api/stats')
def get_stats():
    stats = {
        'total_predictions': predictor.total_predictions,
        'win_rate_10m': predictor.win_rate_10m,
        'win_rate_30m': predictor.win_rate_30m,
        'predictions_history': predictor.predictions_history[-100:]  # 最近100条
    }
    return jsonify(stats)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    
@socketio.on('request_prediction')
def handle_prediction_request(data):
    symbol = data.get('symbol', 'BTCUSDT')
    prediction = predictor.generate_prediction(symbol, data_collector)
    if prediction:
        emit('prediction_update', prediction)

def continuous_prediction_loop():
    """持续预测循环"""
    while True:
        try:
            # 生成BTC预测
            btc_prediction = predictor.generate_prediction('BTCUSDT', data_collector)
            if btc_prediction:
                socketio.emit('prediction_update', btc_prediction)
                
            # 生成ETH预测
            eth_prediction = predictor.generate_prediction('ETHUSDT', data_collector)
            if eth_prediction:
                socketio.emit('prediction_update', eth_prediction)
                
            # 验证历史预测
            current_time = datetime.now()
            for pred in predictor.predictions_history:
                pred_time = datetime.fromisoformat(pred['current_time'])
                
                # 验证10分钟预测
                if (current_time - pred_time).seconds >= 600 and not pred.get('verified_10m'):
                    symbol = pred['symbol']
                    current_data = data_collector.btc_data if symbol == 'BTCUSDT' else data_collector.eth_data
                    if current_data:
                        actual_price = list(current_data)[-1]['close']
                        predictor.verify_prediction(pred, actual_price, actual_price)
                        pred['verified_10m'] = True
                        
                # 验证30分钟预测
                if (current_time - pred_time).seconds >= 1800 and not pred.get('verified_30m'):
                    symbol = pred['symbol']
                    current_data = data_collector.btc_data if symbol == 'BTCUSDT' else data_collector.eth_data
                    if current_data:
                        actual_price = list(current_data)[-1]['close']
                        predictor.verify_prediction(pred, actual_price, actual_price)
                        pred['verified_30m'] = True

            time.sleep(30)  # 每30秒更新一次

        except Exception as e:
            print(f"Prediction loop error: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # 获取历史数据
    print("Loading historical data...")
    btc_hist = data_collector.fetch_historical_data('BTCUSDT')
    eth_hist = data_collector.fetch_historical_data('ETHUSDT')
    
    # 初始化数据队列
    for _, row in btc_hist.iterrows():
        data_collector.btc_data.append({
            'time': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
        
    for _, row in eth_hist.iterrows():
        data_collector.eth_data.append({
            'time': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
    
    # 计算初始指标
    print("Calculating initial indicators...")
    data_collector.calculate_indicators('BTCUSDT')
    data_collector.calculate_indicators('ETHUSDT')
    
    # 启动WebSocket
    print("Starting WebSocket connections...")
    data_collector.start_websocket()
    
    # 启动预测循环
    threading.Thread(target=continuous_prediction_loop, daemon=True).start()
    
    # 启动Flask服务器
    print("Starting server on http://localhost:5000")
    socketio.run(app, debug=False, port=5000)