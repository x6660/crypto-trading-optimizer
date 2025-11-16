# 加密货币交易优化系统

一个基于深度学习的加密货币价格预测和交易信号生成系统，支持实时数据采集、技术分析和智能预测。

## 功能特性

- **实时数据采集**: 通过 Binance WebSocket 获取 BTC 和 ETH 的实时价格数据
- **技术指标计算**: 支持 20+ 种技术指标（MA、EMA、MACD、RSI、布林带等）
- **深度学习预测**: 基于 LSTM + Transformer 的混合神经网络模型
- **多时间框架**: 同时预测 10分钟 和 30分钟 的价格走势
- **Web 可视化界面**: 实时显示预测信号、价格图表和历史记录
- **自动优化**: 自动优化策略参数以提高预测准确率
- **实时推送**: 通过 WebSocket 实时推送预测结果到前端

## 系统要求

- Python 3.8+
- Windows / Linux / macOS
- 至少 4GB RAM
- 稳定的网络连接（用于连接 Binance API）

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/x6660/crypto-trading-optimizer.git
cd crypto-trading-optimizer
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装 TA-Lib

TA-Lib 是一个技术分析库，需要先安装 C 库：

**Windows:**
```bash
# 下载预编译的 wheel 文件
# 访问: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# 下载对应 Python 版本的 .whl 文件，然后安装：
pip install TA_Lib-0.4.27-cp311-cp311-win_amd64.whl
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

### 4. 安装 PyTorch

访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择适合你系统的版本：

```bash
# CPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU 版本 (如果有 NVIDIA 显卡)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 5. 安装其他依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动主服务器

```bash
python server.py
```

服务器启动后会：
1. 加载 BTC 和 ETH 的历史数据
2. 计算技术指标
3. 启动 WebSocket 连接获取实时数据
4. 启动预测循环
5. 在 http://localhost:5000 启动 Web 服务器

### 访问 Web 界面

在浏览器中打开: `http://localhost:5000`

界面功能：
- 实时显示 BTC/ETH 价格
- 10分钟和30分钟预测信号
- 价格走势图表
- 预测历史记录
- 胜率统计

### 启动自动优化器（可选）

```bash
python optimizer.py
```

优化器会：
- 每小时评估当前策略性能
- 如果胜率低于 60%，自动优化参数
- 保存性能历史到 `performance_history.json`
- 每30分钟生成性能报告

## 项目结构

```
crypto-trading-optimizer/
├── server.py              # 主服务器 - 数据采集、预测、Web服务
├── optimizer.py           # 自动优化器 - 参数优化
├── requirements.txt       # Python 依赖
├── templates/
│   └── index.html        # Web 前端界面
├── .gitignore
└── README.md
```

## 技术架构

### 数据采集
- **数据源**: Binance API
- **实时数据**: WebSocket 连接
- **历史数据**: REST API

### 技术指标
- 移动平均线 (SMA, EMA)
- 布林带 (Bollinger Bands)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Stochastic
- ATR, ADX, CCI, MFI, OBV, SAR, Williams %R
- Ichimoku Cloud
- VWAP

### 深度学习模型
- **架构**: LSTM + Transformer 混合模型
- **输入**: 50维特征向量（技术指标 + 价格变化）
- **输出**: 4个概率值（10分钟涨/跌，30分钟涨/跌）
- **优化器**: Adam
- **损失函数**: Binary Cross Entropy

### 入场信号生成
- 布林带策略
- MACD 交叉
- RSI 超买超卖
- 多指标共振确认

## API 接口

### 获取预测
```
GET /api/prediction/<symbol>
```
参数: `symbol` - BTCUSDT 或 ETHUSDT

返回:
```json
{
  "symbol": "BTCUSDT",
  "current_price": 50000.0,
  "predictions": {
    "10m": {
      "up_probability": 0.65,
      "down_probability": 0.35,
      "direction": "UP"
    },
    "30m": {
      "up_probability": 0.72,
      "down_probability": 0.28,
      "direction": "UP"
    }
  },
  "entry_signal": "LONG",
  "confidence": 0.72
}
```

### 获取统计信息
```
GET /api/stats
```

返回:
```json
{
  "total_predictions": 100,
  "win_rate_10m": {
    "BTC": 65.5,
    "ETH": 62.3
  },
  "win_rate_30m": {
    "BTC": 68.2,
    "ETH": 64.7
  },
  "predictions_history": [...]
}
```

## 注意事项

⚠️ **免责声明**: 本系统仅供学习和研究使用。加密货币交易存在高风险，请勿将此系统用于实际交易决策。

- 预测结果不保证准确性
- 历史表现不代表未来结果
- 请勿投入无法承受损失的资金
- 建议在模拟环境中测试

## 配置说明

系统会自动生成 `config.json` 配置文件，可调整参数：

```json
{
  "rsi_oversold": 30,
  "rsi_overbought": 70,
  "bb_period": 20,
  "bb_std": 2,
  "confidence_threshold": 0.55,
  "entry_threshold": 0.7
}
```

## 故障排除

### TA-Lib 安装失败
- Windows: 确保下载了正确 Python 版本的 wheel 文件
- Linux: 确保安装了 build-essential
- macOS: 使用 Homebrew 安装

### WebSocket 连接失败
- 检查网络连接
- 确认防火墙未阻止 WebSocket 连接
- Binance API 可能需要代理访问（中国大陆）

### 内存不足
- 减少历史数据量 (`limit` 参数)
- 减小模型大小 (`hidden_dim` 参数)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 作者

x6660 (ukeux@outlook.com)

---

⭐ 如果这个项目对你有帮助，请给它一个 Star！
