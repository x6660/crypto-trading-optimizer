"""
DeepSeek AI 交易决策系统
基于用户提供的完整策略prompt整合AI分析能力
"""

import os
import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekAIAnalyzer:
    """DeepSeek AI 分析引擎"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY', 'sk-80849bf92e2b43f992b77a319910765d')
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.system_prompt = self._load_system_prompt()
        self.decision_history = []

    def _load_system_prompt(self) -> str:
        """
        加载完整的系统Prompt
        基于用户提供的详细策略要求
        """
        return """# 🎯 专业加密货币交易AI - DeepSeek 版本

你是一个专业的加密货币交易AI，专注于BTC和ETH的10分钟和30分钟价格预测。

## 核心目标

**最大化夏普比率（Sharpe Ratio）**

夏普比率 = 平均收益 / 收益波动率

这意味着：
- ✅ 高质量交易（高胜率、大盈亏比）→ 提升夏普
- ✅ 稳定收益、控制回撤 → 提升夏普
- ✅ 耐心持仓、让利润奔跑 → 提升夏普
- ❌ 频繁交易、小盈小亏 → 增加波动，严重降低夏普
- ❌ 过度交易、手续费损耗 → 直接亏损
- ❌ 过早平仓、频繁进出 → 错失大行情

**关键认知**: 系统每3分钟扫描一次，但不意味着每次都要交易！大多数时候应该是 wait 或 hold，只在极佳机会时才开仓。

---

## 零号原则：疑惑优先（最高优先级）

⚠️ **当你不确定时，默认选择 wait**

这是最高优先级原则，覆盖所有其他规则：
- **有任何疑虑** → 选 wait（不要尝试"勉强开仓"）
- **完全确定**（信心 ≥85 且无任何犹豫）→ 才开仓
- **不确定是否违反某条款** = 视为违反 → 选 wait
- **宁可错过机会，不做模糊决策**

### 自我检查

在输出决策前问自己：
1. 我是否 100% 确定这是高质量机会？
2. 如果用自己的钱，我会开这单吗？
3. 我能清楚说出 3 个开仓理由吗？

**3 个问题任一回答"否" → 选 wait**

---

## 决策流程（严格顺序）

### 第 0 步：疑惑检查
**在所有分析之前，先问自己：我对当前市场有清晰判断吗？**
- 若感到困惑、矛盾、不确定 → 直接输出 wait
- 若完全清晰 → 继续后续步骤

### 第 1 步：多周期趋势确认

开仓前必须同时检查 3分钟、15分钟、1小时、4小时 的K线形态：
- 若四个周期中至少三个周期的结构方向一致（如均为上升通道或EMA20>EMA50），则可顺势开仓
- 若多周期趋势方向不一致，必须等待趋势共振信号再开仓
- 若任意周期出现顶部或底部反转形态，禁止盲目开仓

### 第 2 步：BTC 状态确认（最关键）

⚠️ **BTC 是市场领导者，交易任何币种前必须先确认 BTC 状态**

分析 BTC 的多周期趋势方向：
- **15m MACD** 方向？（>0 多头，<0 空头）
- **1h MACD** 方向？
- **4h MACD** 方向？

**判断标准**：
- ✅ **BTC 多周期一致（3 个都 >0 或都 <0）** → BTC 状态明确
- ✅ **BTC 多周期中性（2 个同向，1 个反向）** → BTC 状态尚可
- ❌ **BTC 多周期矛盾（15m 多头但 1h/4h 空头）** → BTC 状态不明

**不通过 → 输出 wait，reasoning 写明"BTC 状态不明确"**

### 第 3 步：多空确认清单

⚠️ **至少 5/8 项一致才能开仓，4/8 不足**

#### 做多确认清单

| 指标 | 做多条件 |
|------|---------|
| MACD | >0（多头） |
| 价格 vs EMA20 | 价格 > EMA20 |
| RSI | <35（超卖反弹）或 35-50 |
| 成交量 | 放大（>1.5x 均量） |
| BTC 状态 | 多头或中性 |
| 布林带 | 触及下轨或中下区域 |
| 道氏理论 | 多周期共振看多 |
| 支撑位 | 价格接近关键支撑 |

#### 做空确认清单

| 指标 | 做空条件 |
|------|---------|
| MACD | <0（空头） |
| 价格 vs EMA20 | 价格 < EMA20 |
| RSI | >65（超买回落）或 50-65 |
| 成交量 | 放大（>1.5x 均量） |
| BTC 状态 | 空头或中性 |
| 布林带 | 触及上轨或中上区域 |
| 道氏理论 | 多周期共振看空 |
| 阻力位 | 价格接近关键阻力 |

**一致性不足 → 输出 wait**

### 第 4 步：防假突破检测

在开仓前额外检查以下假突破信号，若触发则禁止开仓：

#### 做多禁止条件
- ❌ **当前 K 线长上影 > 实体长度 × 2** → 上方抛压大，假突破概率高
- ❌ **价格突破但成交量萎缩（<均量 × 0.8）** → 缺乏动能，易回撤

#### 做空禁止条件
- ❌ **当前 K 线长下影 > 实体长度 × 2** → 下方承接力强，假跌破概率高
- ❌ **价格跌破但成交量萎缩（<均量 × 0.8）** → 缺乏动能，易反弹

### 第 5 步：计算信心度并评估机会

#### 信心度客观评分公式

**基础分：60 分**

#### 加分项（每项 +5 分，最高 100 分）

1. ✅ **多空确认清单 ≥5/8 项一致**：+5 分
2. ✅ **BTC 状态明确支持**：+5 分
3. ✅ **多时间框架共振**（15m/1h/4h MACD 同向）：+5 分
4. ✅ **强技术位明确**（1h/4h EMA20 或整数关口）：+5 分
5. ✅ **成交量确认**（放量 >1.5x 均量）：+5 分
6. ✅ **风险回报比 ≥1:4**：+5 分
7. ✅ **止盈技术位距离 2-5%**（理想范围）：+5 分
8. ✅ **道氏/波浪/江恩理论一致**：+5 分

#### 减分项（每项 -10 分）

1. ❌ **指标矛盾**（MACD vs 价格 或 RSI vs 成交量）：-10 分
2. ❌ **BTC 状态不明**（多周期矛盾）：-10 分
3. ❌ **技术位不清晰**：-10 分
4. ❌ **成交量萎缩**（<均量 × 0.7）：-10 分

#### 强制规则

- **信心度 <85** → 禁止开仓
- **信心度 85-90** → 可以开仓，但仓位保守
- **信心度 90-95** → 标准仓位
- **信心度 >95** → 可适度加大（慎用）

---

## 输出格式

你必须输出JSON格式，包含以下字段：

```json
{
  "symbol": "BTCUSDT",
  "action": "open_long",  // open_long | open_short | wait | hold
  "timeframe_10m": {
    "direction": "UP",  // UP | DOWN | NEUTRAL
    "confidence": 88,
    "reasoning": "详细理由"
  },
  "timeframe_30m": {
    "direction": "UP",
    "confidence": 85,
    "reasoning": "详细理由"
  },
  "entry_price": 50000.0,
  "stop_loss": 49500.0,
  "take_profit": 51000.0,
  "risk_reward_ratio": 4.0,
  "multi_cycle_check": {
    "3m": "BULLISH",
    "15m": "BULLISH",
    "1h": "BULLISH",
    "4h": "BULLISH"
  },
  "indicators_check": {
    "macd": "BULLISH",
    "rsi": "NEUTRAL",
    "bollinger": "OVERSOLD",
    "volume": "INCREASING",
    "support_resistance": "NEAR_SUPPORT"
  },
  "overall_confidence": 88,
  "detailed_reasoning": "完整的思维链分析，说明为什么做出这个决策..."
}
```

---

## 核心原则

1. **不要追多 也不要追空** - 等待回调至技术位
2. **突破压力位考虑回弹做多** - 但要确认回踩不破
3. **跌破支撑考虑回弹做空** - 但要确认反弹不破
4. **结合布林带** - 上轨做空，下轨做多
5. **盈利如果要跌回去信号，就平仓** - 保护利润
6. **可以移动止盈止损** - 动态追踪

记住: 你在用真金白银交易真实市场。每个决策都有后果。系统化交易，严格管理风险，让概率随时间为你服务。
"""

    def analyze_market(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict,
        multi_timeframe_data: Dict,
        historical_performance: Dict = None
    ) -> Dict:
        """
        使用DeepSeek AI进行市场分析

        Args:
            symbol: 交易对
            current_price: 当前价格
            indicators: 技术指标字典
            multi_timeframe_data: 多周期数据
            historical_performance: 历史表现数据

        Returns:
            AI分析结果
        """
        try:
            # 构建用户消息
            user_message = self._build_analysis_prompt(
                symbol, current_price, indicators, multi_timeframe_data, historical_performance
            )

            # 调用DeepSeek API
            logger.info(f"Calling DeepSeek API for {symbol}...")
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # 解析响应
            ai_response = response.choices[0].message.content
            logger.info(f"DeepSeek AI响应: {ai_response[:200]}...")

            # 尝试解析JSON
            try:
                # 寻找JSON块
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = ai_response[json_start:json_end]
                    analysis = json.loads(json_str)
                else:
                    # 如果找不到JSON，创建默认响应
                    analysis = self._create_default_analysis(symbol, ai_response)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from AI response, using default analysis")
                analysis = self._create_default_analysis(symbol, ai_response)

            # 添加元数据
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['raw_response'] = ai_response

            # 保存到历史
            self.decision_history.append(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_analysis(symbol, str(e))

    def _build_analysis_prompt(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict,
        multi_timeframe_data: Dict,
        historical_performance: Dict = None
    ) -> str:
        """构建发送给AI的分析prompt"""

        prompt = f"""# 市场分析请求

## 基础信息
- **交易对**: {symbol}
- **当前价格**: ${current_price:,.2f}
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 多周期数据分析
"""

        # 添加多周期数据
        for timeframe, data in multi_timeframe_data.items():
            prompt += f"\n### {timeframe} 周期\n"
            if data and isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float, np.number)):
                        prompt += f"- {key}: {value:.4f}\n"

        # 添加关键技术指标
        prompt += "\n## 关键技术指标\n"

        # 趋势指标
        if 'EMA_20' in indicators and len(indicators['EMA_20']) > 0:
            ema20 = indicators['EMA_20'][-1]
            if not np.isnan(ema20):
                prompt += f"- **EMA20**: ${ema20:.2f} (价格{'高于' if current_price > ema20 else '低于'}均线)\n"

        if 'EMA_50' in indicators and len(indicators['EMA_50']) > 0:
            ema50 = indicators['EMA_50'][-1]
            if not np.isnan(ema50):
                prompt += f"- **EMA50**: ${ema50:.2f}\n"

        # MACD
        if 'MACD' in indicators and 'MACD_signal' in indicators:
            macd = indicators['MACD']
            macd_signal = indicators['MACD_signal']
            if len(macd) > 0 and len(macd_signal) > 0:
                if not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
                    prompt += f"- **MACD**: {macd[-1]:.4f}, Signal: {macd_signal[-1]:.4f}\n"

        # RSI
        if 'RSI_14' in indicators and len(indicators['RSI_14']) > 0:
            rsi = indicators['RSI_14'][-1]
            if not np.isnan(rsi):
                rsi_status = '超卖' if rsi < 30 else ('超买' if rsi > 70 else '中性')
                prompt += f"- **RSI**: {rsi:.2f} ({rsi_status})\n"

        # 布林带
        if all(k in indicators for k in ['BB_upper_20_2', 'BB_middle_20_2', 'BB_lower_20_2']):
            bb_upper = indicators['BB_upper_20_2']
            bb_middle = indicators['BB_middle_20_2']
            bb_lower = indicators['BB_lower_20_2']
            if len(bb_upper) > 0 and len(bb_lower) > 0:
                if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
                    bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) * 100
                    prompt += f"- **布林带**: 上轨${bb_upper[-1]:.2f}, 中轨${bb_middle[-1]:.2f}, 下轨${bb_lower[-1]:.2f}\n"
                    prompt += f"  价格位置: {bb_position:.1f}% ({'接近上轨' if bb_position > 80 else ('接近下轨' if bb_position < 20 else '中性')})\n"

        # 成交量
        if 'VOLUME_RATIO' in indicators and len(indicators['VOLUME_RATIO']) > 0:
            vol_ratio = indicators['VOLUME_RATIO'][-1]
            if not np.isnan(vol_ratio):
                prompt += f"- **成交量比率**: {vol_ratio:.2f}x 平均量\n"

        # 支撑阻力
        if 'SUPPORT_LEVEL' in indicators:
            prompt += f"- **支撑位**: ${indicators['SUPPORT_LEVEL']:.2f}\n"
        if 'RESISTANCE_LEVEL' in indicators:
            prompt += f"- **阻力位**: ${indicators['RESISTANCE_LEVEL']:.2f}\n"

        # 道氏理论
        if 'DOW_resonance' in indicators:
            prompt += f"- **道氏理论共振**: {indicators['DOW_resonance']}\n"

        # 波浪理论
        if 'ELLIOTT_current_wave' in indicators:
            prompt += f"- **波浪理论**: {indicators['ELLIOTT_current_wave']} - {indicators.get('ELLIOTT_direction', 'N/A')}\n"

        # 江恩理论
        if 'GANN_position' in indicators:
            prompt += f"- **江恩位置**: {indicators['GANN_position']}\n"

        # 历史表现
        if historical_performance:
            prompt += "\n## 历史表现参考\n"
            prompt += f"- 总交易次数: {historical_performance.get('total_trades', 0)}\n"
            prompt += f"- 10分钟胜率: {historical_performance.get('win_rate_10m', 0):.1f}%\n"
            prompt += f"- 30分钟胜率: {historical_performance.get('win_rate_30m', 0):.1f}%\n"
            prompt += f"- 夏普比率: {historical_performance.get('sharpe_ratio', 0):.2f}\n"

        prompt += """

## 任务要求

请基于以上数据，进行深度分析并给出：

1. **10分钟预测**: 价格是涨还是跌？信心度多少？
2. **30分钟预测**: 价格是涨还是跌？信心度多少？
3. **交易建议**: 现在应该开仓吗？做多还是做空？还是观望？
4. **进场点位**: 如果开仓，在什么价位进场最佳？
5. **止损止盈**: 合理的止损和止盈价格是多少？
6. **风险评估**: 这个交易的风险回报比如何？

**严格按照JSON格式输出**，并提供详细的思维链分析。

记住：
- ❌ 不要追多追空
- ✅ 等待技术位回踩/反弹
- ✅ 多空确认清单至少5/8项一致
- ✅ BTC状态必须明确
- ✅ 信心度必须≥85才开仓
- ✅ 风险回报比必须≥1:4
"""

        return prompt

    def _create_default_analysis(self, symbol: str, ai_response: str) -> Dict:
        """创建默认分析结果"""
        return {
            "symbol": symbol,
            "action": "wait",
            "timeframe_10m": {
                "direction": "NEUTRAL",
                "confidence": 50,
                "reasoning": "AI响应解析失败，默认观望"
            },
            "timeframe_30m": {
                "direction": "NEUTRAL",
                "confidence": 50,
                "reasoning": "AI响应解析失败，默认观望"
            },
            "overall_confidence": 50,
            "detailed_reasoning": ai_response
        }

    def _create_error_analysis(self, symbol: str, error_msg: str) -> Dict:
        """创建错误分析结果"""
        return {
            "symbol": symbol,
            "action": "wait",
            "timeframe_10m": {
                "direction": "NEUTRAL",
                "confidence": 0,
                "reasoning": f"分析错误: {error_msg}"
            },
            "timeframe_30m": {
                "direction": "NEUTRAL",
                "confidence": 0,
                "reasoning": f"分析错误: {error_msg}"
            },
            "overall_confidence": 0,
            "detailed_reasoning": f"系统错误，无法进行分析: {error_msg}",
            "error": error_msg
        }

    def get_decision_history(self, limit: int = 100) -> List[Dict]:
        """获取决策历史"""
        return self.decision_history[-limit:]


# 全局实例
deepseek_analyzer = DeepSeekAIAnalyzer()
