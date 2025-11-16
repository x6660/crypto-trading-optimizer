"""
完整的技术分析引擎
整合道氏理论、波浪理论、江恩理论及所有主流技术指标
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveTechnicalAnalysis:
    """综合技术分析类 - 整合所有专业技术分析方法"""

    def __init__(self):
        self.indicators_cache = {}
        self.support_resistance_levels = {}

    def calculate_all_indicators(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        计算所有技术指标

        Args:
            df: 包含OHLCV数据的DataFrame
            symbol: 交易对符号

        Returns:
            包含所有指标的字典
        """
        if df is None or len(df) < 100:
            return {}

        indicators = {}

        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            open_price = df['open'].values
            volume = df['volume'].values

            # ============ 趋势指标 ============
            indicators.update(self._calculate_trend_indicators(close, high, low, open_price))

            # ============ 动量指标 ============
            indicators.update(self._calculate_momentum_indicators(close, high, low))

            # ============ 波动率指标 ============
            indicators.update(self._calculate_volatility_indicators(close, high, low))

            # ============ 成交量指标 ============
            indicators.update(self._calculate_volume_indicators(close, high, low, volume))

            # ============ 支撑阻力位 ============
            indicators.update(self._calculate_support_resistance(df))

            # ============ K线形态识别 ============
            indicators.update(self._recognize_candlestick_patterns(open_price, high, low, close))

            # ============ 道氏理论分析 ============
            indicators.update(self._dow_theory_analysis(df))

            # ============ 波浪理论分析 ============
            indicators.update(self._elliott_wave_analysis(close))

            # ============ 江恩理论分析 ============
            indicators.update(self._gann_theory_analysis(df))

            # ============ 斐波那契分析 ============
            indicators.update(self._fibonacci_analysis(close))

            # 缓存结果
            self.indicators_cache[symbol] = indicators

            logger.info(f"Calculated {len(indicators)} indicators for {symbol}")
            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def _calculate_trend_indicators(self, close, high, low, open_price) -> Dict:
        """计算趋势指标"""
        indicators = {}

        try:
            # 移动平均线
            for period in [5, 10, 20, 30, 50, 100, 200]:
                indicators[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
                indicators[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)

            # MACD
            indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )

            # ADX - 趋势强度
            indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            indicators['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            indicators['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # Parabolic SAR
            indicators['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

            # Aroon
            indicators['AROON_down'], indicators['AROON_up'] = talib.AROON(high, low, timeperiod=25)
            indicators['AROON_OSC'] = talib.AROONOSC(high, low, timeperiod=25)

            # 一目均衡表 (Ichimoku)
            indicators.update(self._calculate_ichimoku(high, low, close))

        except Exception as e:
            logger.error(f"Error in trend indicators: {e}")

        return indicators

    def _calculate_momentum_indicators(self, close, high, low) -> Dict:
        """计算动量指标"""
        indicators = {}

        try:
            # RSI - 多周期
            for period in [7, 14, 21, 28]:
                indicators[f'RSI_{period}'] = talib.RSI(close, timeperiod=period)

            # Stochastic
            indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(
                high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
            )

            # Stochastic RSI
            indicators['STOCHRSI_K'], indicators['STOCHRSI_D'] = talib.STOCHRSI(
                close, timeperiod=14, fastk_period=5, fastd_period=3
            )

            # Williams %R
            for period in [14, 28]:
                indicators[f'WILLR_{period}'] = talib.WILLR(high, low, close, timeperiod=period)

            # CCI - Commodity Channel Index
            for period in [14, 20]:
                indicators[f'CCI_{period}'] = talib.CCI(high, low, close, timeperiod=period)

            # ROC - Rate of Change
            for period in [9, 12, 25]:
                indicators[f'ROC_{period}'] = talib.ROC(close, timeperiod=period)

            # MOM - Momentum
            indicators['MOM'] = talib.MOM(close, timeperiod=10)

            # Ultimate Oscillator
            indicators['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        except Exception as e:
            logger.error(f"Error in momentum indicators: {e}")

        return indicators

    def _calculate_volatility_indicators(self, close, high, low) -> Dict:
        """计算波动率指标"""
        indicators = {}

        try:
            # ATR - Average True Range
            for period in [7, 14, 21]:
                indicators[f'ATR_{period}'] = talib.ATR(high, low, close, timeperiod=period)

            # 布林带 - 多周期
            for period, std in [(20, 2), (20, 2.5), (20, 3), (50, 2)]:
                upper, middle, lower = talib.BBANDS(
                    close, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0
                )
                indicators[f'BB_upper_{period}_{std}'] = upper
                indicators[f'BB_middle_{period}_{std}'] = middle
                indicators[f'BB_lower_{period}_{std}'] = lower
                indicators[f'BB_width_{period}_{std}'] = (upper - lower) / middle * 100

                # 布林带位置百分比
                indicators[f'BB_percent_{period}_{std}'] = (close - lower) / (upper - lower) * 100

            # Keltner Channel
            ema = talib.EMA(close, timeperiod=20)
            atr = talib.ATR(high, low, close, timeperiod=10)
            indicators['KELTNER_upper'] = ema + (2 * atr)
            indicators['KELTNER_middle'] = ema
            indicators['KELTNER_lower'] = ema - (2 * atr)

            # Donchian Channel
            indicators['DONCHIAN_upper'] = pd.Series(high).rolling(window=20).max().values
            indicators['DONCHIAN_lower'] = pd.Series(low).rolling(window=20).min().values
            indicators['DONCHIAN_middle'] = (indicators['DONCHIAN_upper'] + indicators['DONCHIAN_lower']) / 2

        except Exception as e:
            logger.error(f"Error in volatility indicators: {e}")

        return indicators

    def _calculate_volume_indicators(self, close, high, low, volume) -> Dict:
        """计算成交量指标"""
        indicators = {}

        try:
            # OBV - On Balance Volume
            indicators['OBV'] = talib.OBV(close, volume)

            # AD - Accumulation/Distribution
            indicators['AD'] = talib.AD(high, low, close, volume)

            # ADOSC - Chaikin A/D Oscillator
            indicators['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

            # MFI - Money Flow Index
            indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)

            # VWAP - Volume Weighted Average Price
            indicators['VWAP'] = (close * volume).cumsum() / volume.cumsum()

            # 成交量移动平均
            for period in [5, 10, 20]:
                indicators[f'VOLUME_MA_{period}'] = talib.SMA(volume, timeperiod=period)

            # 成交量比率
            volume_ma_20 = talib.SMA(volume, timeperiod=20)
            indicators['VOLUME_RATIO'] = volume / volume_ma_20

        except Exception as e:
            logger.error(f"Error in volume indicators: {e}")

        return indicators

    def _calculate_ichimoku(self, high, low, close) -> Dict:
        """计算一目均衡表"""
        indicators = {}

        try:
            # 转换线 (Tenkan-sen): (9日最高+9日最低)/2
            period9_high = pd.Series(high).rolling(window=9).max()
            period9_low = pd.Series(low).rolling(window=9).min()
            indicators['ICHIMOKU_tenkan'] = (period9_high + period9_low) / 2

            # 基准线 (Kijun-sen): (26日最高+26日最低)/2
            period26_high = pd.Series(high).rolling(window=26).max()
            period26_low = pd.Series(low).rolling(window=26).min()
            indicators['ICHIMOKU_kijun'] = (period26_high + period26_low) / 2

            # 先行带A (Senkou Span A): (转换线+基准线)/2
            indicators['ICHIMOKU_senkou_a'] = (indicators['ICHIMOKU_tenkan'] + indicators['ICHIMOKU_kijun']) / 2

            # 先行带B (Senkou Span B): (52日最高+52日最低)/2
            period52_high = pd.Series(high).rolling(window=52).max()
            period52_low = pd.Series(low).rolling(window=52).min()
            indicators['ICHIMOKU_senkou_b'] = (period52_high + period52_low) / 2

            # 滞后线 (Chikou Span): 当前收盘价
            indicators['ICHIMOKU_chikou'] = close

        except Exception as e:
            logger.error(f"Error in Ichimoku: {e}")

        return indicators

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """计算支撑阻力位"""
        indicators = {}

        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # Pivot Points (标准)
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            indicators['PIVOT'] = pivot
            indicators['R1'] = 2 * pivot - low[-1]
            indicators['R2'] = pivot + (high[-1] - low[-1])
            indicators['R3'] = high[-1] + 2 * (pivot - low[-1])
            indicators['S1'] = 2 * pivot - high[-1]
            indicators['S2'] = pivot - (high[-1] - low[-1])
            indicators['S3'] = low[-1] - 2 * (high[-1] - pivot)

            # Fibonacci Pivot Points
            range_price = high[-1] - low[-1]
            indicators['FIB_R1'] = pivot + 0.382 * range_price
            indicators['FIB_R2'] = pivot + 0.618 * range_price
            indicators['FIB_R3'] = pivot + 1.0 * range_price
            indicators['FIB_S1'] = pivot - 0.382 * range_price
            indicators['FIB_S2'] = pivot - 0.618 * range_price
            indicators['FIB_S3'] = pivot - 1.0 * range_price

            # 识别局部高低点
            window = 10
            local_maxima = []
            local_minima = []

            for i in range(window, len(close) - window):
                if close[i] == max(close[i-window:i+window+1]):
                    local_maxima.append(close[i])
                if close[i] == min(close[i-window:i+window+1]):
                    local_minima.append(close[i])

            if local_maxima:
                indicators['RESISTANCE_LEVEL'] = np.mean(sorted(local_maxima)[-3:])
            if local_minima:
                indicators['SUPPORT_LEVEL'] = np.mean(sorted(local_minima)[:3])

        except Exception as e:
            logger.error(f"Error in support/resistance: {e}")

        return indicators

    def _recognize_candlestick_patterns(self, open_price, high, low, close) -> Dict:
        """识别K线形态"""
        patterns = {}

        try:
            # 主要K线形态
            pattern_functions = {
                'DOJI': talib.CDLDOJI,
                'HAMMER': talib.CDLHAMMER,
                'INVERTED_HAMMER': talib.CDLINVERTEDHAMMER,
                'HANGING_MAN': talib.CDLHANGINGMAN,
                'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
                'ENGULFING': talib.CDLENGULFING,
                'HARAMI': talib.CDLHARAMI,
                'PIERCING': talib.CDLPIERCING,
                'DARK_CLOUD': talib.CDLDARKCLOUDCOVER,
                'MORNING_STAR': talib.CDLMORNINGSTAR,
                'EVENING_STAR': talib.CDLEVENINGSTAR,
                'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS,
                'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS,
                'SPINNING_TOP': talib.CDLSPINNINGTOP,
                'MARUBOZU': talib.CDLMARUBOZU,
            }

            for name, func in pattern_functions.items():
                try:
                    patterns[f'PATTERN_{name}'] = func(open_price, high, low, close)
                except:
                    pass

        except Exception as e:
            logger.error(f"Error in candlestick patterns: {e}")

        return patterns

    def _dow_theory_analysis(self, df: pd.DataFrame) -> Dict:
        """道氏理论分析"""
        analysis = {}

        try:
            close = df['close'].values

            # 识别主要趋势 (Primary Trend)
            # 使用200日均线作为主要趋势判断
            sma_200 = talib.SMA(close, timeperiod=200)
            if len(sma_200) > 0 and not np.isnan(sma_200[-1]):
                if close[-1] > sma_200[-1]:
                    analysis['DOW_primary_trend'] = 'BULLISH'
                else:
                    analysis['DOW_primary_trend'] = 'BEARISH'

            # 识别次要趋势 (Secondary Trend)
            # 使用50日均线作为次要趋势判断
            sma_50 = talib.SMA(close, timeperiod=50)
            if len(sma_50) > 0 and not np.isnan(sma_50[-1]):
                if close[-1] > sma_50[-1]:
                    analysis['DOW_secondary_trend'] = 'BULLISH'
                else:
                    analysis['DOW_secondary_trend'] = 'BEARISH'

            # 识别短期波动 (Minor Trend)
            # 使用20日均线作为短期趋势判断
            sma_20 = talib.SMA(close, timeperiod=20)
            if len(sma_20) > 0 and not np.isnan(sma_20[-1]):
                if close[-1] > sma_20[-1]:
                    analysis['DOW_minor_trend'] = 'BULLISH'
                else:
                    analysis['DOW_minor_trend'] = 'BEARISH'

            # 趋势共振度
            trends = [
                analysis.get('DOW_primary_trend'),
                analysis.get('DOW_secondary_trend'),
                analysis.get('DOW_minor_trend')
            ]
            bullish_count = trends.count('BULLISH')
            bearish_count = trends.count('BEARISH')

            if bullish_count == 3:
                analysis['DOW_resonance'] = 'STRONG_BULLISH'
            elif bullish_count == 2:
                analysis['DOW_resonance'] = 'MODERATE_BULLISH'
            elif bearish_count == 3:
                analysis['DOW_resonance'] = 'STRONG_BEARISH'
            elif bearish_count == 2:
                analysis['DOW_resonance'] = 'MODERATE_BEARISH'
            else:
                analysis['DOW_resonance'] = 'NEUTRAL'

        except Exception as e:
            logger.error(f"Error in Dow theory analysis: {e}")

        return analysis

    def _elliott_wave_analysis(self, close) -> Dict:
        """波浪理论分析 (简化版)"""
        analysis = {}

        try:
            # 简化的波浪识别
            # 寻找波峰波谷
            window = 20
            peaks = []
            troughs = []

            for i in range(window, len(close) - window):
                if close[i] == max(close[i-window:i+window+1]):
                    peaks.append(i)
                if close[i] == min(close[i-window:i+window+1]):
                    troughs.append(i)

            # 判断当前阶段
            if len(peaks) >= 2 and len(troughs) >= 2:
                last_peak = peaks[-1] if peaks else 0
                last_trough = troughs[-1] if troughs else 0

                if last_peak > last_trough:
                    analysis['ELLIOTT_current_wave'] = 'CORRECTIVE'
                    analysis['ELLIOTT_direction'] = 'DOWN'
                else:
                    analysis['ELLIOTT_current_wave'] = 'IMPULSIVE'
                    analysis['ELLIOTT_direction'] = 'UP'
            else:
                analysis['ELLIOTT_current_wave'] = 'UNCLEAR'
                analysis['ELLIOTT_direction'] = 'NEUTRAL'

        except Exception as e:
            logger.error(f"Error in Elliott wave analysis: {e}")

        return analysis

    def _gann_theory_analysis(self, df: pd.DataFrame) -> Dict:
        """江恩理论分析"""
        analysis = {}

        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # 江恩角度线 (简化版)
            # 1x1线: 45度角
            lookback = min(50, len(close))
            if lookback > 0:
                start_price = close[-lookback]
                current_price = close[-1]
                price_change = current_price - start_price

                # 理想的1x1线变化
                ideal_change = lookback  # 45度角

                if abs(price_change) > ideal_change * 0.8:
                    if price_change > 0:
                        analysis['GANN_trend_strength'] = 'STRONG_BULLISH'
                    else:
                        analysis['GANN_trend_strength'] = 'STRONG_BEARISH'
                else:
                    analysis['GANN_trend_strength'] = 'WEAK'

            # 江恩扇形线
            recent_high = max(high[-20:])
            recent_low = min(low[-20:])
            range_price = recent_high - recent_low

            # 计算关键价位
            analysis['GANN_382_level'] = recent_low + 0.382 * range_price
            analysis['GANN_50_level'] = recent_low + 0.5 * range_price
            analysis['GANN_618_level'] = recent_low + 0.618 * range_price

            # 判断当前价格位置
            current_price = close[-1]
            if current_price > analysis['GANN_618_level']:
                analysis['GANN_position'] = 'UPPER_ZONE'
            elif current_price < analysis['GANN_382_level']:
                analysis['GANN_position'] = 'LOWER_ZONE'
            else:
                analysis['GANN_position'] = 'MIDDLE_ZONE'

        except Exception as e:
            logger.error(f"Error in Gann theory analysis: {e}")

        return analysis

    def _fibonacci_analysis(self, close) -> Dict:
        """斐波那契分析"""
        analysis = {}

        try:
            # 找到最近的高点和低点
            lookback = min(100, len(close))
            recent_high = max(close[-lookback:])
            recent_low = min(close[-lookback:])
            range_price = recent_high - recent_low

            # 斐波那契回撤位
            analysis['FIB_0'] = recent_high
            analysis['FIB_236'] = recent_high - 0.236 * range_price
            analysis['FIB_382'] = recent_high - 0.382 * range_price
            analysis['FIB_50'] = recent_high - 0.5 * range_price
            analysis['FIB_618'] = recent_high - 0.618 * range_price
            analysis['FIB_786'] = recent_high - 0.786 * range_price
            analysis['FIB_100'] = recent_low

            # 斐波那契扩展位
            analysis['FIB_EXT_1272'] = recent_high + 0.272 * range_price
            analysis['FIB_EXT_1618'] = recent_high + 0.618 * range_price
            analysis['FIB_EXT_2618'] = recent_high + 1.618 * range_price

            # 当前价格位置分析
            current_price = close[-1]

            if current_price >= analysis['FIB_236']:
                analysis['FIB_zone'] = 'STRONG_ZONE'
            elif current_price >= analysis['FIB_50']:
                analysis['FIB_zone'] = 'MODERATE_ZONE'
            elif current_price >= analysis['FIB_786']:
                analysis['FIB_zone'] = 'WEAK_ZONE'
            else:
                analysis['FIB_zone'] = 'REVERSAL_ZONE'

        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")

        return analysis

    def generate_signal_summary(self, indicators: Dict, current_price: float) -> Dict:
        """
        生成综合信号摘要

        Returns:
            包含多空信号、信心度等的字典
        """
        summary = {
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'overall_direction': 'NEUTRAL',
            'confidence': 0,
            'key_levels': {},
            'warnings': []
        }

        try:
            # 趋势信号
            if 'EMA_20' in indicators and len(indicators['EMA_20']) > 0:
                ema20 = indicators['EMA_20'][-1]
                if not np.isnan(ema20):
                    if current_price > ema20:
                        summary['bullish_signals'] += 1
                    else:
                        summary['bearish_signals'] += 1

            if 'EMA_50' in indicators and len(indicators['EMA_50']) > 0:
                ema50 = indicators['EMA_50'][-1]
                if not np.isnan(ema50):
                    if current_price > ema50:
                        summary['bullish_signals'] += 1
                    else:
                        summary['bearish_signals'] += 1

            # MACD信号
            if 'MACD' in indicators and 'MACD_signal' in indicators:
                macd = indicators['MACD']
                macd_signal = indicators['MACD_signal']
                if len(macd) > 0 and len(macd_signal) > 0:
                    if not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
                        if macd[-1] > macd_signal[-1]:
                            summary['bullish_signals'] += 1
                        else:
                            summary['bearish_signals'] += 1

            # RSI信号
            if 'RSI_14' in indicators and len(indicators['RSI_14']) > 0:
                rsi = indicators['RSI_14'][-1]
                if not np.isnan(rsi):
                    if rsi < 30:
                        summary['bullish_signals'] += 2  # 超卖是强烈买入信号
                        summary['warnings'].append('RSI超卖')
                    elif rsi > 70:
                        summary['bearish_signals'] += 2  # 超买是强烈卖出信号
                        summary['warnings'].append('RSI超买')
                    elif rsi < 50:
                        summary['bearish_signals'] += 1
                    else:
                        summary['bullish_signals'] += 1

            # 布林带信号
            if 'BB_upper_20_2' in indicators and 'BB_lower_20_2' in indicators:
                bb_upper = indicators['BB_upper_20_2']
                bb_lower = indicators['BB_lower_20_2']
                if len(bb_upper) > 0 and len(bb_lower) > 0:
                    if not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
                        if current_price <= bb_lower[-1] * 1.001:
                            summary['bullish_signals'] += 2
                            summary['warnings'].append('触及布林带下轨')
                        elif current_price >= bb_upper[-1] * 0.999:
                            summary['bearish_signals'] += 2
                            summary['warnings'].append('触及布林带上轨')

            # 道氏理论共振
            if 'DOW_resonance' in indicators:
                resonance = indicators['DOW_resonance']
                if resonance == 'STRONG_BULLISH':
                    summary['bullish_signals'] += 3
                elif resonance == 'STRONG_BEARISH':
                    summary['bearish_signals'] += 3
                elif resonance == 'MODERATE_BULLISH':
                    summary['bullish_signals'] += 1
                elif resonance == 'MODERATE_BEARISH':
                    summary['bearish_signals'] += 1

            # 计算总体方向和信心度
            total_signals = summary['bullish_signals'] + summary['bearish_signals'] + summary['neutral_signals']
            if total_signals > 0:
                if summary['bullish_signals'] > summary['bearish_signals']:
                    summary['overall_direction'] = 'BULLISH'
                    summary['confidence'] = int((summary['bullish_signals'] / total_signals) * 100)
                elif summary['bearish_signals'] > summary['bullish_signals']:
                    summary['overall_direction'] = 'BEARISH'
                    summary['confidence'] = int((summary['bearish_signals'] / total_signals) * 100)
                else:
                    summary['overall_direction'] = 'NEUTRAL'
                    summary['confidence'] = 50

            # 关键价位
            if 'RESISTANCE_LEVEL' in indicators:
                summary['key_levels']['resistance'] = indicators['RESISTANCE_LEVEL']
            if 'SUPPORT_LEVEL' in indicators:
                summary['key_levels']['support'] = indicators['SUPPORT_LEVEL']
            if 'PIVOT' in indicators:
                summary['key_levels']['pivot'] = indicators['PIVOT']

        except Exception as e:
            logger.error(f"Error generating signal summary: {e}")

        return summary


# 全局实例
technical_analyzer = ComprehensiveTechnicalAnalysis()
