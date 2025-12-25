
import pandas as pd
import pandas_ta as ta
from typing import List
from .models import MarketRegimeInput, MarketRegimeOutput, PricePoint, LearnedPatterns, IndicatorSettings


def _classify_regime(indicators: dict, settings: IndicatorSettings) -> MarketRegimeOutput:
    """Internal logic for classifying regime based on pre-computed indicators."""

    explanation = []
    learned_patterns = LearnedPatterns()
    risk_notes = []
    regime = "undefined" # Default to undefined
    confidence = 0.0

    # Unpack indicator booleans
    is_ema_trend_up = indicators['is_ema_trend_up']
    is_ema_trend_down = indicators['is_ema_trend_down']
    is_strong_trend = indicators['is_strong_trend']
    is_bullish_momentum = indicators['is_bullish_momentum']
    is_bearish_momentum = indicators['is_bearish_momentum']
    is_weak_trend = indicators['is_weak_trend']
    is_in_band = indicators['is_in_band']
    is_ema_slope_flat = indicators['is_ema_slope_flat']

    # Scoring-based classification
    uptrend_signals = [is_ema_trend_up, is_strong_trend, is_bullish_momentum]
    downtrend_signals = [is_ema_trend_down, is_strong_trend, is_bearish_momentum]
    ranging_signals = [is_weak_trend, is_in_band, is_ema_slope_flat]

    scores = {
        "uptrend": sum(uptrend_signals),
        "downtrend": sum(downtrend_signals),
        "ranging": sum(ranging_signals)
    }

    max_score = max(scores.values())
    best_regimes = [r for r, s in scores.items() if s == max_score]

    if max_score >= 2 and len(best_regimes) == 1:
        regime = best_regimes[0]
        if regime == "uptrend":
            confidence = max_score / len(uptrend_signals)
            if is_ema_trend_up: explanation.append(f"EMA {settings.ema_fast} is above EMA {settings.ema_slow}.")
            if is_strong_trend: explanation.append(f"ADX at {indicators['adx']:.2f} indicates a strong trend.")
            if is_bullish_momentum: explanation.append(f"RSI at {indicators['rsi']:.2f} shows bullish momentum.")
            learned_patterns.trend_character = "steady_accumulation"
            learned_patterns.false_breakout_risk = "low" if confidence > 0.9 else "medium"
            learned_patterns.best_strategy_fit = ["trend_following", "pullback_entry"]
            risk_notes.append("Watch for RSI divergence as a sign of potential exhaustion.")

        elif regime == "downtrend":
            confidence = max_score / len(downtrend_signals)
            if is_ema_trend_down: explanation.append(f"EMA {settings.ema_fast} is below EMA {settings.ema_slow}.")
            if is_strong_trend: explanation.append(f"ADX at {indicators['adx']:.2f} indicates a strong trend.")
            if is_bearish_momentum: explanation.append(f"RSI at {indicators['rsi']:.2f} shows bearish momentum.")
            learned_patterns.trend_character = "consistent_distribution"
            learned_patterns.false_breakout_risk = "low" if confidence > 0.9 else "medium"
            learned_patterns.best_strategy_fit = ["trend_following", "short_selling"]
            risk_notes.append("Be cautious of sharp reversals if volume diminishes.")

        elif regime == "ranging":
            confidence = max_score / len(ranging_signals)
            if is_weak_trend: explanation.append(f"ADX at {indicators['adx']:.2f} suggests a lack of clear trend.")
            if is_in_band: explanation.append(f"Price is oscillating within a ±1.5% band of the 50-period SMA.")
            if is_ema_slope_flat: explanation.append("EMA slope is nearly flat, indicating consolidation.")
            learned_patterns.trend_character = "sideways_consolidation"
            learned_patterns.false_breakout_risk = "high"
            learned_patterns.best_strategy_fit = ["mean_reversion", "range_trading"]
            risk_notes.append("Avoid trend-following strategies. Watch for a breakout with volume expansion.")
    else:
        regime = "undefined"
        confidence = 0.3
        explanation.append("Market conditions are mixed and do not clearly fit any defined regime.")

    return MarketRegimeOutput(
        market_regime=regime, confidence=confidence, explanation=explanation,
        learned_patterns=learned_patterns, risk_notes=risk_notes
    )

def run_regime_analysis(request: MarketRegimeInput) -> MarketRegimeOutput:
    """
    Analyzes price history to determine the current market regime.
    """
    # Convert price history to a Pandas DataFrame
    price_history_dict = [p.model_dump() for p in request.price_history]
    df = pd.DataFrame(price_history_dict)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 1. Validate Learning Readiness
    min_data_points = request.indicators.ema_slow + 50  # Increased buffer for swing analysis
    if len(df) < min_data_points:
        return MarketRegimeOutput(
            learning_state="insufficient_data",
            confidence=0.2,
            reasoning=[f"Price history length < {min_data_points} bars, EMA and ADX cannot be computed reliably"]
        )

    # --- 2. Calculate Indicators ---
    df.ta.ema(length=request.indicators.ema_fast, append=True)
    df.ta.ema(length=request.indicators.ema_slow, append=True)
    df.ta.adx(length=request.indicators.adx_period, append=True)
    df.ta.atr(length=request.indicators.atr_period, append=True)
    df.ta.rsi(length=request.indicators.rsi_period, append=True)
    df.ta.sma(length=50, append=True)

    df.dropna(inplace=True)
    if df.empty:
        return MarketRegimeOutput(learning_state="insufficient_data", confidence=0.2, reasoning=["Not enough data for indicators."])

    latest = df.iloc[-1]
    ema_fast = latest[f'EMA_{request.indicators.ema_fast}']
    ema_slow = latest[f'EMA_{request.indicators.ema_slow}']
    adx = latest[f'ADX_{request.indicators.adx_period}']
    rsi = latest[f'RSI_{request.indicators.rsi_period}']

    # --- 4. Define Analysis Criteria ---
    is_ema_trend_up = ema_fast > ema_slow
    is_ema_trend_down = ema_fast < ema_slow
    ema_slope = (df[f'EMA_{request.indicators.ema_fast}'].diff() / df[f'EMA_{request.indicators.ema_fast}']).iloc[-1]
    is_ema_slope_flat = abs(ema_slope) < 0.0005
    is_strong_trend = adx > 25
    is_weak_trend = adx < 20
    is_bullish_momentum = rsi > 60
    is_bearish_momentum = rsi < 40
    atr_avg = df[f'ATRr_{request.indicators.atr_period}'].rolling(window=20).mean().iloc[-1]
    is_atr_spike = latest[f'ATRr_{request.indicators.atr_period}'] > 1.5 * atr_avg
    sma_50 = latest['SMA_50']
    is_in_band = (latest['close'] > sma_50 * 0.985) and (latest['close'] < sma_50 * 1.015)

    is_structure_break = False
    swing_highs = df['high'][(df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))][-50:]
    prior_swing_highs = swing_highs[swing_highs.index < df.index[-5]]
    if not prior_swing_highs.empty and df['high'][-5:].max() > prior_swing_highs.max():
        is_structure_break = True

    swing_lows = df['low'][(df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))][-50:]
    prior_swing_lows = swing_lows[swing_lows.index < df.index[-5]]
    if not prior_swing_lows.empty and df['low'][-5:].min() < prior_swing_lows.min():
        is_structure_break = True

    indicators = {
        'is_ema_trend_up': is_ema_trend_up, 'is_ema_trend_down': is_ema_trend_down,
        'is_strong_trend': is_strong_trend, 'is_bullish_momentum': is_bullish_momentum,
        'is_bearish_momentum': is_bearish_momentum, 'is_weak_trend': is_weak_trend,
        'is_in_band': is_in_band, 'is_ema_slope_flat': is_ema_slope_flat,
        'adx': adx, 'rsi': rsi
    }

    # --- 5. Classify Regime ---
    if is_atr_spike or is_structure_break:
        explanation = []
        if is_atr_spike: explanation.append(f"ATR has spiked, indicating a sharp increase in volatility.")
        if is_structure_break: explanation.append("Price has broken a recent swing structure, suggesting a potential change.")
        return MarketRegimeOutput(
            market_regime="volatile_transition", confidence=1.0, explanation=explanation,
            learned_patterns=LearnedPatterns(
                trend_character="unpredictable", false_breakout_risk="very_high",
                best_strategy_fit=["breakout_strategies", "volatility_trading"]
            ),
            risk_notes=["High risk of whipsaws. Position sizing should be reduced."]
        )

    return _classify_regime(indicators, request.indicators)
