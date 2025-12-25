
import unittest
from datetime import datetime, timedelta
from learning_agent.models import MarketRegimeInput, IndicatorSettings, PricePoint
from learning_agent.regime_logic import run_regime_analysis

class TestMarketRegimeLogic(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.indicator_settings = IndicatorSettings(
            ema_fast=10,
            ema_slow=20,
            adx_period=14,
            atr_period=14,
            rsi_period=14
        )

    def _generate_price_data(self, count, start_price, trend):
        """Helper to generate sample price data."""
        prices = []
        current_price = start_price
        for i in range(count):
            timestamp = (datetime.now() - timedelta(days=count - i)).isoformat()
            open_p = current_price
            high_p = open_p + 1 + (trend * i * 0.1)
            low_p = open_p - 1 + (trend * i * 0.1)
            close_p = open_p + (trend * i * 0.1)
            volume = 100000 + (i * 1000)
            prices.append(PricePoint(timestamp=timestamp, open=open_p, high=high_p, low=low_p, close=close_p, volume=volume))
            current_price = close_p
        return prices

    def test_insufficient_data(self):
        """Test the agent's response to insufficient price history."""
        price_history = self._generate_price_data(10, 100, 0)
        request = MarketRegimeInput(
            symbol="TEST",
            timeframe="1D",
            price_history=price_history,
            indicators=self.indicator_settings
        )
        result = run_regime_analysis(request)
        self.assertEqual(result.learning_state, "insufficient_data")
        self.assertEqual(result.confidence, 0.2)

    def test_uptrend_regime(self):
        """Test a clear uptrend scenario."""
        # Generate data with a clear upward trend
        price_history = self._generate_price_data(100, 100, 1)
        # Manually adjust the last data point to ensure strong RSI/ADX
        last_price = price_history[-1].close
        price_history[-1].high = last_price * 1.05
        price_history[-1].close = last_price * 1.04

        request = MarketRegimeInput(
            symbol="TEST",
            timeframe="1D",
            price_history=price_history,
            indicators=self.indicator_settings
        )
        result = run_regime_analysis(request)
        self.assertEqual(result.market_regime, "uptrend")
        self.assertGreater(result.confidence, 0.9) # Expect 3/3 indicators to align

    def test_downtrend_regime(self):
        """Test a clear downtrend scenario."""
        # Generate data with a clear downward trend
        price_history = self._generate_price_data(100, 150, -1)
        last_price = price_history[-1].close
        price_history[-1].low = last_price * 0.95
        price_history[-1].close = last_price * 0.96

        request = MarketRegimeInput(
            symbol="TEST",
            timeframe="1D",
            price_history=price_history,
            indicators=self.indicator_settings
        )
        result = run_regime_analysis(request)
        self.assertEqual(result.market_regime, "downtrend")
        self.assertGreater(result.confidence, 0.9)

    def test_ranging_regime(self):
        """Test a ranging market scenario."""
        prices = []
        start_price = 100
        # Generate 150 days of data to ensure SMA/ADX are stable
        for i in range(150):
            timestamp = (datetime.now() - timedelta(days=150 - i)).isoformat()
            # Oscillate in a very tight band (e.g., +/- 1.0)
            price_offset = (i % 3) - 1
            close_price = start_price + price_offset
            # Ensure high/low are tight to keep ADX low
            high_price = close_price + 0.5
            low_price = close_price - 0.5
            prices.append(PricePoint(
                timestamp=timestamp,
                open=close_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=100000
            ))

        request = MarketRegimeInput(
            symbol="TEST",
            timeframe="1D",
            price_history=prices,
            indicators=self.indicator_settings
        )
        result = run_regime_analysis(request)
        self.assertEqual(result.market_regime, "ranging")
        # Ranging requires at least 2 of 3 conditions
        self.assertGreaterEqual(result.confidence, 2/3)

    def test_volatile_regime(self):
        """Test a volatile market scenario with an ATR spike."""
        price_history = self._generate_price_data(100, 100, 0.1) # Mild trend
        # Engineer a massive price spike in the last bar to spike ATR
        last_price = price_history[-1].close
        price_history[-1].high = last_price * 1.20 # 20% spike
        price_history[-1].low = last_price * 0.80
        price_history[-1].close = last_price * 1.15

        request = MarketRegimeInput(
            symbol="TEST",
            timeframe="1D",
            price_history=price_history,
            indicators=self.indicator_settings
        )
        result = run_regime_analysis(request)
        self.assertEqual(result.market_regime, "volatile_transition")
        self.assertGreater(result.confidence, 0.9) # Expect 1/1

    def test_structure_break_regime(self):
        """Test a volatile regime caused by a structure break."""
        price_history = self._generate_price_data(100, 100, 0.05) # Very mild uptrend

        # Engineer a very obvious swing high 10 bars ago
        # Make the high a clear peak surrounded by lower highs
        price_history[-11].high = 108
        price_history[-10].high = 115 # The peak
        price_history[-9].high = 108

        # Now, decisively break that high in the most recent bar
        price_history[-1].high = 116
        price_history[-1].close = 115.5

        request = MarketRegimeInput(
            symbol="TEST",
            timeframe="1D",
            price_history=price_history,
            indicators=self.indicator_settings
        )
        result = run_regime_analysis(request)
        self.assertEqual(result.market_regime, "volatile_transition")
        self.assertTrue(any("Price has broken a recent swing structure" in s for s in result.explanation))

    def test_classification_logic_fallback(self):
        """Test the internal classification logic directly for the fallback case."""
        from learning_agent.regime_logic import _classify_regime

        # Scenario: Only one indicator is true (score of 1), forcing a fallback
        indicators = {
            'is_ema_trend_up': True, 'is_ema_trend_down': False,
            'is_strong_trend': False, 'is_bullish_momentum': False,
            'is_bearish_momentum': False, 'is_weak_trend': False,
            'is_in_band': False, 'is_ema_slope_flat': False,
            'adx': 22, 'rsi': 55
        }

        result = _classify_regime(indicators, self.indicator_settings)
        self.assertEqual(result.market_regime, "undefined")
        self.assertEqual(result.confidence, 0.3)
