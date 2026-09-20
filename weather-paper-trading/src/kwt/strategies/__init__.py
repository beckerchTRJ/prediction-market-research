"""Strategy registry."""
from __future__ import annotations

from .base import Book, MarketCtx, Order, Services, Signal, Strategy
from .calibration_overlay import CalibrationOverlayStrategy
from .climatology import ClimatologyStrategy
from .copy_trading import CopyTradingStrategy
from .ensemble_divergence import EnsembleDivergenceStrategy
from .ensemble_divergence_lh import EnsembleDivergenceLongHorizonStrategy
from .event_arbitrage import EventArbitrageStrategy
from .intraday_nowcast import IntradayNowcastStrategy
from .longshot_fade import LongshotFadeDayBeforeStrategy, LongshotFadeStrategy
from .market_making import MarketMakingStrategy

REGISTRY: dict[str, type[Strategy]] = {
    ClimatologyStrategy.name: ClimatologyStrategy,
    EnsembleDivergenceStrategy.name: EnsembleDivergenceStrategy,
    EnsembleDivergenceLongHorizonStrategy.name: EnsembleDivergenceLongHorizonStrategy,
    LongshotFadeStrategy.name: LongshotFadeStrategy,
    LongshotFadeDayBeforeStrategy.name: LongshotFadeDayBeforeStrategy,
    CalibrationOverlayStrategy.name: CalibrationOverlayStrategy,
    IntradayNowcastStrategy.name: IntradayNowcastStrategy,
    MarketMakingStrategy.name: MarketMakingStrategy,
    EventArbitrageStrategy.name: EventArbitrageStrategy,
    CopyTradingStrategy.name: CopyTradingStrategy,
}

__all__ = ["REGISTRY", "Book", "MarketCtx", "Order", "Services", "Signal", "Strategy"]
