from .base import Signal, Strategy, StrategyContext
from .ma_crossover import MACrossoverStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy
from .breakout import BreakoutStrategy
from .ensemble import EnsembleStrategy


def build_strategy(name: str, params: dict, all_params: dict | None = None) -> Strategy:
    """Factory: build a strategy by name.

    `all_params` is the full `strategies` config dict — needed by ensemble
    to instantiate its child strategies.
    """
    name = name.lower()
    if name == "ma_crossover":
        return MACrossoverStrategy(**params)
    if name == "rsi_mean_reversion":
        return RSIMeanReversionStrategy(**params)
    if name == "breakout":
        return BreakoutStrategy(**params)
    if name == "ensemble":
        if all_params is None:
            raise ValueError("ensemble requires the full strategies config")
        children = [
            MACrossoverStrategy(**all_params["ma_crossover"]),
            RSIMeanReversionStrategy(**all_params["rsi_mean_reversion"]),
            BreakoutStrategy(**all_params["breakout"]),
        ]
        return EnsembleStrategy(children=children, **params)
    raise ValueError(f"Unknown strategy: {name}")


__all__ = [
    "Signal",
    "Strategy",
    "StrategyContext",
    "MACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "BreakoutStrategy",
    "EnsembleStrategy",
    "build_strategy",
]
