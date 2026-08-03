"""探索性数据分析。"""

__all__ = ["run_eda"]


def __getattr__(name):
    if name == "run_eda":
        from .eda import run_eda

        return run_eda
    raise AttributeError(name)