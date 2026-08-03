"""模型实验模块。"""

__all__ = ["run_linear_svm", "run_iris_svm", "run_svr"]


def __getattr__(name):
    if name == "run_linear_svm":
        from .linear_svm import run_linear_svm

        return run_linear_svm
    if name == "run_iris_svm":
        from .iris_svm import run_iris_svm

        return run_iris_svm
    if name == "run_svr":
        from .svr import run_svr

        return run_svr
    raise AttributeError(name)