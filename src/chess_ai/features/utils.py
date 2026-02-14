"""Utility functions for feature module loading."""

import importlib.util
import types


def load_feature_module(path: str) -> types.ModuleType:
    """Load a feature module from a file path.

    Args:
        path: Path to the Python module file containing extract_features function

    Returns:
        The loaded module

    Raises:
        RuntimeError: If the module cannot be loaded or doesn't define extract_features
    """
    spec = importlib.util.spec_from_file_location("user_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load features module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "extract_features"):
        raise RuntimeError(
            "Feature module must define extract_features(board) -> Dict[str, number|bool]"
        )
    return mod
