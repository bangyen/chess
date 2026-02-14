"""
Pytest configuration and shared fixtures.

Provides shared fixtures and compatibility patches for mutmut mutation testing.
"""

import multiprocessing

import pytest

# ---------------------------------------------------------------------------
# macOS / Python 3.12+ multiprocessing fix for mutmut
# ---------------------------------------------------------------------------
_orig_set_start_method = multiprocessing.set_start_method


def _patched_set_start_method(method, force=False):
    try:
        _orig_set_start_method(method, force=force)
    except RuntimeError:
        # If it's already set, we just ignore it
        pass


multiprocessing.set_start_method = _patched_set_start_method

# ---------------------------------------------------------------------------
# mutmut 3.x trampoline patch – strip 'src.' prefix from module names
# ---------------------------------------------------------------------------
try:
    import mutmut.__main__

    _orig_record_trampoline_hit = mutmut.__main__.record_trampoline_hit

    def _patched_record_trampoline_hit(name):
        name = name.removeprefix("src.")
        return _orig_record_trampoline_hit(name)

    mutmut.__main__.record_trampoline_hit = _patched_record_trampoline_hit
except (ImportError, AttributeError):
    pass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> dict:
    """Provide sample data for tests."""
    return {"name": "Test User", "age": 30}


@pytest.fixture
def sample_config() -> dict:
    """Provide sample configuration for tests."""
    return {"name": "test-app", "version": "1.0.0"}
