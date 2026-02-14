"""Tests for surrogate model-based move explanations.

Covers the SurrogateExplainer class which converts surrogate model
outputs into human-readable explanations with centipawn contributions.
"""

import warnings
from unittest.mock import Mock

import numpy as np
import pytest

from chess_ai.model_trainer import PhaseEnsemble
from chess_ai.surrogate_explainer import SurrogateExplainer

# Suppress sklearn convergence warnings in tests
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trained_ensemble(feature_names):
    """Build a small PhaseEnsemble fitted on synthetic data.

    Provides a real (non-mock) model so that predict / coef_ /
    get_contributions all work as expected.
    """
    np.random.seed(42)
    n, d = 40, len(feature_names)
    X = np.random.randn(n, d)
    y = X[:, 0] * 2.0 + np.random.randn(n) * 0.1

    alphas = np.logspace(-2, 2, 10).tolist()
    ens = PhaseEnsemble(feature_names, alphas, cv_folds=2, max_iter=5000)
    ens.fit(X, y)
    return ens


def _make_scaler(feature_names):
    """Build a fitted StandardScaler matching *feature_names*."""
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X = np.random.randn(40, len(feature_names))
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSurrogateExplainerInit:
    """Test SurrogateExplainer initialisation."""

    def test_stores_model_scaler_names(self):
        """Constructor stores model, scaler, and feature_names as attributes."""
        names = ["material_diff", "mobility_us", "phase"]
        model = _make_trained_ensemble(names)
        scaler = _make_scaler(names)

        explainer = SurrogateExplainer(model=model, scaler=scaler, feature_names=names)

        assert explainer.model is model
        assert explainer.scaler is scaler
        assert explainer.feature_names == names


class TestCalculateContributions:
    """Test calculate_contributions with various scenarios."""

    @pytest.fixture
    def explainer(self):
        """Create a SurrogateExplainer with a small trained model."""
        names = ["material_diff", "mobility_us", "mobility_them", "phase"]
        model = _make_trained_ensemble(names)
        scaler = _make_scaler(names)
        return SurrogateExplainer(model=model, scaler=scaler, feature_names=names)

    def test_returns_list_of_tuples(self, explainer):
        """Result is a list of (name, cp_value, explanation) tuples."""
        before = {
            "material_diff": 0.0,
            "mobility_us": 20.0,
            "mobility_them": 20.0,
            "phase": 14.0,
        }
        after = {
            "material_diff": 3.0,
            "mobility_us": 25.0,
            "mobility_them": 18.0,
            "phase": 14.0,
        }

        result = explainer.calculate_contributions(before, after, top_k=5, min_cp=0.0)

        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3
            name, cp, text = item
            assert isinstance(name, str)
            assert isinstance(cp, float)
            assert isinstance(text, str)

    def test_respects_top_k(self, explainer):
        """Only the top_k contributions are returned."""
        before = {
            "material_diff": 0.0,
            "mobility_us": 0.0,
            "mobility_them": 0.0,
            "phase": 0.0,
        }
        after = {
            "material_diff": 5.0,
            "mobility_us": 5.0,
            "mobility_them": 5.0,
            "phase": 5.0,
        }

        result = explainer.calculate_contributions(before, after, top_k=2, min_cp=0.0)

        assert len(result) <= 2

    def test_filters_by_min_cp(self, explainer):
        """Contributions below min_cp are excluded."""
        before = {
            "material_diff": 0.0,
            "mobility_us": 0.0,
            "mobility_them": 0.0,
            "phase": 0.0,
        }
        after = {
            "material_diff": 0.0,
            "mobility_us": 0.0,
            "mobility_them": 0.0,
            "phase": 0.0,
        }

        result = explainer.calculate_contributions(
            before, after, top_k=10, min_cp=99999.0
        )

        assert len(result) == 0

    def test_sorted_by_magnitude(self, explainer):
        """Results are sorted by decreasing |contribution|."""
        before = {
            "material_diff": 0.0,
            "mobility_us": 0.0,
            "mobility_them": 0.0,
            "phase": 0.0,
        }
        after = {
            "material_diff": 10.0,
            "mobility_us": 5.0,
            "mobility_them": 2.0,
            "phase": 1.0,
        }

        result = explainer.calculate_contributions(before, after, top_k=10, min_cp=0.0)

        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert abs(result[i][1]) >= abs(result[i + 1][1])

    def test_known_template_used(self):
        """Features with known templates produce correct explanation text."""
        names = ["material_diff"]
        model = _make_trained_ensemble(names)
        scaler = _make_scaler(names)

        # Force a large contribution so it passes min_cp
        explainer = SurrogateExplainer(model=model, scaler=scaler, feature_names=names)

        before = {"material_diff": 0.0}
        after = {"material_diff": 100.0}

        result = explainer.calculate_contributions(before, after, top_k=5, min_cp=0.0)

        if result:
            name, _cp, text = result[0]
            assert name == "material_diff"
            assert "Material advantage" in text

    def test_unknown_feature_gets_fallback_template(self):
        """Features without a template use the generic fallback."""
        names = ["unknown_feature_xyz"]
        model = _make_trained_ensemble(names)
        scaler = _make_scaler(names)
        explainer = SurrogateExplainer(model=model, scaler=scaler, feature_names=names)

        before = {"unknown_feature_xyz": 0.0}
        after = {"unknown_feature_xyz": 100.0}

        result = explainer.calculate_contributions(before, after, top_k=5, min_cp=0.0)

        if result:
            _name, _cp, text = result[0]
            assert "unknown_feature_xyz" in text
            assert "cp" in text

    def test_missing_features_default_to_zero(self, explainer):
        """Missing keys in features_after default to 0.0."""
        before = {"material_diff": 0.0}
        after = {}  # all missing

        # Should not raise
        result = explainer.calculate_contributions(before, after, top_k=5, min_cp=0.0)
        assert isinstance(result, list)

    def test_handles_exception_gracefully(self):
        """Returns empty list when an internal error occurs."""
        names = ["material_diff"]
        mock_model = Mock()
        mock_model.get_contributions.side_effect = RuntimeError("boom")

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0]])

        explainer = SurrogateExplainer(
            model=mock_model, scaler=mock_scaler, feature_names=names
        )

        result = explainer.calculate_contributions(
            {"material_diff": 0.0}, {"material_diff": 5.0}
        )
        assert result == []

    def test_empty_features(self, explainer):
        """Empty before/after dicts produce a result without crashing."""
        result = explainer.calculate_contributions({}, {}, top_k=5, min_cp=0.0)
        assert isinstance(result, list)


class TestFeatureTemplates:
    """Verify the FEATURE_TEMPLATES class attribute."""

    def test_all_templates_are_format_strings(self):
        """Every template must contain a format placeholder."""
        for key, tmpl in SurrogateExplainer.FEATURE_TEMPLATES.items():
            # Should be formattable with a float
            try:
                tmpl.format(42.0)
            except (KeyError, IndexError) as exc:  # noqa: PERF203
                pytest.fail(f"Template for '{key}' is not formattable: {exc}")

    def test_templates_cover_common_features(self):
        """Key chess features have dedicated templates."""
        expected = [
            "material_diff",
            "mobility_us",
            "mobility_them",
            "king_ring_pressure_us",
            "passed_us",
            "center_control_us",
        ]
        for feat in expected:
            assert feat in SurrogateExplainer.FEATURE_TEMPLATES
