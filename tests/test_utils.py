"""Tests for the shared helpers.

Plain functions with no dependencies of their own,
which is why they live outside the modules that use them.
"""

from __future__ import annotations

import pytest

from slurm_workflows.utils import floor_power_of_two, format_mapping, format_param


class TestFloorPowerOfTwo:
    @pytest.mark.parametrize(
        "n,expected",
        [(1, 1), (2, 2), (3, 2), (4, 4), (7, 4), (8, 8), (9, 8), (63, 32), (64, 64)],
    )
    def test_truncates_down(self, n, expected):
        assert floor_power_of_two(n) == expected

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            floor_power_of_two(0)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


class TestFormatParam:
    def test_a_float_gets_a_fixed_precision(self):
        assert format_param(1 / 3) == "0.333333"

    def test_everything_else_prints_as_itself(self):
        assert format_param(7) == "7"
        assert format_param("adam") == "adam"
        assert format_param(None) == "None"

    def test_a_bool_is_not_treated_as_a_float(self):
        assert format_param(True) == "True"


class TestFormatMapping:
    def test_it_renders_every_pair(self):
        assert format_mapping({"lr": 0.5, "n": 4}) == "lr=0.5, n=4"

    def test_an_empty_mapping_renders_as_nothing(self):
        assert format_mapping({}) == ""
