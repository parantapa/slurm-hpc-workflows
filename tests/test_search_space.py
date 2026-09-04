"""Tests for search spaces and their ranges.

These need neither botorch nor a queue:
a range is arithmetic on one value,
which is why it lives in a module of its own
and why this file has no `importorskip` at the top of it.
"""

from __future__ import annotations

import math

import pytest

from slurm_workflows.search_space import (
    CategoricalRange,
    FloatRange,
    IntRange,
    space_dim,
    to_params,
    to_unit,
)


class TestIntRange:
    def test_endpoints_map_to_the_unit_interval(self):
        r = IntRange(3, 11)
        assert r.standardize(3) == 0.0
        assert r.standardize(11) == 1.0
        assert r.unstandardize(0.0) == 3
        assert r.unstandardize(1.0) == 11

    def test_every_integer_round_trips(self):
        r = IntRange(-4, 7)
        for x in range(-4, 8):
            assert r.unstandardize(r.standardize(x)) == x

    def test_returns_an_int_not_a_float(self):
        # The value is handed to the objective as a keyword argument;
        # a 3.0 where the objective expects 3 is a bug the caller has to debug.
        value = IntRange(0, 10).unstandardize(0.5)
        assert isinstance(value, int)

    def test_rounds_to_nearest(self):
        r = IntRange(0, 10)
        assert r.unstandardize(0.44) == 4
        assert r.unstandardize(0.46) == 5

    def test_clamps_outside_the_unit_interval(self):
        # optimize_acqf can return a point a hair outside the bounds.
        r = IntRange(3, 11)
        assert r.unstandardize(-0.4) == 3
        assert r.unstandardize(1.7) == 11

    def test_rejects_a_degenerate_range(self):
        with pytest.raises(ValueError):
            IntRange(5, 5)
        with pytest.raises(ValueError):
            IntRange(5, 4)


class TestFloatRange:
    def test_linear_mapping(self):
        r = FloatRange(-5.0, 5.0)
        assert r.standardize(0.0) == 0.5
        assert r.unstandardize(0.5) == 0.0
        assert r.unstandardize(0.0) == -5.0
        assert r.unstandardize(1.0) == 5.0

    def test_linear_round_trip(self):
        r = FloatRange(-5.0, 5.0)
        for y in (0.0, 0.13, 0.5, 0.87, 1.0):
            assert math.isclose(r.standardize(r.unstandardize(y)), y)

    def test_log_range_midpoint_is_the_geometric_mean(self):
        # The point of log_range:
        # half the budget goes to each decade,
        # not to each half of the interval.
        r = FloatRange(1e-4, 1e-1, log_range=True)
        assert math.isclose(r.unstandardize(0.5), math.sqrt(1e-4 * 1e-1))
        assert math.isclose(r.unstandardize(1 / 3), 1e-3)

    def test_log_range_round_trip(self):
        r = FloatRange(1e-5, 1e2, log_range=True)
        for y in (0.0, 0.25, 0.5, 1.0):
            assert math.isclose(r.standardize(r.unstandardize(y)), y, abs_tol=1e-12)

    def test_defaults_to_linear(self):
        assert FloatRange(0.0, 1.0).log_range is False

    def test_clamps_outside_the_unit_interval(self):
        r = FloatRange(-5.0, 5.0)
        assert r.unstandardize(-0.2) == -5.0
        assert r.unstandardize(1.2) == 5.0

    def test_rejects_a_degenerate_range(self):
        with pytest.raises(ValueError):
            FloatRange(1.0, 1.0)
        with pytest.raises(ValueError):
            FloatRange(1.0, 0.0)

    def test_rejects_a_log_range_that_reaches_zero(self):
        # log(0) is not a number the search can work in.
        with pytest.raises(ValueError):
            FloatRange(0.0, 1.0, log_range=True)
        with pytest.raises(ValueError):
            FloatRange(-1.0, 1.0, log_range=True)


class TestCategoricalRange:
    def test_endpoints_map_to_the_unit_interval(self):
        r = CategoricalRange(4)
        assert r.standardize(0) == 0.0
        assert r.standardize(3) == 1.0

    def test_every_category_round_trips(self):
        r = CategoricalRange(7)
        for i in range(7):
            assert r.unstandardize(r.standardize(i)) == i

    def test_categories_partition_the_unit_interval(self):
        r = CategoricalRange(4)
        assert [r.unstandardize(y) for y in (0.0, 0.34, 0.66, 1.0)] == [0, 1, 2, 3]

    def test_a_single_category_is_not_a_division_by_zero(self):
        r = CategoricalRange(1)
        assert r.standardize(0) == 0.0
        assert r.unstandardize(0.7) == 0

    def test_clamps_outside_the_unit_interval(self):
        r = CategoricalRange(3)
        assert r.unstandardize(-0.5) == 0
        assert r.unstandardize(1.5) == 2

    def test_rejects_an_empty_range(self):
        with pytest.raises(ValueError):
            CategoricalRange(0)


SPACE = {
    "lr": FloatRange(1e-4, 1e-1, log_range=True),
    "width": IntRange(8, 64),
    "optimizer": CategoricalRange(3),
}


class TestSpaceDim:
    def test_it_counts_the_parameters(self):
        assert space_dim(SPACE) == 3

    def test_an_empty_space_has_no_dimensions(self):
        assert space_dim({}) == 0


class TestConversions:
    def test_the_unit_cube_corners_are_the_range_endpoints(self):
        low = to_params(SPACE, [0.0, 0.0, 0.0])
        high = to_params(SPACE, [1.0, 1.0, 1.0])

        # The log range goes through exp(log(x)), so it lands next to its
        # endpoint rather than on it; the other two are exact.
        assert math.isclose(low["lr"], 1e-4)
        assert math.isclose(high["lr"], 1e-1)
        assert (low["width"], low["optimizer"]) == (8, 0)
        assert (high["width"], high["optimizer"]) == (64, 2)

    def test_a_point_round_trips_through_the_unit_cube(self):
        params = to_params(SPACE, [0.3, 0.7, 0.5])

        unit = to_unit(SPACE, params)

        assert to_params(SPACE, unit) == params

    def test_coordinates_follow_the_order_of_the_space(self):
        """`to_unit` and `to_params` have to agree on which column is which."""
        assert list(to_params(SPACE, [0.0, 0.0, 0.0])) == list(SPACE)

        reordered = {name: SPACE[name] for name in reversed(list(SPACE))}
        params = {"lr": 1e-2, "width": 16, "optimizer": 1}

        assert to_unit(reordered, params) == list(reversed(to_unit(SPACE, params)))

    def test_each_parameter_uses_its_own_range(self):
        # Halfway along a log range is the geometric mean, not the midpoint.
        params = to_params(SPACE, [0.5, 0.5, 0.5])

        assert math.isclose(params["lr"], math.sqrt(1e-4 * 1e-1))
        assert params["width"] == 36
        assert params["optimizer"] == 1

    def test_a_rounded_parameter_does_not_round_trip_to_its_proposal(self):
        """Why the optimizer records `to_unit` of the point it evaluated.

        A continuous proposal lands between two integers;
        what ran is the rounded one, and that is what the model is told.
        """
        proposal = [0.5, 0.51, 0.5]

        params = to_params(SPACE, proposal)

        assert to_unit(SPACE, params) != proposal
        assert to_unit(SPACE, params) == to_unit(SPACE, to_params(SPACE, proposal))
