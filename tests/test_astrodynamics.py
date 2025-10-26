import math
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gtoc13 import astrodynamics
from gtoc13.orbital_elements import OrbitalElements


def test_solve_kepler_circular_orbit():
    M = 1.234
    e = 0.0

    E = astrodynamics.solve_kepler(M, e)

    assert pytest.approx(float(E), abs=1e-6) == M


def test_solve_kepler_eccentric_orbit():
    M = 0.75
    e = 0.3

    E = astrodynamics.solve_kepler(M, e)

    # Validate Kepler's equation M = E - e*sin(E)
    lhs = float(E - e * jnp.sin(E))
    assert pytest.approx(lhs, abs=1e-9) == M


def test_elements_to_cartesian_circular_equatorial():
    elements = OrbitalElements(
        a=astrodynamics.KMPAU,
        e=0.0,
        i=0.0,
        Omega=0.0,
        omega=0.0,
        M0=0.0,
        mu_body=0.0,
        radius=0.0,
        weight=0.0,
    )

    state = astrodynamics.elements_to_cartesian(elements, t=0.0)

    expected_speed = math.sqrt(astrodynamics.MU_ALTAIRA / astrodynamics.KMPAU)

    np.testing.assert_allclose(
        np.array(state.r), np.array([astrodynamics.KMPAU, 0.0, 0.0]), rtol=1e-8, atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(state.v), np.array([0.0, expected_speed, 0.0]), rtol=1e-8, atol=1e-6
    )


def test_solar_sail_acceleration_direction_and_magnitude():
    r = jnp.array([2.0 * astrodynamics.KMPAU, 0.0, 0.0])
    u_n = jnp.array([1.0, 0.0, 0.0])

    accel = astrodynamics.solar_sail_acceleration(r, u_n)

    expected_magnitude = (
        -2.0
        * astrodynamics.C_FLUX
        * astrodynamics.SAIL_AREA
        / astrodynamics.SPACECRAFT_MASS
        * (astrodynamics.R0 / (2.0 * astrodynamics.KMPAU)) ** 2
    )

    np.testing.assert_allclose(
        np.array(accel), np.array([expected_magnitude, 0.0, 0.0]), rtol=0, atol=5e-12
    )


def test_keplerian_derivatives_matches_two_body_equations():
    r = jnp.array([astrodynamics.KMPAU, 0.0, 0.0])
    v = jnp.array([0.0, 1.0, 0.0])
    y = jnp.concatenate([r, v])

    derivatives = astrodynamics.keplerian_derivatives(0.0, y, None)

    accel_expected = -astrodynamics.MU_ALTAIRA * r / (jnp.linalg.norm(r) ** 3)
    np.testing.assert_allclose(np.array(derivatives[:3]), np.array(v), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(derivatives[3:]), np.array(accel_expected), rtol=0, atol=1e-12)


def test_solar_sail_derivatives_adds_pressure_term():
    r = jnp.array([astrodynamics.KMPAU, 0.0, 0.0])
    v = jnp.array([0.0, 1.0, 0.0])
    y = jnp.concatenate([r, v])
    u_n = jnp.array([1.0, 0.0, 0.0])

    derivatives = astrodynamics.solar_sail_derivatives(0.0, y, (u_n,))

    accel_grav = -astrodynamics.MU_ALTAIRA * r / (jnp.linalg.norm(r) ** 3)
    accel_sail = astrodynamics.solar_sail_acceleration(r, u_n)
    np.testing.assert_allclose(np.array(derivatives[:3]), np.array(v), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(derivatives[3:]), np.array(accel_grav + accel_sail), rtol=0, atol=1e-12)


def test_compute_v_infinity_difference():
    v_sc = jnp.array([10.0, 2.0, -1.0])
    v_body = jnp.array([7.0, -1.0, 0.0])

    v_inf, mag = astrodynamics.compute_v_infinity(v_sc, v_body)

    np.testing.assert_allclose(np.array(v_inf), np.array([3.0, 3.0, -1.0]), rtol=0, atol=1e-12)
    assert pytest.approx(float(mag), abs=1e-6) == math.sqrt(19.0)


def test_patched_conic_flyby_valid_turn():
    v_inf_in = jnp.array([10.0, 0.0, 0.0])
    v_inf_out = jnp.array([10.0 * math.cos(math.radians(20.0)), 10.0 * math.sin(math.radians(20.0)), 0.0])
    mu = 3.986e5
    radius = 6378.0

    altitude, valid = astrodynamics.patched_conic_flyby(v_inf_in, v_inf_out, mu, radius)

    assert bool(valid)
    assert float(altitude) > 0.0


def test_flyby_excess_delta_v_zero_for_identical_vectors():
    v_inf = jnp.array([8.0, -1.0, 0.5])
    mu = 3.986e5
    radius = 6378.0

    delta_v = astrodynamics.flyby_excess_delta_v(
        v_inf, v_inf, mu, radius, min_altitude=0.0, max_altitude=1e9
    )

    assert delta_v < 1e-3


def test_flyby_excess_delta_v_requires_impulse_when_turn_exceeds_limit():
    v_inf_in = jnp.array([10.0, 0.0, 0.0])
    v_inf_out = jnp.array([0.0, 10.0, 0.0])
    mu = 3.986e5
    radius = 6378.0

    delta_v = astrodynamics.flyby_excess_delta_v(v_inf_in, v_inf_out, mu, radius, max_altitude=500.0)

    assert delta_v > 0.0


def test_seasonal_penalty_first_flyby_is_unity():
    current = jnp.array([1.0, 0.0, 0.0])
    previous = jnp.zeros((0, 3))

    penalty = astrodynamics.seasonal_penalty(current, previous)

    assert pytest.approx(float(penalty), abs=1e-12) == 1.0


def test_seasonal_penalty_decreases_with_alignment():
    current = jnp.array([1.0, 0.0, 0.0])
    previous = jnp.array([[math.cos(math.radians(10.0)), math.sin(math.radians(10.0)), 0.0]])

    penalty = astrodynamics.seasonal_penalty(current, previous)

    assert 0.1 <= float(penalty) < 1.0


def test_flyby_velocity_penalty_matches_formula():
    def manual_penalty(v):
        return 0.2 + math.exp(-v / 13.0) / (1.0 + math.exp(-5.0 * (v - 1.5)))

    speeds = [0.0, 5.0, 20.0]
    for speed in speeds:
        expected = manual_penalty(speed)
        actual = astrodynamics.flyby_velocity_penalty(speed)
        assert pytest.approx(float(actual), rel=1e-6) == expected


def test_time_bonus_piecewise_definition():
    early = astrodynamics.time_bonus(5.0)
    late = astrodynamics.time_bonus(20.0)

    assert pytest.approx(float(early), abs=1e-6) == 1.13
    assert pytest.approx(float(late), abs=1e-6) == -0.005 * 20.0 + 1.165


def test_compute_score_returns_zero_for_empty_mission():
    score = astrodynamics.compute_score(
        flybys=[],
        body_weights={},
        grand_tour_achieved=False,
        submission_time_days=10.0,
    )

    assert pytest.approx(float(score), abs=1e-12) == 0.0
