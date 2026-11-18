"""Tests for the three algorithms and the whole pipeline. Run:  python -m pytest"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scheduler as sc  # noqa: E402

S = sc.Settings()


# ---------------------------------------------------------------- hill climbing
def test_hill_climbing_default_path():
    assert sc.hill_climbing([4, 6, 9, 7, 3]) == [0, 1, 2]


def test_hill_climbing_stops_at_top_not_beyond():
    path = sc.hill_climbing([4, 6, 9, 7, 3])
    assert [4, 6, 9, 7, 3][path[-1]] == 9


def test_hill_climbing_can_get_stuck_on_small_peak():
    scores = [2, 5, 3, 7, 6]
    assert sc.hill_climbing(scores) == [0, 1]          # stops at 5, the best is 7


def test_restarts_fix_the_stuck_case():
    scores = [2, 5, 3, 7, 6]
    best = sc.find_best(scores, restarts=True)
    assert scores[best[-1]] == 7


def test_hill_climbing_from_the_right_side():
    assert sc.hill_climbing([4, 6, 9, 7, 3], start=4) == [4, 3, 2]


def test_hill_climbing_equal_scores_does_not_move():
    assert sc.hill_climbing([5, 5, 5]) == [0]


def test_hill_climbing_rejects_bad_scores():
    with pytest.raises(ValueError):
        sc.hill_climbing([])
    with pytest.raises(ValueError):
        sc.hill_climbing([1, "a", 3])


# ---------------------------------------------------------------- Bayes
def test_bayes_report_free_gives_75_percent():
    p, p_report = sc.bayes("free", S)
    assert p == pytest.approx(0.75)
    assert p_report == pytest.approx(0.6)


def test_bayes_report_busy_gives_12_5_percent():
    p, p_report = sc.bayes("busy", S)
    assert p == pytest.approx(0.125)
    assert p_report == pytest.approx(0.4)


def test_bayes_useless_report_does_not_change_belief():
    s = sc.Settings(p_free_if_available=0.5, p_free_if_taken=0.5)
    assert sc.bayes("free", s)[0] == pytest.approx(s.prior)


def test_bayes_perfect_report_gives_certainty():
    s = sc.Settings(p_free_if_available=1.0, p_free_if_taken=0.0)
    assert sc.bayes("free", s)[0] == pytest.approx(1.0)
    assert sc.bayes("busy", s)[0] == pytest.approx(0.0)


def test_bayes_impossible_report_gives_clear_error():
    s = sc.Settings(prior=0.0, p_free_if_taken=0.0)
    with pytest.raises(ValueError):
        sc.bayes("free", s)


def test_bayes_result_is_always_a_probability():
    random.seed(1)
    for _ in range(300):
        s = sc.Settings(prior=random.random(), p_free_if_available=random.random(), p_free_if_taken=random.random())
        for report in ("free", "busy"):
            try:
                p, _ = sc.bayes(report, s)
            except ValueError:
                continue
            assert 0 <= p <= 1


def test_bayes_rejects_unknown_report():
    with pytest.raises(ValueError):
        sc.bayes("maybe", S)


# ---------------------------------------------------------------- MDP
def test_mdp_when_probably_free_book_now():
    r = sc.mdp(0.75, 9, S)
    assert r["book"] == pytest.approx(6.25)
    assert r["wait"] == pytest.approx(3.05)
    assert r["action"] == "Book now"


def test_mdp_when_probably_taken_wait():
    r = sc.mdp(0.125, 9, S)
    assert r["book"] == pytest.approx(-0.625)
    assert r["wait"] == pytest.approx(1.025)
    assert r["action"] == "Wait"


def test_mdp_decision_flips_near_34_percent():
    assert sc.mdp(0.33, 9, S)["action"] == "Wait"
    assert sc.mdp(0.35, 9, S)["action"] == "Book now"


def test_mdp_with_gamma_zero_waiting_is_just_the_waiting_cost():
    s = sc.Settings(gamma=0.0)
    assert sc.mdp(0.5, 9, s)["wait"] == pytest.approx(s.reward_wait)


# ---------------------------------------------------------------- the whole chain
def test_pipeline_report_free():
    d = sc.decide("free")
    assert d["best_name"] == "Room 204, 13:00" and d["best_score"] == 9
    assert d["p_available"] == pytest.approx(0.75)
    assert d["action"] == "Book now"


def test_pipeline_report_busy():
    d = sc.decide("busy")
    assert d["p_available"] == pytest.approx(0.125)
    assert d["action"] == "Wait"


def test_algorithms_are_really_connected():
    """A better best room makes booking more attractive (hill climbing -> MDP)."""
    low = sc.decide("free", [1, 2, 3, 2, 1])
    high = sc.decide("free", [1, 2, 10, 2, 1])
    assert high["mdp"]["book"] > low["mdp"]["book"]
    """A more reliable report changes the probability (Bayes -> MDP)."""
    weak = sc.decide("free", settings=sc.Settings(p_free_if_taken=0.6))
    strong = sc.decide("free", settings=sc.Settings(p_free_if_taken=0.1))
    assert strong["p_available"] > weak["p_available"]
    assert strong["mdp"]["book"] > weak["mdp"]["book"]


def test_evaluation_numbers_of_the_report():
    e = sc.evaluate()
    assert e["always_book"] == pytest.approx(3.5)
    assert e["always_wait"] == pytest.approx(2.24)
    assert e["ours"] == pytest.approx(4.16, abs=0.005)


def test_our_method_is_never_worse_than_fixed_strategies():
    """Using the report can never hurt on average (a real property of the maths)."""
    random.seed(7)
    for _ in range(300):
        s = sc.Settings(prior=random.random(), p_free_if_available=random.random(), p_free_if_taken=random.random(),
                        reward_book_taken=-random.randint(0, 5), reward_wait=-random.randint(0, 3),
                        p_still_available=random.random(), p_becomes_available=random.random(), gamma=random.random())
        scores = [random.randint(0, 10) for _ in range(5)]
        e = sc.evaluate(scores, s)
        assert e["ours"] >= max(e["always_book"], e["always_wait"]) - 1e-9


# ---------------------------------------------------------------- input checks
@pytest.mark.parametrize("field,value", [("prior", 1.5), ("prior", -0.1), ("p_free_if_available", 2),
                                         ("gamma", -1), ("p_free_if_taken", "x")])
def test_bad_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        sc.decide("free", settings=sc.Settings(**{field: value}))


def test_unknown_report_is_rejected_by_pipeline():
    with pytest.raises(ValueError):
        sc.decide("maybe")


def test_works_with_other_number_of_choices():
    d = sc.decide("free", [1, 8, 2])
    assert d["best_score"] == 8 and d["names"][0] == "Choice 1"
