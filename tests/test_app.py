"""Tests for the web app (Flask). Run:  python -m pytest"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_home_page_opens(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Study Room Scheduler" in response.data


def test_decide_free_books_now(client):
    r = client.post("/api/decide", json={"report": "free"})
    data = r.get_json()
    assert r.status_code == 200
    assert data["action"] == "Book now"
    assert data["best_name"] == "Room 204, 13:00"
    assert data["p_available"] == pytest.approx(0.75)
    assert data["evaluation"]["ours"] == pytest.approx(4.16, abs=0.005)


def test_decide_busy_waits(client):
    data = client.post("/api/decide", json={"report": "busy"}).get_json()
    assert data["action"] == "Wait"
    assert data["p_available"] == pytest.approx(0.125)


def test_user_can_change_scores_and_restarts(client):
    stuck = client.post("/api/decide", json={"report": "free", "scores": [2, 5, 3, 7, 6]}).get_json()
    fixed = client.post("/api/decide", json={"report": "free", "scores": [2, 5, 3, 7, 6], "restarts": True}).get_json()
    assert stuck["best_score"] == 5 and fixed["best_score"] == 7


def test_numbers_as_text_from_the_form_are_accepted(client):
    r = client.post("/api/decide", json={"report": "free", "scores": ["4", "6", "9", "7", "3"], "prior": "0.5"})
    assert r.status_code == 200


@pytest.mark.parametrize("payload", [
    {"report": "free", "prior": 1.5},
    {"report": "maybe"},
    {"report": "free", "scores": []},
    {"report": "free", "scores": [1, "abc", 3]},
    {"report": "free", "scores": ""},
    {"report": "free", "prior": "abc"},
    {"report": "free", "prior": 0, "p_free_if_taken": 0},
])
def test_bad_input_gives_400_and_a_message(client, payload):
    r = client.post("/api/decide", json=payload)
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_not_json_gives_400(client):
    assert client.post("/api/decide", data="hello").status_code == 400
