import json

import pytest
import requests

from here_error import params, routing
from here_error.routing import RoutingError
from here_fixtures import fix, to_latlon

KEY = "SECRET-KEY-123"


def enc(points):
    from test_polyline import encode

    return encode(points)


def make_request(rid="p1", east=(0, 1000), n=11, observed=100.0, y=0.0):
    fixes = [fix(1788901800 + i * 10, east[0] + i * (east[1] - east[0]) / (n - 1), y) for i in range(n)]
    return routing.request_for_fixes(rid, "pass", "run_t", fixes, observed)


def body_for(points_xy, duration=90, base=80, length=1000):
    pts = [to_latlon(x, y) for x, y in points_xy]
    return {"routes": [{"sections": [{"summary": {"duration": duration, "baseDuration": base, "length": length},
                                      "polyline": enc(pts)}]}]}


STRAIGHT = [(0, 0), (500, 0), (1000, 0)]


def test_request_parameters():
    q = routing.request_params(make_request())
    assert q["transportMode"] == "car" and q["return"] == "summary,polyline"
    assert q["departureTime"] == "2026-09-08T17:10:00-04:00"
    assert "apiKey" not in q
    assert q["origin"].count(",") == 1


def test_course_is_appended_from_the_first_and_last_fix_headings():
    fixes = [fix(1788901800, 0, 0, heading=44.6), fix(1788901810, 500, 0, heading=90.0), fix(1788901820, 1000, 0, heading=359.6)]
    q = routing.request_params(routing.request_for_fixes("p", "pass", "r", fixes, 20.0))
    assert q["origin"].endswith(";course=45") and q["destination"].endswith(";course=0")
    assert q["origin"].count(";") == 1
    fixes[0] = fix(1788901800, 0, 0, heading=None)
    q = routing.request_params(routing.request_for_fixes("p", "pass", "r", fixes, 20.0))
    assert ";course" not in q["origin"] and ";course=0" in q["destination"]


def test_course_changes_the_cache_key(tmp_path):
    a = make_request()
    b = routing.request_for_fixes("p1", "pass", "run_t", [fix(1788901800 + i * 10, i * 100, 0, heading=h) for i, h in
                                                          zip(range(11), [200.0] + [90.0] * 10)], 100.0)
    assert routing.cache_file(tmp_path, routing.request_params(a)) != routing.cache_file(tmp_path, routing.request_params(b))


def test_call_is_made_once_and_cached(tmp_path):
    calls = []

    def fetch(url, query):
        calls.append((url, dict(query)))
        return body_for(STRAIGHT)

    req = make_request()
    r1 = routing.run_routing([req], tmp_path, fetch, env={"HERE_API_KEY": KEY})
    assert r1.n_calls == 1 and r1.n_cached == 0
    assert calls[0][1]["apiKey"] == KEY and calls[0][0] == routing.ROUTER_URL
    r2 = routing.run_routing([req], tmp_path, fetch, env={})   # no key needed for a cache hit
    assert len(calls) == 1 and r2.n_calls == 0 and r2.n_cached == 1
    assert r2.results[0].duration_s == 90 and r2.results[0].no_traffic_duration_s == 80


def test_key_never_in_cache_files(tmp_path):
    routing.run_routing([make_request()], tmp_path, lambda u, q: body_for(STRAIGHT), env={"HERE_API_KEY": KEY})
    files = list(tmp_path.glob("*.json"))
    assert files
    for f in files:
        assert KEY not in f.read_text() and "apiKey" not in f.read_text()


def test_missing_key_refused_before_any_call(tmp_path):
    def fetch(u, q):
        raise AssertionError("no call expected")

    with pytest.raises(RoutingError, match="HERE_API_KEY"):
        routing.run_routing([make_request()], tmp_path, fetch, env={})


def test_budget_refused_before_any_call(tmp_path):
    reqs = [make_request(rid=f"p{i}", east=(0, 1000 + i)) for i in range(params.HERE_CALL_BUDGET + 1)]

    def fetch(u, q):
        raise AssertionError("no call expected")

    with pytest.raises(RoutingError, match="budget"):
        routing.run_routing(reqs, tmp_path, fetch, env={"HERE_API_KEY": KEY})
    assert list(tmp_path.glob("*.json")) == []


def test_dry_run_plan_makes_no_call_and_reports_cache_state(tmp_path):
    req = make_request()
    plan = routing.plan_calls([req], tmp_path)
    assert [c.cached for c in plan] == [False]
    routing.run_routing([req], tmp_path, lambda u, q: body_for(STRAIGHT), env={"HERE_API_KEY": KEY})
    assert [c.cached for c in routing.plan_calls([req], tmp_path)] == [True]


def test_http_errors_have_the_key_scrubbed(monkeypatch, tmp_path):
    class Resp:
        status_code = 401
        text = f"bad key {KEY}"

    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Resp())
    with pytest.raises(RoutingError) as ei:
        routing.default_fetch(routing.ROUTER_URL, {"apiKey": KEY})
    assert KEY not in str(ei.value) and "401" in str(ei.value)

    def boom(url, params, timeout):
        raise requests.ConnectionError(f"failed for {url}?apiKey={KEY}")

    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(RoutingError) as ei:
        routing.default_fetch(routing.ROUTER_URL, {"apiKey": KEY})
    import traceback

    text = "".join(traceback.format_exception(ei.value))
    assert KEY not in text and KEY not in repr(ei.value)


def test_route_that_follows_the_path_passes():
    req = make_request()
    res = routing.evaluate(req, body_for(STRAIGHT, duration=90))
    assert res.follows_path and res.exclusion is None
    assert res.signed_error == pytest.approx(-0.1)            # route time shorter than the car's
    assert res.share_within == 1.0


def test_route_on_a_parallel_road_fails_the_check():
    req = make_request()
    res = routing.evaluate(req, body_for([(0, 60), (500, 60), (1000, 60)]))
    assert not res.follows_path and res.exclusion == "route_does_not_follow_path"
    assert res.share_within == 0.0


def test_route_much_longer_than_the_path_fails_the_check():
    req = make_request()
    detour = [(0, 0), (500, 0), (1000, 0), (1400, 0), (1000, 0)]
    res = routing.evaluate(req, body_for(detour))
    assert not res.follows_path and res.length_rel_diff > params.ROUTE_LENGTH_TOL


def test_no_route_raises():
    with pytest.raises(RoutingError, match="no route"):
        routing.evaluate(make_request(), {"routes": []})


def test_key_straddling_character_200_of_an_error_body_is_scrubbed(monkeypatch):
    class Resp:
        status_code = 500
        text = "x" * 190 + KEY + "y" * 50          # the key spans characters 190 to 204

    monkeypatch.setattr(requests, "get", lambda url, params, timeout: Resp())
    with pytest.raises(RoutingError) as ei:
        routing.default_fetch(routing.ROUTER_URL, {"apiKey": KEY})
    msg = str(ei.value)
    assert KEY not in msg and "SECRET" not in msg and "KEY-123" not in msg
