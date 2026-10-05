"""Stage 6: HERE Routing v8 travel time for each pass and stretch, with a cache and a call budget.

The API key comes only from the environment. It is never written to a log line, an exception
message, a cache file or an output file.
"""
from __future__ import annotations

import datetime as dt
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from here_error import params
from here_error.geometry import LocalFrame, haversine_m, polyline_length, project_points
from here_error.models import RouteRequest, RouteResult
from here_error.polyline import decode

ROUTER_URL = "https://router.hereapi.com/v8/routes"
KEY_ENV = "HERE_API_KEY"


class RoutingError(RuntimeError):
    pass


#: Takes the URL and the query parameters including the key; returns the parsed JSON body.
Fetch = Callable[[str, dict], dict]


def scrub(text: str, key: str | None) -> str:
    return text.replace(key, "***") if key else text


def request_params(req: RouteRequest) -> dict:
    """Query parameters without the key: these are what the cache is keyed on."""
    departure = dt.datetime.fromtimestamp(req.departure_utc_s, ZoneInfo(params.LOCAL_TZ))
    return {
        "transportMode": "car",
        "origin": f"{req.origin_lat:.6f},{req.origin_lon:.6f}",
        "destination": f"{req.dest_lat:.6f},{req.dest_lon:.6f}",
        "departureTime": departure.isoformat(timespec="seconds"),
        "return": "summary,polyline",
    }


def cache_file(cache_dir: Path, query: dict) -> Path:
    digest = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
    return cache_dir / f"{digest}.json"


def default_fetch(url: str, query: dict) -> dict:
    import requests

    key = query.get("apiKey")
    failure: str | None = None
    body = None
    try:
        resp = requests.get(url, params=query, timeout=30)
        if resp.status_code >= 400:
            failure = f"HERE routing returned HTTP {resp.status_code}: {scrub(resp.text[:200], key)}"
        else:
            body = resp.json()
    except requests.RequestException as exc:
        failure = f"HERE routing request failed: {scrub(str(exc), key)}"
    except ValueError:
        failure = "HERE routing returned a body that is not JSON"
    if failure is not None:
        # Raised outside the handler so no exception chain holds the original text with the key.
        raise RoutingError(failure)
    return body


@dataclass(frozen=True)
class PlannedCall:
    request: RouteRequest
    query: dict
    path: Path
    cached: bool


def plan_calls(requests: Sequence[RouteRequest], cache_dir: Path) -> list[PlannedCall]:
    out = []
    for r in requests:
        q = request_params(r)
        path = cache_file(cache_dir, q)
        out.append(PlannedCall(r, q, path, path.exists()))
    return out


def _route_totals(body: dict) -> tuple[float, float | None, float, list[tuple[float, float]]]:
    routes = body.get("routes") or []
    if not routes:
        raise RoutingError("HERE returned no route")
    dur = base = length = 0.0
    have_base = True
    pts: list[tuple[float, float]] = []
    for sec in routes[0]["sections"]:
        s = sec["summary"]
        dur += s["duration"]
        length += s["length"]
        if "baseDuration" in s:
            base += s["baseDuration"]
        else:
            have_base = False
        pts.extend(decode(sec["polyline"]))
    return dur, (base if have_base else None), length, pts


def follows_path(route_pts, req: RouteRequest) -> tuple[bool, float, float]:
    """Share of the driven fixes within ROUTE_MATCH_M of the route, and the route-length difference.

    The route follows the path when the share is at least ROUTE_MATCH_SHARE and the route
    length is within ROUTE_LENGTH_TOL of the driven length.
    """
    lat = np.array([p[0] for p in route_pts])
    lon = np.array([p[1] for p in route_pts])
    frame = LocalFrame(float(lat.mean()), float(lon[0]))
    xy = frame.to_xy(lat, lon)
    fixes = frame.to_xy(np.array(req.fix_lat), np.array(req.fix_lon))
    within = sum(pr.distance_m <= params.ROUTE_MATCH_M for pr in project_points(xy, fixes))
    share = within / len(fixes)
    rel = abs(polyline_length(xy) - req.matched_m) / req.matched_m
    return (share >= params.ROUTE_MATCH_SHARE and rel <= params.ROUTE_LENGTH_TOL), share, rel


def evaluate(req: RouteRequest, body: dict) -> RouteResult:
    dur, base, length, pts = _route_totals(body)
    ok, share, rel = follows_path(pts, req)
    return RouteResult(
        request_id=req.request_id, kind=req.kind, duration_s=dur, no_traffic_duration_s=base,
        length_m=length, observed_s=req.observed_s, signed_error=(dur - req.observed_s) / req.observed_s,
        follows_path=ok, share_within=share, length_rel_diff=rel,
        exclusion=None if ok else "route_does_not_follow_path",
    )


@dataclass(frozen=True)
class RoutingRun:
    results: tuple[RouteResult, ...]
    n_calls: int
    n_cached: int


def run_routing(
    requests: Sequence[RouteRequest],
    cache_dir: Path,
    fetch: Fetch = default_fetch,
    env: dict | None = None,
) -> RoutingRun:
    """Answer every request from the cache or HERE. Makes no call when everything is cached.

    Refuses before the first call when the key is absent or the uncached calls exceed the budget.
    """
    env = os.environ if env is None else env
    plan = plan_calls(requests, cache_dir)
    todo = [c for c in plan if not c.cached]
    key = env.get(KEY_ENV)
    if todo and not key:
        raise RoutingError(f"{KEY_ENV} is not set; refusing to call HERE")
    if len(todo) > params.HERE_CALL_BUDGET:
        raise RoutingError(f"{len(todo)} uncached calls exceed the budget of {params.HERE_CALL_BUDGET}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for c in todo:
        body = fetch(ROUTER_URL, {**c.query, "apiKey": key})
        c.path.write_text(json.dumps({"params": c.query, "response": body}, sort_keys=True))
    results = []
    for c in plan:
        body = json.loads(c.path.read_text())["response"]
        results.append(evaluate(c.request, body))
    return RoutingRun(tuple(results), len(todo), len(plan) - len(todo))


def request_for_fixes(request_id: str, kind: str, run: str, fixes, observed_s: float) -> RouteRequest:
    """A request from the first fix to the last, departing at the first fix's time.

    The driven length is the path through the given fixes, including any jump between them.
    """
    lat = np.array([f.lat for f in fixes])
    lon = np.array([f.lon for f in fixes])
    driven = float(np.sum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])))
    return RouteRequest(
        request_id=request_id, kind=kind, run=run,
        origin_lat=fixes[0].lat, origin_lon=fixes[0].lon, dest_lat=fixes[-1].lat, dest_lon=fixes[-1].lon,
        departure_utc_s=fixes[0].utc_s, observed_s=observed_s, matched_m=driven,
        fix_lat=tuple(lat.tolist()), fix_lon=tuple(lon.tolist()),
    )


CSV_COLUMNS = ("request_id", "kind", "observed_s", "duration_s", "no_traffic_duration_s", "length_m",
               "signed_error", "follows_path", "share_within", "length_rel_diff", "exclusion")


def write_csv(results: Sequence[RouteResult], path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in results:
            w.writerow([r.request_id, r.kind, r.observed_s, r.duration_s,
                        "" if r.no_traffic_duration_s is None else r.no_traffic_duration_s, r.length_m,
                        r.signed_error, int(r.follows_path), r.share_within, r.length_rel_diff, r.exclusion or ""])


def read_csv(path: Path) -> list[RouteResult]:
    with path.open(newline="") as fh:
        return [RouteResult(
            request_id=r["request_id"], kind=r["kind"], duration_s=float(r["duration_s"]),
            no_traffic_duration_s=float(r["no_traffic_duration_s"]) if r["no_traffic_duration_s"] else None,
            length_m=float(r["length_m"]), observed_s=float(r["observed_s"]), signed_error=float(r["signed_error"]),
            follows_path=bool(int(r["follows_path"])), share_within=float(r["share_within"]),
            length_rel_diff=float(r["length_rel_diff"]), exclusion=r["exclusion"] or None,
        ) for r in csv.DictReader(fh)]
