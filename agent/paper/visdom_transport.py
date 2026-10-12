"""Visdom JSON transport and localhost frontend verification."""
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from .storage import write_json


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None

def horizontal(value, name, maximum):
    return {
        "type": "scatter",
        "mode": "lines",
        "name": name,
        "x": [0, max(1, maximum)],
        "y": [value, value],
        "line": {"dash": "dash"},
    }

def client(port):
    import visdom

    return visdom.Visdom(
        server="http://127.0.0.1",
        port=port,
        use_incoming_socket=False,
        raise_exceptions=True,
        proxies={},
    )

def send_figure(vis, fig, *, win, env):
    # Visdom 0.2.4's plotlyplot uses this same events payload after importing the
    # optional Plotly encoder. Our figures already contain plain JSON numbers.
    return vis._send(
        {
            "data": fig["data"],
            "layout": fig["layout"],
            "win": win,
            "eid": env,
            "opts": {"title": fig["layout"]["title"]},
        }
    )

def verify_service(port, root):
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    html = opener.open(url, timeout=5).read().decode()
    assets = sorted(set(re.findall(r"(?:src|href)=[\"\']?([^\s\"\'>]+)", html)))
    checked = []
    for asset in assets:
        if "/static/" not in asset or not (".js" in asset or ".css" in asset):
            continue
        if "../user/" in asset:
            continue  # Optional user CSS is an empty dynamic endpoint by default.
        response = opener.open(url + "/" + asset.lstrip("/"), timeout=10)
        body = response.read()
        if (
            response.status != 200
            or len(body) < 20
            or b"<!doctype html>" in body[:100].lower()
        ):
            raise RuntimeError(f"Visdom static asset failed: {asset}")
        checked.append(asset)
    if not any("plotly" in x for x in checked) or not any("main" in x for x in checked):
        raise RuntimeError("Visdom page missing frontend/plotly assets")
    vis = client(port)
    expected = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x": [0, 1],
                "y": [123.0, 123.0],
                "name": "transport self-test",
            }
        ],
        "layout": {"title": "Display transport self-test — not experiment data"},
    }
    send_figure(vis, expected, win="transport-test", env="paper-health")
    received = json.loads(
        vis.get_window_data(win="transport-test", env="paper-health")
    )
    text = json.dumps(received)
    if "123" not in text or "transport self-test" not in text:
        raise RuntimeError("Visdom did not retain sample plot")
    write_json(
        Path(root) / "visdom-acceptance.json",
        {
            "url": url,
            "assets": checked,
            "expected": expected,
            "received": received,
            "passed": True,
            "checked_at": time.time(),
        },
    )
    return url
