"""Best-effort visdom monitoring for long Scheme-2 runs.

CSV files remain the authoritative experiment record.  This adapter only adds
live visibility and deliberately never lets a missing visdom service interrupt
an expensive PPO run.
"""

from __future__ import annotations

import json
import socket
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np


class VisdomLogger:
    """Append scalar training diagnostics to one isolated visdom environment."""

    def __init__(
        self,
        *,
        enabled: bool,
        server: str,
        port: int,
        env_prefix: str,
        run_name: str,
        update_every_iterations: int,
        config_summary: dict[str, Any],
    ) -> None:
        self.enabled = bool(enabled)
        self.update_every_iterations = max(1, int(update_every_iterations))
        self._client = None
        self._windows: set[str] = set()
        self._warned = False
        self.env = f"{env_prefix}_{run_name}"
        parsed = urlparse(str(server) if "://" in str(server) else f"http://{server}")
        self._host = parsed.hostname or "localhost"
        self._port = int(port)
        if not self.enabled:
            return
        if not self._reachable():
            self._disable(f"visdom server {self._host}:{self._port} is unavailable")
            return
        try:
            from visdom import Visdom

            self._client = Visdom(
                server=str(server),
                port=int(port),
                env=self.env,
                raise_exceptions=False,
                use_incoming_socket=False,
            )
            self._client.text(
                f"<pre>{json.dumps(config_summary, ensure_ascii=False, indent=2, default=str)}</pre>",
                win="run_config",
                opts={"title": "Scheme-2 configuration"},
            )
        except Exception as exc:  # Monitoring must never abort scientific runs.
            self._disable(f"visdom disabled ({exc})")

    def _disable(self, message: str) -> None:
        self._client = None
        if not self._warned:
            print(f"[monitoring] {message}; continuing with CSV logs only", flush=True)
            self._warned = True

    def _reachable(self) -> bool:
        """Avoid visdom's verbose client-side failures when its server is absent."""
        try:
            with socket.create_connection((self._host, self._port), timeout=0.15):
                return True
        except OSError:
            return False

    def scalar(self, name: str, *, iteration: int, value: float, title: str) -> None:
        if self._client is None or iteration % self.update_every_iterations != 0:
            return
        if not self._reachable():
            self._disable(f"visdom server {self._host}:{self._port} is unavailable")
            return
        try:
            update = "append" if name in self._windows else None
            self._client.line(
                X=np.asarray([float(iteration)]),
                Y=np.asarray([float(value)]),
                win=name,
                update=update,
                opts={"title": title, "xlabel": "PPO iteration", "ylabel": title},
            )
            self._windows.add(name)
        except Exception as exc:
            self._disable(f"visdom update failed ({exc})")

    def training(self, *, iteration: int, metrics: dict[str, Any]) -> None:
        for key, title in (
            ("reward_mean", "Mean reward"),
            ("policy_loss", "Policy loss"),
            ("value_loss", "Value loss"),
            ("approx_kl", "Approximate KL"),
            ("sps", "Events per second"),
        ):
            value = metrics.get(key)
            if value is not None:
                self.scalar(key, iteration=iteration, value=float(value), title=title)

    def validation(self, *, iteration: int, mean_twt: float) -> None:
        self.scalar(
            "validation_mean_twt",
            iteration=iteration,
            value=float(mean_twt),
            title="Fixed-validation mean TWT",
        )


def settings_summary(settings: Any) -> dict[str, Any]:
    """Convert settings into a small visdom text panel without tensor payloads."""
    if is_dataclass(settings):
        values = asdict(settings)
    else:
        values = dict(settings)
    return {
        key: value
        for key, value in values.items()
        if key not in {"resume_checkpoint"}
    }


__all__ = ["VisdomLogger", "settings_summary"]
