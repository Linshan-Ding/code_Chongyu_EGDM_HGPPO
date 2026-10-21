"""Persistent worker transport and exact potential accounting."""
import pickle
import time
from environment.reward import tardiness_potential


def phi(env):
    return tardiness_potential(
        env.sim.instance,
        env.sim.orders,
        env.sim.operations,
        env.sim.time,
        float(env.cfg.env.reward.potential_eta),
        str(getattr(env.cfg.env.reward, "potential_mode", "legacy")),
    )

class WorkerFailure(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(str(evidence))

class WorkerPool:
    def receive(self, expected="ok", deadline=None):
        remaining = set(range(len(self.workers)))
        results = [None] * len(self.workers)
        while remaining:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError("actor collection deadline reached")
            for i in list(remaining):
                process, pipe = self.workers[i]
                if pipe.poll(0.02):
                    try:
                        status, output = pickle.loads(pipe.recv_bytes())
                    except EOFError as exc:
                        raise WorkerFailure(
                            {"worker": i, "exitcode": process.exitcode, "error": "EOF"}
                        ) from exc
                    if status != expected:
                        self.last_evidence.append(output)
                        raise WorkerFailure(output)
                    results[i] = output
                    remaining.remove(i)
                elif not process.is_alive():
                    raise WorkerFailure({"worker": i, "exitcode": process.exitcode})
        return results

    def close(self):
        for p, pipe in self.workers:
            try:
                if p.is_alive():
                    pipe.send_bytes(pickle.dumps(("close", None)))
            except (BrokenPipeError, OSError):
                pass
        for p, pipe in self.workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
            pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
