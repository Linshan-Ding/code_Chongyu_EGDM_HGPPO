"""Sample total resident memory of the learner and its worker process tree."""
import threading
import psutil


class PeakMemory:
    def __init__(self):
        self.peak_bytes = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        parent = psutil.Process()
        total = 0
        for process in [parent] + parent.children(recursive=True):
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass
        self.peak_bytes = max(self.peak_bytes, total)

    def run(self):
        while not self.stop.wait(.25):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()
