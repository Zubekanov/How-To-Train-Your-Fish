"""The %cpu/%ram/%gpu sampler feeding the dashboard's system panel: sane
percentages, never exceptions (telemetry must not be able to hurt training)."""
import time

from fishrl.train import sysstats


def test_probes_report_sane_values_directly():
    ram = sysstats._ram_pct()
    assert ram is not None and 0.0 <= ram <= 100.0
    t1 = sysstats._cpu_times()
    assert t1 is not None and t1[1] >= t1[0] >= 0
    gpu = sysstats._make_gpu_probe()()                 # None allowed (no GPU/tooling)
    assert gpu is None or 0.0 <= gpu <= 100.0


def test_sampler_thread_populates_latest():
    sysstats.start(interval_s=0.2)
    sysstats.start()                                   # idempotent
    deadline = time.time() + 10
    v = sysstats.latest()
    while time.time() < deadline and v["cpu"] is None:
        time.sleep(0.1)
        v = sysstats.latest()
    assert v["cpu"] is not None and 0.0 <= v["cpu"] <= 100.0
    assert v["ram"] is not None and 0.0 <= v["ram"] <= 100.0
    assert v["gpu"] is None or 0.0 <= v["gpu"] <= 100.0
