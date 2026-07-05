"""keep_awake: succeeds on Windows, no-ops elsewhere, and is always undoable."""
import os

from fishrl.train.keepawake import allow_sleep, keep_awake


def test_keep_awake_matches_platform():
    ok = keep_awake("test")
    try:
        assert ok is (os.name == "nt")   # Windows: the OS accepted the request
    finally:
        allow_sleep()
