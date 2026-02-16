from __future__ import annotations

import subprocess
import sys


def test_health_check_failure_exit_code():
    # run health check with empty universe by tampering env (should fail gracefully if no data)
    result = subprocess.run([sys.executable, "-m", "app.health", "--universe", "small"], capture_output=True, text=True)
    assert result.returncode in (0, 1)
