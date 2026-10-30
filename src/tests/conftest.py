"""Shared pytest fixtures for this test suite.

Imports tests/fixtures/synthetic_omnet_obs.py, which (as a side effect of
import) sets OMNET_APPTAINER_SIF/OMNET_SMOKE_DIR and inserts omnet_module/
and src/ onto sys.path -- the two things omnet_env.py needs to be importable
OFFLINE (no OMNeT++/INET install here; see that module's docstring). conftest
.py is collected by pytest before any test_*.py in this directory, so any
test module can do a bare `import omnet_env` without repeating that setup.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))

import pytest

from synthetic_omnet_obs import make_layout, build_obs  # noqa: E402


@pytest.fixture
def synthetic_omnet_obs():
    """Factory fixture: synthetic_omnet_obs(seed, nd=5, step=0) -> obs_json.

    Builds one schema-correct obs_json for omnet_env.build_drone_tokens /
    build_raw_user_tokens, using the same Friis/log-distance radio model
    max_flow_reward.py scores rewards with. See
    tests/fixtures/synthetic_omnet_obs.py for the generator itself.
    """
    def _make(seed: int, nd: int = 5, step: int = 0):
        users, pairs, drones = make_layout(seed, nd=nd)
        return build_obs(drones, users, pairs, step=step)
    return _make
