import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run"
    )

import random
import numpy as np
from sdirl.environment import Environment

def reset_env():
    Environment.__instance = None

class TestEnvironment():

    def test_initialization_with_default_arguments_works(self):
        reset_env()
        env = Environment()
        assert env.random_state is not None
        assert env.client is None

    def test_initialization_with_random_seed_works(self):
        reset_env()
        env = Environment(["script_name", "123"])
        assert env.random_state is not None
        rnd_val = env.random_state.randint(1e7)
        env2 = Environment(["script_name", "123"])
        rnd_val2 = env.random_state.randint(1e7)
        assert rnd_val == rnd_val2

    @slow  # ~5s
    def test_initialization_with_client_works(self):
        reset_env()
        with pytest.raises(OSError):
            env = Environment(["script_name", "123", "12345"])  # fails while trying to connect

