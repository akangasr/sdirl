
from unittest.mock import Mock

from sdirl.rl.simulator import RLSimulator, RLParams
from pybrain.rl.environments import Environment, EpisodicTask
from sdirl.model import ModelParameter

class TestRLModel():

    def test_can_be_initialized(self):
        rl_params = RLParams()
        model = RLSimulator(rl_params,
            parameters = [ModelParameter("param1", (0,1))],
            env=Mock(Environment),
            task=Mock(EpisodicTask))

