
from unittest.mock import Mock

from sdirl.rl.simulator import RLSimulator
from pybrain.rl.environments import Environment, EpisodicTask
from sdirl.model import ModelParameter

class TestRLModel():

    def test_can_be_initialized(self):
        model = RLSimulator(n_training_episodes=1,
            n_episodes_per_epoch=1,
            n_simulation_episodes=1,
            parameters = [ModelParameter("param1", (0,1))],
            env=Mock(Environment),
            task=Mock(EpisodicTask))

