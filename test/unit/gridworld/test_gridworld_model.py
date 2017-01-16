import pytest
import numpy as np

from sdirl.gridworldmodel.model import GridWorldModel
from sdirl.gridworldmodel.mdp import InitialStateUniformlyAtEdge
from sdirl.gridworldmodel.mdp import GridWorldEnvironment
from sdirl.gridworldmodel.mdp import State

def trivial_model(training_episodes=100):
    return GridWorldModel(["step_penalty", "feature1_value"],
        grid_size=3,
        prob_rnd_move=0.0,
        world_seed=0,
        n_training_episodes=training_episodes,
        n_episodes_per_epoch=10,
        n_simulation_episodes=100)


class TestGridWorldEnvironment():

    def test_state_restriction_works_as_expected(self):
        env = GridWorldEnvironment(grid_size=3, target_state=State(0,0))
        for state in [State(0, 0), State(0, 1), State(2, 2)]:
            rstate = env.restrict_state(state)
            assert rstate == state
        state = State(-1, 0)
        rstate = env.restrict_state(state)
        assert rstate == State(0, 0)
        state = State(5, -1)
        rstate = env.restrict_state(state)
        assert rstate == State(2, 0)


class TestGridWorldModel():

    def test_can_be_initialized(self):
        model = GridWorldModel([""])

    def test_can_simulate_observations_with_trivial_model(self):
        model = trivial_model()
        rs = np.random.RandomState(1)
        model.simulate_observations([0, 1], rs)

    def test_can_simulate_sensible_behavior_with_trivial_model(self):
        model = trivial_model(1000)
        rs = np.random.RandomState(1)
        obs = model.simulate_observations([-0.1, 1], rs)[0]
        for session in obs["sessions"]:
            assert len(session["path"]) < 3, session["path"]


class TestInitialStateUniformlyAtEdge:

    def test_raises_error_at_limit(self):
        gen = InitialStateUniformlyAtEdge(grid_size=3)
        with pytest.raises(ValueError):
            gen.get_initial_state(-1)
        with pytest.raises(ValueError):
            gen.get_initial_state(8)

    def test_gives_correct_results_on_small_grid(self):
        gen = InitialStateUniformlyAtEdge(grid_size=3)
        assert gen.n_initial_states == 8
        assert gen.get_initial_state(0) == State(0, 0)
        assert gen.get_initial_state(1) == State(1, 0)
        assert gen.get_initial_state(2) == State(2, 0)
        assert gen.get_initial_state(3) == State(2, 1)
        assert gen.get_initial_state(4) == State(2, 2)
        assert gen.get_initial_state(5) == State(1, 2)
        assert gen.get_initial_state(6) == State(0, 2)
        assert gen.get_initial_state(7) == State(0, 1)
