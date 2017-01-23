import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run")

import numpy as np

from sdirl.gridworldmodel.model import GridWorldModel, Observation
from sdirl.gridworldmodel.mdp import InitialStateUniformlyAtEdge
from sdirl.gridworldmodel.mdp import InitialStateUniformlyAnywhere
from sdirl.gridworldmodel.mdp import GridWorldEnvironment
from sdirl.gridworldmodel.mdp import State
from sdirl.rl.utils import Path, Transition

def trivial_model(training_episodes=100):
    return GridWorldModel(["step_penalty", "feature1_value"],
        grid_size=3,
        prob_rnd_move=0.0,
        world_seed=0,
        n_training_episodes=training_episodes,
        n_episodes_per_epoch=10,
        n_simulation_episodes=100)

def simple_model(training_episodes=100, simulation_episodes=100, use_test_grid=False):
    """ Simple 5x5 grid with very specific type of grid
    """
    model = GridWorldModel(["step_penalty", "feature1_value", "feature2_value", "feature3_value", "feature4_value"],
        grid_size=5,
        prob_rnd_move=0.01,
        world_seed=0,
        n_training_episodes=training_episodes,
        n_episodes_per_epoch=10,
        n_simulation_episodes=simulation_episodes)
    if use_test_grid is True:
        model.grid = test_grid()
    return model

def test_grid():
    """ Return test 5x5 grid that looks like this:
        A A A A A
        A B C B A
        A D X E A
        A B F B A
        A A A A A
    """
    A = [0, 0, 0, 0, 0]
    B = [0, 1, 1, 1, 1]
    C = [0, 0, 1, 1, 1]
    D = [0, 1, 0, 1, 1]
    E = [0, 1, 1, 0, 1]
    F = [0, 1, 1, 1, 0]
    X = [1, 0, 0, 0, 0]
    return {
        State(4, 0): A, State(4, 1): A, State(4, 2): A, State(4, 3): A, State(4, 4): A,
        State(3, 0): A, State(3, 1): B, State(3, 2): C, State(3, 3): B, State(3, 4): A,
        State(2, 0): A, State(2, 1): D, State(2, 2): X, State(2, 3): E, State(2, 4): A,
        State(1, 0): A, State(1, 1): B, State(1, 2): F, State(1, 3): B, State(1, 4): A,
        State(0, 0): A, State(0, 1): A, State(0, 2): A, State(0, 3): A, State(0, 4): A
        }

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

    def test_returns_correct_number_of_state_transitions(self):
        env = GridWorldEnvironment(grid_size=3, target_state=State(0,0))
        trans_center = env.get_transitions(State(1,1))
        assert len(trans_center) == 4*4
        trans_corner = env.get_transitions(State(2,0))
        assert len(trans_corner) == 4*3

    @slow
    def test_transition_functions_are_coherent(self):
        n_iters = 100000
        env = GridWorldEnvironment(grid_size=3, target_state=State(0,0), prob_rnd_move=0.3)
        env.random_state = np.random.RandomState(4)
        for init_state in [State(1,1), State(0,0), State(1,2)]:
            next_states = set([t.next_state for t in env.get_transitions(init_state)])
            for action in env.actions:
                emp_next_states = [env.do_transition(init_state, action) for i in range(n_iters)]
                for next_state in next_states:
                    transition = Transition(init_state, action, next_state)
                    emp_trans_prob = sum(next_state == s for s in emp_next_states) / float(n_iters)
                    trans_prob = env.transition_prob(transition)
                    assert 1.0 > emp_trans_prob > 0.0
                    assert 1.0 > trans_prob > 0.0
                    assert np.abs(emp_trans_prob - trans_prob) < 0.005, transition


class TestGridWorldModel():

    def test_can_be_initialized(self):
        model = GridWorldModel([""])

    def test_can_simulate_observations_with_trivial_model(self):
        model = trivial_model()
        rs = np.random.RandomState(1)
        model.simulate_observations([0, 1], rs)

    def test_can_simulate_sensible_behavior_with_trivial_model(self):
        model = trivial_model(2000)
        rs = np.random.RandomState(1)
        obs = model.simulate_observations([-0.2, -0.1], rs)
        for o in obs:
            assert o.path_len < 3
 
    def test_can_generate_all_probable_paths_for_observation(self):
        model = trivial_model()
        for init_state in [State(0,0), State(2,2)]:
            obs = Observation(start_state=init_state, path_len=2)
            paths = model.get_all_paths_for_obs(obs)
            transitions = model.env.get_transitions(init_state)
            expected_paths = list()
            for t1 in transitions:
                next_transitions = model.env.get_transitions(t1.next_state)
                for t2 in next_transitions:
                    expected_paths.append(Path([t1, t2]))
            i = 0
            for path in paths:
                assert path in expected_paths
                assert model._prob_obs(obs, path) > 0.0, (obs, path)
                i += 1
            assert i == len(expected_paths)

    @slow
    def test_discrepancy_results_are_sensible(self):
        rs = np.random.RandomState(1)
        obs = list()
        models = list()
        loc = [[-0.1, -1, 0, 0, 0], [-0.1, 0, -1, 0, 0], [-0.1, 0, 0, -1, 0]]
        for l in loc:
            model = simple_model(20000, use_test_grid=True)
            obs.append(model.simulate_observations(l, rs))
            models.append(model)  # use separate models to avoid recomputing the policy
        assert model.evaluate_discrepancy(loc[0], obs[0], rs) < model.evaluate_discrepancy(loc[0], obs[1], rs)
        assert model.evaluate_discrepancy(loc[0], obs[0], rs) < model.evaluate_discrepancy(loc[0], obs[2], rs)
        assert model.evaluate_discrepancy(loc[1], obs[1], rs) < model.evaluate_discrepancy(loc[1], obs[0], rs)
        assert model.evaluate_discrepancy(loc[1], obs[1], rs) < model.evaluate_discrepancy(loc[1], obs[2], rs)
        assert model.evaluate_discrepancy(loc[2], obs[2], rs) < model.evaluate_discrepancy(loc[2], obs[0], rs)
        assert model.evaluate_discrepancy(loc[2], obs[2], rs) < model.evaluate_discrepancy(loc[2], obs[1], rs)

    @slow
    def test_loglikelihood_results_are_sensible(self):
        rs = np.random.RandomState(1)
        obs = list()
        models = list()
        loc = [[-0.1, -1, 0, 0, 0], [-0.1, 0, -1, 0, 0], [-0.1, 0, 0, -1, 0]]
        for l in loc:
            model = simple_model(20000, use_test_grid=True)
            obs.append(model.simulate_observations(l, rs))
            models.append(model)  # use separate models to avoid recomputing the policy
        assert models[0].evaluate_loglikelihood(loc[0], obs[0], rs) > models[0].evaluate_loglikelihood(loc[0], obs[1], rs)
        assert models[0].evaluate_loglikelihood(loc[0], obs[0], rs) > models[0].evaluate_loglikelihood(loc[0], obs[2], rs)
        assert models[1].evaluate_loglikelihood(loc[1], obs[1], rs) > models[1].evaluate_loglikelihood(loc[1], obs[0], rs)
        assert models[1].evaluate_loglikelihood(loc[1], obs[1], rs) > models[1].evaluate_loglikelihood(loc[1], obs[2], rs)
        assert models[2].evaluate_loglikelihood(loc[2], obs[2], rs) > models[2].evaluate_loglikelihood(loc[2], obs[0], rs)
        assert models[2].evaluate_loglikelihood(loc[2], obs[2], rs) > models[2].evaluate_loglikelihood(loc[2], obs[1], rs)

    #@slow
    def test_loglikelihood_results_are_accurate(self):
        model = simple_model(50000, 1, use_test_grid=True)
        rs = np.random.RandomState(1)
        loc = [-0.1, 0, 0, 0, -1]
        obs = [model.simulate_observations(loc, rs)[0] for i in range(100)]
        unique_obs = set(obs)
        emp_p = list()
        pred_p = list()
        for uo in unique_obs:
            emp_p.append(sum([o == uo for o in obs]) / float(len(obs)))
            pred_p.append(np.exp(model.evaluate_loglikelihood(loc, [uo], rs)))
        norm_pred_p = np.array(pred_p) / sum(pred_p)
        for ep, pp in zip(emp_p, norm_pred_p):
            assert np.abs(ep - pp) / pp < 0.1, (emp_p, norm_pred_p)


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


class TestInitialStateUniformlyAnywhere:

    def test_raises_error_at_limit(self):
        gen = InitialStateUniformlyAnywhere(grid_size=3)
        with pytest.raises(ValueError):
            gen.get_initial_state(-1)
        with pytest.raises(ValueError):
            gen.get_initial_state(9)

    def test_gives_correct_results_on_small_grid(self):
        gen = InitialStateUniformlyAnywhere(grid_size=3)
        assert gen.n_initial_states == 8
        assert gen.get_initial_state(0) == State(0, 0)
        assert gen.get_initial_state(1) == State(1, 0)
        assert gen.get_initial_state(2) == State(2, 0)
        assert gen.get_initial_state(3) == State(0, 1)
        assert gen.get_initial_state(4) == State(2, 2)
        assert gen.get_initial_state(5) == State(2, 1)
        assert gen.get_initial_state(6) == State(0, 2)
        assert gen.get_initial_state(7) == State(1, 2)

