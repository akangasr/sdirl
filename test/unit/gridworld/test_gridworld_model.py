import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run")

from collections import defaultdict
import numpy as np

from sdirl.gridworldmodel.model import *
from sdirl.gridworldmodel.mdp import *
from sdirl.rl.utils import Path, Transition

def trivial_model(training_episodes=100):
    return GridWorldModel(["feature1_value"],
        grid_size=3,
        step_penalty=0.1,
        prob_rnd_move=0.0,
        world_seed=0,
        n_training_episodes=training_episodes,
        n_episodes_per_epoch=10,
        n_simulation_episodes=100)

def simple_model(training_episodes=100, simulation_episodes=100, prob_rnd_move=0.1, use_test_grid=False):
    """ Simple 5x5 grid with very specific type of grid
    """
    model = GridWorldModel(["feature1_value", "feature2_value", "feature3_value", "feature4_value"],
        grid_size=5,
        step_penalty=0.1,
        prob_rnd_move=prob_rnd_move,
        world_seed=0,
        n_training_episodes=training_episodes,
        n_episodes_per_epoch=10,
        n_simulation_episodes=simulation_episodes)
    if use_test_grid is True:
        model.env.grid = get_test_grid()
    return model

def get_test_grid():
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
        env = GridWorldEnvironment(grid_size=3,
                target_state=State(0,0),
                grid_generator=UniformGrid())
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
        env = GridWorldEnvironment(grid_size=3,
                target_state=State(0,0),
                grid_generator=UniformGrid())
        trans_center = env.get_transitions(State(1,1))
        assert len(trans_center) == 4*4
        trans_corner = env.get_transitions(State(2,0))
        assert len(trans_corner) == 4*3

    @slow
    def test_transition_functions_are_coherent(self):
        n_iters = 100000
        env = GridWorldEnvironment(grid_size=3,
                target_state=State(0,0),
                prob_rnd_move=0.3,
                grid_generator=UniformGrid())
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

    def test_can_simulate_observations_with_trivial_model(self):
        model = trivial_model()
        rs = np.random.RandomState(1)
        model.simulate_observations([1], rs)

    def test_can_simulate_sensible_behavior_with_trivial_model(self):
        model = trivial_model(2000)
        rs = np.random.RandomState(1)
        obs = model.simulate_observations([-0.1], rs)
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

    @slow  # ~5min
    def test_discrepancy_results_are_sensible(self):
        rs = np.random.RandomState(1)
        obs = list()
        models = list()
        loc = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]]
        for l in loc:
            model = simple_model(100000, 1000, prob_rnd_move=0.05, use_test_grid=True)
            obs.append(model.simulate_observations(l, rs))
            models.append(model)  # use separate models to avoid recomputing the policy
        d00 = models[0].evaluate_discrepancy(loc[0], obs[0], rs)
        d01 = models[0].evaluate_discrepancy(loc[0], obs[1], rs)
        d02 = models[0].evaluate_discrepancy(loc[0], obs[2], rs)
        d10 = models[1].evaluate_discrepancy(loc[1], obs[0], rs)
        d11 = models[1].evaluate_discrepancy(loc[1], obs[1], rs)
        d12 = models[1].evaluate_discrepancy(loc[1], obs[2], rs)
        d20 = models[2].evaluate_discrepancy(loc[2], obs[0], rs)
        d21 = models[2].evaluate_discrepancy(loc[2], obs[1], rs)
        d22 = models[2].evaluate_discrepancy(loc[2], obs[2], rs)
        assert d00 < d01
        assert d00 < d02
        assert d11 < d10
        assert d11 < d12
        assert d22 < d20
        assert d22 < d21

    @slow  # ~30min
    def test_loglikelihood_results_are_sensible(self):
        rs = np.random.RandomState(1)
        obs = list()
        models = list()
        loc = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]]
        for l in loc:
            model = simple_model(100000, 100000, prob_rnd_move=0.1, use_test_grid=True)
            sim = model.simulate_observations(l, rs)
            n_obs = 100
            filt_sim = [s for s in sim if s.path_len < 6][:n_obs]  # to reduce computation time
            assert len(filt_sim) == n_obs
            obs.append(filt_sim)
            models.append(model)  # use separate models to avoid recomputing the policy
        ll00 = models[0].evaluate_loglikelihood(loc[0], obs[0], rs)
        ll01 = models[0].evaluate_loglikelihood(loc[0], obs[1], rs)
        ll02 = models[0].evaluate_loglikelihood(loc[0], obs[2], rs)
        ll10 = models[1].evaluate_loglikelihood(loc[1], obs[0], rs)
        ll11 = models[1].evaluate_loglikelihood(loc[1], obs[1], rs)
        ll12 = models[1].evaluate_loglikelihood(loc[1], obs[2], rs)
        ll20 = models[2].evaluate_loglikelihood(loc[2], obs[0], rs)
        ll21 = models[2].evaluate_loglikelihood(loc[2], obs[1], rs)
        ll22 = models[2].evaluate_loglikelihood(loc[2], obs[2], rs)
        assert ll00 > ll01
        assert ll00 > ll02
        assert ll11 > ll10
        assert ll11 > ll12
        assert ll22 > ll20
        assert ll22 > ll21

    @slow  # ~1min
    def test_policy_returns_correct_probabilities(self):
        model = simple_model(training_episodes=100000, prob_rnd_move=0, use_test_grid=True)
        loc = [-1, 0, 0, 0]
        rs = np.random.RandomState(1)
        model.simulate_observations(loc, rs)
        policy = model._get_optimal_policy(loc, None)
        # policy:
        # >>v<<
        # ^?v?^
        # ^>?<^
        # ^?^?^
        # ^<?>^
        assert policy(State(0,0), Action.UP) == 1.0
        assert policy(State(0,0), Action.DOWN) == 0.0
        assert policy(State(0,0), Action.RIGHT) == 0.0
        assert policy(State(0,0), Action.LEFT) == 0.0

        assert policy(State(1,0), Action.UP) == 1.0
        assert policy(State(2,0), Action.UP) == 1.0
        assert policy(State(3,0), Action.UP) == 1.0
        assert policy(State(4,0), Action.RIGHT) == 1.0
        assert policy(State(0,1), Action.LEFT) == 1.0
        assert policy(State(2,1), Action.RIGHT) == 1.0
        assert policy(State(4,1), Action.RIGHT) == 1.0
        assert policy(State(0,3), Action.RIGHT) == 1.0
        assert policy(State(2,3), Action.LEFT) == 1.0
        assert policy(State(4,3), Action.LEFT) == 1.0
        assert policy(State(0,4), Action.UP) == 1.0
        assert policy(State(1,4), Action.UP) == 1.0
        assert policy(State(2,4), Action.UP) == 1.0
        assert policy(State(3,4), Action.UP) == 1.0
        assert policy(State(4,4), Action.LEFT) == 1.0

    @slow  # ~30min
    def test_loglikelihood_results_are_accurate(self):
        rs = np.random.RandomState(1)
        obs = list()
        models = list()
        loc = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]]
        sel_obs = list()
        for l in loc:
            model = simple_model(20000, 1000000, prob_rnd_move=0.1, use_test_grid=True)
            obs.append(model.simulate_observations(l, rs))
            models.append(model)  # use separate models to avoid recomputing the policy
        common_obs = list(set(obs[0]).intersection(set(obs[1])).intersection(set(obs[2])))
        for i in range(5):
            while True:
                so = rs.choice(common_obs)
                if so.path_len < 6:
                    ep0 = float(sum([so == o for o in obs[0]]))
                    ep1 = float(sum([so == o for o in obs[1]]))
                    ep2 = float(sum([so == o for o in obs[2]]))
                    if ep0 > 500 and ep1 > 500 and ep2 > 500:
                        break
                common_obs.remove(so)
                assert len(common_obs) > 0
            print("{} counts: {}, {}, {}".format(so, ep0, ep1, ep2))
            ep01 = ep0 / ep1
            ep02 = ep0 / ep2
            ep12 = ep1 / ep2
            print("empirical: {}, {}, {}".format(ep01, ep02, ep12))
            ll0 = models[0].evaluate_loglikelihood(loc[0], [so], rs)
            ll1 = models[1].evaluate_loglikelihood(loc[1], [so], rs)
            ll2 = models[2].evaluate_loglikelihood(loc[2], [so], rs)
            p01 = np.exp(ll0 - ll1)
            p02 = np.exp(ll0 - ll2)
            p12 = np.exp(ll1 - ll2)
            print("theoretical: {}, {}, {}".format(p01, p02, p12))
            assert np.abs(p01 - ep01) / min([p01, ep01]) < 0.05
            assert np.abs(p02 - ep02) / min([p02, ep02]) < 0.05
            assert np.abs(p12 - ep12) / min([p12, ep12]) < 0.05


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

