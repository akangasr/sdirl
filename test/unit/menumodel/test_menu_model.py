import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run")

from collections import defaultdict
import numpy as np

from sdirl.menumodel.model import *
from sdirl.menumodel.mdp import *
from sdirl.rl.utils import Path, Transition

def simple_model(variables=["focus_duration_100ms"], values=[4.0]):
    model = MenuSearchModel(
                 variables,
                 menu_type="semantic",
                 menu_groups=2,
                 menu_items_per_group=4,
                 semantic_levels=3,
                 gap_between_items=0.75,
                 prop_target_absent=0.1,
                 length_observations=False,
                 n_training_menus=10000,
                 n_training_episodes=100000,
                 n_episodes_per_epoch=20,
                 n_simulation_episodes=100,
                 verbose=True)
    model.env.setup(values, np.random.RandomState(0))
    return model

class TestSearchMDP():

    @slow  # ~5s
    def test_menu_generation_works_as_expected(self):
        env = simple_model().env
        n_menus = env.n_training_menus
        training_menus = set([env._get_menu() for i in range(n_menus*2)])
        assert len(training_menus) <= n_menus  # one menu created at start, so might not be ==
        env.training = False
        new_menus = [env._get_menu() for i in range(n_menus)]
        same = 0
        for menu in new_menus:
            # after training should also get fresh menus
            if menu in training_menus:
                same += 1
        assert same < n_menus

    def test_initialization_works_as_expected(self):
        env = simple_model().env
        for item in env.state.obs_items:
            assert item.item_relevance == ItemRelevance.NOT_OBSERVED
            assert item.item_length == ItemLength.NOT_OBSERVED
        assert env.state.focus == Focus.ABOVE_MENU
        assert env.state.click == Click.NOT_CLICKED
        assert env.state.quit == Quit.NOT_QUIT
        assert env.prev_state == env.state
        for item in env.items:
            assert item.item_relevance != ItemRelevance.NOT_OBSERVED
            assert item.item_length != ItemLength.NOT_OBSERVED

    def test_look_transitions_work_as_expected(self):
        env = simple_model().env
        env.performAction([int(Action.LOOK_2)])
        assert env.state.focus == Focus.ITEM_2
        env.performAction([int(Action.LOOK_6)])
        assert env.state.focus == Focus.ITEM_6
        env.performAction([int(Action.LOOK_8)])
        assert env.state.focus == Focus.ITEM_8
        env.performAction([int(Action.LOOK_1)])
        assert env.state.focus == Focus.ITEM_1

    def test_taking_maximum_number_of_actions_leads_to_task_ending(self):
        model = simple_model()
        env = model.env
        task = model.task
        for i in range(task.max_number_of_actions_per_session):
            assert task.isFinished() is False
            env.performAction([int(Action.LOOK_1)])
        assert task.isFinished() is True

    def test_click_transitions_work_as_expected(self):
        model = simple_model()
        env = model.env
        task = model.task
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.CLICK)])
        assert env.state.click == Click.CLICK_2
        assert task.isFinished() is True
        env.reset()
        assert task.isFinished() is False
        env.performAction([int(Action.LOOK_6)])
        env.performAction([int(Action.LOOK_8)])
        env.performAction([int(Action.CLICK)])
        assert env.state.click == Click.CLICK_8
        assert task.isFinished() is True

    def test_quit_transitions_work_as_expected(self):
        model = simple_model()
        env = model.env
        task = model.task
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.CLICK)])
        assert env.state.click == Click.CLICK_2
        assert task.isFinished() is True
        env.reset()
        assert task.isFinished() is False
        env.performAction([int(Action.LOOK_6)])
        env.performAction([int(Action.LOOK_8)])
        env.performAction([int(Action.CLICK)])
        assert env.state.click == Click.CLICK_8
        assert task.isFinished() is True

    def test_total_duration_of_session_is_logged_correctly(self):
        env = simple_model().env
        env.start_logging()
        env.reset()
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.CLICK)])
        assert sum(env.log["sessions"][0]["action_duration"]) == 441 + 437 + 437  # assume focus duration 400ms
        env.reset()
        env.performAction([int(Action.QUIT)])
        assert sum(env.log["sessions"][1]["action_duration"]) == 0


class TestMenuSearchModel():

    @slow  # ~5min
    def test_can_simulate_sensible_behavior_with_simple_model(self):
        model = simple_model()
        rs = np.random.RandomState(1)
        obs = model.simulate_observations([4.0], rs)
        for o in obs:
            assert o.task_completion_time < 8000

    @slow  # ~10min
    def test_discrepancy_results_are_sensible(self):
        rs = np.random.RandomState(1)
        obs = list()
        models = list()
        loc = [[1.0], [3.0], [5.0]]
        for l in loc:
            model = simple_model()
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
