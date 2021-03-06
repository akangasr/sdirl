import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run")

from collections import defaultdict
import numpy as np

from sdirl.model import ModelParameter
from sdirl.menumodel.model import *
from sdirl.menumodel.mdp import *
from sdirl.rl.simulator import RLParams
from sdirl.rl.utils import Path, Transition

def simple_model(parameters=None, values=None):
    if parameters is None:
        parameters = [ModelParameter("focus_duration_100ms", bounds=(1,6))]
    if values is None:
        values = {"focus_duration_100ms": 4.0}
    rl_params = RLParams(
                 n_training_episodes=100000,
                 n_episodes_per_epoch=20,
                 n_simulation_episodes=100,
                 q_alpha=0.2)
    msf = MenuSearchFactory(
                 parameters,
                 menu_type="semantic",
                 menu_groups=2,
                 menu_items_per_group=4,
                 semantic_levels=3,
                 gap_between_items=0.75,
                 prop_target_absent=0.1,
                 length_observations=True,
                 n_training_menus=10000)
    model = msf.get_new_instance(approximate=True)
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
        parameters = [ModelParameter("focus_duration_100ms", bounds=(1,6)),
                      ModelParameter("selection_delay_s", bounds=(0,1))]
        values = {"focus_duration_100ms": 2.0,
                  "selection_delay_s": 0.3}
        env = simple_model(parameters, values).env
        env.start_logging()
        env.reset()
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.CLICK)])
        assert sum(env.log["sessions"][0]["action_duration"]) == 241 + 237 + 237 + 300
        env.reset()
        env.performAction([int(Action.QUIT)])
        assert sum(env.log["sessions"][1]["action_duration"]) == 0

    def test_actions_are_logged_corectly(self):
        env = simple_model().env
        env.start_logging()
        env.reset()
        env.performAction([int(Action.LOOK_1)])
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.LOOK_3)])
        env.performAction([int(Action.LOOK_4)])
        env.performAction([int(Action.CLICK)])
        assert len(env.log["sessions"][0]["action"]) == 5
        assert env.log["sessions"][0]["action"][0] == 0
        assert env.log["sessions"][0]["action"][1] == 1
        assert env.log["sessions"][0]["action"][2] == 2
        assert env.log["sessions"][0]["action"][3] == 3
        assert env.log["sessions"][0]["action"][4] == 8
        env.reset()
        env.performAction([int(Action.LOOK_8)])
        env.performAction([int(Action.LOOK_7)])
        env.performAction([int(Action.LOOK_6)])
        env.performAction([int(Action.LOOK_5)])
        env.performAction([int(Action.QUIT)])
        assert len(env.log["sessions"][0]["action"]) == 5
        assert env.log["sessions"][1]["action"][0] == 7
        assert env.log["sessions"][1]["action"][1] == 6
        assert env.log["sessions"][1]["action"][2] == 5
        assert env.log["sessions"][1]["action"][3] == 4
        assert env.log["sessions"][1]["action"][4] == 9

    def test_paths_are_logged_correctly(self):
        env = simple_model().env
        env.start_logging()
        env.reset()
        env.performAction([int(Action.LOOK_1)])
        env.performAction([int(Action.LOOK_2)])
        env.performAction([int(Action.LOOK_3)])
        env.performAction([int(Action.LOOK_4)])
        env.performAction([int(Action.CLICK)])
        print(env.log["sessions"][0]["path"])
        for i in range(len(env.log["sessions"][0]["path"].transitions)-1):
            assert env.log["sessions"][0]["path"].transitions[i].next_state == env.log["sessions"][0]["path"].transitions[i+1].prev_state, i
        assert env.log["sessions"][0]["path"].transitions[0].action == Action.LOOK_1
        assert env.log["sessions"][0]["path"].transitions[1].action == Action.LOOK_2
        assert env.log["sessions"][0]["path"].transitions[2].action == Action.LOOK_3
        assert env.log["sessions"][0]["path"].transitions[3].action == Action.LOOK_4

    @slow
    def test_target_item_at_equal_probability_at_any_location_or_absent(self):
        env = simple_model().env
        idx = [0,1,2,3,4,5,6,7,None]
        locs = dict()
        for i in idx:
            locs[i] = 0
        n_reps = 10000
        for i in range(n_reps):
            env.reset()
            locs[env.target_idx] += 1
        tgt = n_reps / float(len(idx))
        assert sum(locs.values()) == n_reps
        for i in idx:
            assert np.abs(locs[i] - tgt) < 100

    def test_menu_recall_probability_works_correctly(self):
        parameters = [ModelParameter("focus_duration_100ms", bounds=(1,6)),
                      ModelParameter("menu_recall_probability", bounds=(0,1))]
        values = {"focus_duration_100ms": 2.0,
                  "menu_recall_probability": 0.5}
        env = simple_model(parameters, values).env
        env.start_logging()
        for i in range(1000):
            env.reset()
            env.performAction([int(Action.LOOK_2)])
            env.performAction([int(Action.CLICK)])
        menus_recalled = 0
        for i in range(1000):
            second_state = env.log["sessions"][i]["path"].transitions[0].next_state
            recall = 1
            for item in second_state.obs_items:
                if item.item_relevance == ItemRelevance.NOT_OBSERVED or \
                   item.item_length == ItemLength.NOT_OBSERVED:
                       recall = 0
                       break
            menus_recalled += recall
        assert 450 < menus_recalled < 550, menus_recalled

    def test_length_observations_work_correctly(self):
        env = simple_model().env
        env.p_obs_len_cur = 0.9
        env.p_obs_len_adj = 0.5
        env.start_logging()
        for i in range(1000):
            env.reset()
            env.performAction([int(Action.LOOK_2)])
            env.performAction([int(Action.CLICK)])
        prev_length_obs = 0
        tgt_length_obs = 0
        next_length_obs = 0
        for i in range(1000):
            second_state = env.log["sessions"][i]["path"].transitions[0].next_state
            if second_state.obs_items[0].item_length != ItemLength.NOT_OBSERVED:
                prev_length_obs += 1
            if second_state.obs_items[1].item_length != ItemLength.NOT_OBSERVED:
                tgt_length_obs += 1
            if second_state.obs_items[2].item_length != ItemLength.NOT_OBSERVED:
                next_length_obs += 1
        assert 450 < prev_length_obs < 550, prev_length_obs
        assert 450 < next_length_obs < 550, next_length_obs
        assert 850 < tgt_length_obs < 950, tgt_length_obs

    def test_semantic_neighbor_observations_work_correctly(self):
        parameters = [ModelParameter("focus_duration_100ms", bounds=(1,6)),
                      ModelParameter("prob_obs_adjacent", bounds=(0,1))]
        values = {"focus_duration_100ms": 2.0,
                  "prob_obs_adjacent": 0.5}
        env = simple_model(parameters, values).env
        env.start_logging()
        for i in range(1000):
            env.reset()
            env.performAction([int(Action.LOOK_2)])
            env.performAction([int(Action.CLICK)])
        prev_rel_obs = 0
        tgt_rel_obs = 0
        next_rel_obs = 0
        for i in range(1000):
            second_state = env.log["sessions"][i]["path"].transitions[0].next_state
            if second_state.obs_items[0].item_relevance != ItemRelevance.NOT_OBSERVED:
                prev_rel_obs += 1
            if second_state.obs_items[1].item_relevance != ItemRelevance.NOT_OBSERVED:
                tgt_rel_obs += 1
            if second_state.obs_items[2].item_relevance != ItemRelevance.NOT_OBSERVED:
                next_rel_obs += 1
        assert 450 < prev_rel_obs < 550, prev_rel_obs
        assert 450 < next_rel_obs < 550, next_rel_obs
        assert tgt_rel_obs == 1000, tgt_rel_obs


class TestMenuSearchModel():

    def test_discrepancy_can_be_computed(self):
        rs = np.random.RandomState(1)
        obs = list()
        models = list()
        model = simple_model()
        loc = [4.0]
        sim = model.summary_function(model.simulate_observations(*loc, random_state=rs))[0]
        model.discrepancy(([sim],), ([sim],))

    @slow  # ~5min
    def test_discrepancy_results_are_sensible(self):
        rs = np.random.RandomState(1)
        obs = list()
        loc = [[1.0], [3.0], [5.0]]
        model = simple_model()
        for l in loc:
            obs.append(model.summary_function(model.simulate_observations(*l, random_state=rs))[0])
        d00 = model.discrepancy(([obs[0]],), ([obs[0]],))
        d01 = model.discrepancy(([obs[0]],), ([obs[1]],))
        d02 = model.discrepancy(([obs[0]],), ([obs[2]],))
        d10 = model.discrepancy(([obs[1]],), ([obs[0]],))
        d11 = model.discrepancy(([obs[1]],), ([obs[1]],))
        d12 = model.discrepancy(([obs[1]],), ([obs[2]],))
        d20 = model.discrepancy(([obs[2]],), ([obs[0]],))
        d21 = model.discrepancy(([obs[2]],), ([obs[1]],))
        d22 = model.discrepancy(([obs[2]],), ([obs[2]],))
        assert d00 < d01
        assert d00 < d02
        assert d11 < d10
        assert d11 < d12
        assert d22 < d20
        assert d22 < d21

