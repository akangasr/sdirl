import math
import numpy as np

from menumodel.utils import vec_state_to_scalar

from pybrain.rl.environments import Environment, EpisodicTask

import logging
logger = logging.getLogger(__name__)


"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Definition of the MDP.
"""

class SearchTask(EpisodicTask):

    reward_success = 10000
    reward_failure = -10000

    def __init__(self, env, max_number_of_actions_per_session):
        EpisodicTask.__init__(self, None)
        self.env = env
        self.env.task = self
        self.max_number_of_actions_per_session = max_number_of_actions_per_session

    def reset(self):
        self.env.reset()

    def getReward(self):
        """ Returns the current reward based on the state of the environment
        """
        # this function should be deterministic and without side effects
        if self.env.click != None:
            if self.env.items[self.env.click] == "a":
                # reward for clicking the correct item after seeing it
                return int(self.reward_success)
            else:
                # penalty for clicking the wrong item
                return int(self.reward_failure)
        elif self.env.quit == True:
            if self.env.target == None:
                # reward for quitting when target is absent
                return int(self.reward_success)
            else:
                # penalty for quitting when target is present
                return int(self.reward_failure)
        # default penalty for spending time
        return int(-1 * self.env.last_action_duration)


    def isFinished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.env.n_actions >= self.max_number_of_actions_per_session:
            ret = True
        elif self.env.click != None:
            # click ends task
            ret = True
        elif self.env.quit == True:
            # quit ends task
            ret = True
        else:
            ret = False
        return ret


class SearchEnvironment(Environment):

    state_names = [
                "?", # semantic not observed, length not observed, not focused, not clicked
                "!", # semantic not observed, target length, not focused, not clicked
                "-", # semantic not observed, not target length, not focused, not clicked

                "A", # observed, target length, relevancy 1.0, focused, not clicked
                "B", # observed, target length, relevancy 0.6, focused, not clicked
                "C", # observed, target length, relevancy 0.3, focused, not clicked
                "D", # observed, target length, relevancy 0.0, focused, not clicked

                "E", # observed, not target length, relevancy 1.0, focused, not clicked
                "F", # observed, not target length, relevancy 0.6, focused, not clicked
                "G", # observed, not target length, relevancy 0.0, focused, not clicked
                "H", # observed, not target length, relevancy 0.3, focused, not clicked

                "I", # observed, length not observed, relevancy 1.0, focused, not clicked
                "J", # observed, length not observed, relevancy 0.6, focused, not clicked
                "K", # observed, length not observed, relevancy 0.0, focused, not clicked
                "L", # observed, length not observed, relevancy 0.3, focused, not clicked

                "a", # observed, target length, relevancy 1.0, not focused, not clicked
                "b", # observed, target length, relevancy 0.6, not focused, not clicked
                "c", # observed, target length, relevancy 0.3, not focused, not clicked
                "d", # observed, target length, relevancy 0.0, not focused, not clicked

                "e", # observed, not target length, relevancy 1.0, not focused, not clicked
                "f", # observed, not target length, relevancy 0.6, not focused, not clicked
                "g", # observed, not target length, relevancy 0.3, not focused, not clicked
                "h", # observed, not target length, relevancy 0.0, not focused, not clicked

                "i", # observed, length not observed, relevancy 1.0, not focused, not clicked
                "j", # observed, length not observed, relevancy 0.6, not focused, not clicked
                "k", # observed, length not observed, relevancy 0.3, not focused, not clicked
                "l", # observed, length not observed, relevancy 0.0, not focused, not clicked

                "X", # observed, relevancy 1.0, focused, clicked
                "Y", # observed, relevancy < 1.0, focused, clicked
                "Z", # observed, focused, exit
                ]

    def __init__(self,
            menu_type="semantic",
            menu_groups=2,
            menu_items_per_group=4,
            semantic_levels=3,
            gap_between_items=0.75,
            prop_target_absent=0.1,
            length_observations=True,
            p_obs_len_cur=0.95,
            p_obs_len_adj=0.89,
            n_training_menus=10000):
        """ Initializes the search environment
        """
        self.v = None # set with setup
        self.random_state = None # set with setup
        self.log = None # set by RL_model
        self.task = None # set by Task
        self.menu_type = menu_type
        self.menu_groups = menu_groups
        self.menu_items_per_group = menu_items_per_group
        self.semantic_levels = semantic_levels
        self.gap_between_items = gap_between_items
        self.prop_target_absent = prop_target_absent
        self.length_observations = length_observations
        self.p_obs_len_cur = p_obs_len_cur
        self.p_obs_len_adj = p_obs_len_adj
        self.n_training_menus = n_training_menus
        self.training_menus = list()
        self.training = True

        # technical variables
        self.state_vec_len = self.menu_groups * self.menu_items_per_group  # number of elements in state vector
        self.n_state_values = len(self.state_names)
        self.n_states = (self.n_state_values ** self.state_vec_len) + 1
        self.discreteStates = True
        self.outdim = 1
        self.indim = 1
        self.discreteActions = True
        self.numActions = self.state_vec_len + 2 # look + click + quit
        self.action_names = ["L%d" % (n) for n in range(self.state_vec_len)] + ["C", "Q"]


    def setup(self, variables, random_state):
        """ Finishes the initialization
        """
        self.v = variables
        self.random_state = random_state
        self.reset()


    def _get_menu(self):
        if self.training is True and len(self.training_menus) >= self.n_training_menus:
            idx = self.random_state.randint(0, self.n_training_menus)
            return self.training_menus[idx]
        # generate menu item semantic relevances and lengths
        items = ["d"] * self.state_vec_len
        if self.menu_type == "semantic":
            items, target = self._get_semantic_menu(self.menu_groups,
                        self.menu_items_per_group,
                        self.semantic_levels,
                        self.prop_target_absent)
        elif self.menu_type == "unordered":
            items, target = self._get_unordered_menu(self.menu_groups,
                        self.menu_items_per_group,
                        self.semantic_levels,
                        self.prop_target_absent)
        else:
            raise ValueError("Unknown menu type: %s" % (self.menu_type))
        lengths = self.random_state.randint(0,3,len(items)).tolist()
        if target != None:
            items[target] = "a"
            target_len = lengths[target]
        else:
            target_len = -1
        menu = (items, target, lengths, target_len)
        if self.training is True:
            self.training_menus.append(menu)
        return menu


    def reset(self):
        """ Called by the library to reset the state
        """
        self.items, self.target, self.lengths, self.target_len = self._get_menu()

        # state observed by agent
        self.obs_items            = ["?"] * self.state_vec_len
        self.obs_lens             = [None] * self.state_vec_len

        # misc environment state variables
        self.focus                = -1  # starts above first item in list
        self.click                = None
        self.last_action_duration = None
        self.quit                 = False
        self.n_actions            = 0
        self.locations            = np.arange(self.gap_between_items, self.gap_between_items*(self.state_vec_len+1), self.gap_between_items)

        # logging
        if self.log != None:
            if "session" not in self.log:
                self.log["session"] = 0
                self.log["sessions"] = [dict()]
            else:
                self.log["session"] += 1
                self.log["sessions"].append(dict())
            self.step_data = self.log["sessions"][self.log["session"]]
            self.step_data["target"]        = self.target
            self.step_data["items"]         = self.items
            self.step_data["observation"]   = list()
            self.step_data["action"]        = list()
            self.step_data["reward"]        = list()
            self.step_data["gaze_location"] = list()
            self.step_data["duration_focus_ms"]   = list()
            self.step_data["duration_saccade_ms"] = list()


    def performAction(self, action):
        """ Changes the state of the environment based on agent action
        """
        if self.log != None:
            self.step_data["observation"].append(self.getSensors())

        act = int(action[0])
        self.n_actions += 1

        # menu recall event may happen at first action
        if self.n_actions == 1:
            if "menu_recall_probability" in self.v and self.random_state.rand() < float(self.v["menu_recall_probability"]):
                self.obs_items = [i for i in self.items]
                self.obs_items[self.focus] = self.items[self.focus].upper()
                self.obs_lens = [self.lengths[i] == self.target_len for i in range(len(self.items))]

        # observe items's state
        if act < self.state_vec_len:
            # saccade
            if self.focus == -1:
                # initial location is above first item
                amplitude = abs(self.locations[0] - self.locations[act]) + abs(self.locations[0] - self.locations[1])
            else:
                self.obs_items[self.focus] = self.items[self.focus]
                amplitude = abs(self.locations[self.focus] - self.locations[act])
            saccade_duration = int(37 + 2.7 * amplitude)
            self.focus = act

            # fixation
            if "focus_duration_100ms" in self.v:
                focus_duration = int(float(self.v["focus_duration_100ms"]) * 100)
            else:
                focus_duration = 400
            # semantic observation at focus
            self.obs_items[self.focus] = self.items[self.focus].upper()
            # possible length observations
            if self.length_observations is True:
                if self.focus > 0 and self.random_state.rand() < self.p_obs_len_adj:
                    self.obs_lens[self.focus-1] = (self.lengths[self.focus-1] == self.target_len)
                if self.random_state.rand() < self.p_obs_len_cur:
                    self.obs_lens[self.focus] = (self.lengths[self.focus] == self.target_len)
                if self.focus < self.state_vec_len-1 and self.random_state.rand() < self.p_obs_len_adj:
                    self.obs_lens[self.focus+1] = (self.lengths[self.focus+1] == self.target_len)
            # possible semantic peripheral observations
            if "prob_obs_adjacent" in self.v:
                # above
                if self.focus > 0 and self.random_state.rand() < float(self.v["prob_obs_adjacent"]):
                    self.obs_items[self.focus-1] = self.items[self.focus-1]
                # below
                if self.focus < self.state_vec_len-1 and self.random_state.rand() < float(self.v["prob_obs_adjacent"]):
                    self.obs_items[self.focus+1] = self.items[self.focus+1]

        # choose item
        elif act == self.state_vec_len:
            if self.focus != -1:
                self.click = self.focus
                if self.items[self.click] == "a":
                    self.obs_items[self.focus] = "X"
                else:
                    self.obs_items[self.focus] = "Y"
            else:
                # trying to select an item when not focusing on any item equals to quitting
                self.quit = True
            if "selection_delay_s" in self.v:
                focus_duration = int(float(self.v["selection_delay_s"]) * 1000)
            else:
                focus_duration = 0
            saccade_duration = 0

        # quit without choosing any item
        elif act > self.state_vec_len:
            self.quit = True
            focus_duration = 0
            saccade_duration = 0
            if self.focus != -1:
                self.obs_items[self.focus] = "Z"

        self.last_action_duration = saccade_duration + focus_duration

        # logging
        if self.log != None:
            self.step_data["reward"].append(self.task.getReward())
            self.step_data["action"].append(act)
            self.step_data["gaze_location"].append(self.focus)
            self.step_data["duration_focus_ms"].append(focus_duration)
            self.step_data["duration_saccade_ms"].append(saccade_duration)

    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should be deterministic and without side effects
        return [vec_state_to_scalar(self._get_state(), self.n_state_values)] # needs to return a list

    def _get_state(self):
        # TODO: document this
        state = list()
        for o, l in zip(self.obs_items, self.obs_lens):
            if l is True:
                if o == "?":
                    state.append(self.state_names.index("!"))
                elif o in ["a", "b", "c", "d", "A", "B", "C", "D", "X", "Y", "Z"]:
                    state.append(self.state_names.index(o))
                else:
                    logger.critical("Error 1 {}".format(o))
            elif l is False:
                if o == "?":
                    state.append(self.state_names.index("-"))
                elif o in ["a", "b", "c", "d", "A", "B", "C", "D"]:
                    state.append(self.state_names.index(o) + 4)
                elif o in ["X", "Y", "Z"]:
                    state.append(self.state_names.index(o))
                else:
                    logger.critical("Error 2 {}".format(o))
            elif l is None:
                if o == "?":
                    state.append(self.state_names.index("?"))
                elif o in ["a", "b", "c", "d", "A", "B", "C", "D"]:
                    state.append(self.state_names.index(o) + 8)
                elif o in ["X", "Y", "Z"]:
                    state.append(self.state_names.index(o))
                else:
                    logger.critical("Error 3 {}".format(o))
            else:
                logger.critical("Obs len wrong? {}".format(l))
        return state


    def _semantic(self, n_groups, n_each_group, p_absent):
        n_items = n_groups * n_each_group
        target_value = 1

        """alpha and beta parameters for the menus with no target"""
        absent_menu_parameters = [2.1422, 13.4426]

        """alpha and beta for non-target/irrelevant menu items"""
        non_target_group_paremeters = [5.3665, 18.8826]

        """alpha and beta for target/relevant menu items"""
        target_group_parameters = [3.1625, 1.2766]

        semantic_menu = np.array([0] * n_items)[np.newaxis]

        """randomly select whether the target is present or abscent"""
        target_type = self.random_state.rand()
        target_location = self.random_state.randint(0, (n_items - 1))

        if target_type > p_absent:
            target_group_samples = self.random_state.beta \
                (target_group_parameters[0], target_group_parameters[1], (1, n_each_group))[0]
            """sample distractors from the Distractor group distribution"""
            distractor_group_samples = self.random_state.beta \
                (non_target_group_paremeters[0], non_target_group_paremeters[1], (1, n_items))[0];

            """ step 3 using the samples above to create Organised Menu and Random Menu
                and then add the target group
                the menu is created with all distractors first
            """
            menu1 = distractor_group_samples
            target_in_group = math.ceil((target_location + 1) / float(n_each_group))
            begin = (target_in_group - 1) * n_each_group
            end = (target_in_group - 1) * n_each_group + n_each_group

            menu1[begin:end] = target_group_samples
            menu1[target_location] = target_value
        else:
            target_location = None
            menu1 = self.random_state.beta\
                (absent_menu_parameters[0],\
                 absent_menu_parameters[1],\
                 (1, n_items))

        semantic_menu = menu1
        return semantic_menu, target_location


    def _get_unordered_menu(self, n_groups, n_each_group, n_grids, p_absent):
        assert(n_groups > 1)
        assert(n_each_group > 1)
        assert(n_grids > 0)
        semantic_menu, target = self._semantic(n_groups, n_each_group, p_absent)
        unordered_menu = self.random_state.permutation(semantic_menu)
        gridded_menu = self._griding(unordered_menu, target, n_grids)
        menu_length = n_each_group * n_groups
        coded_menu = ["d"] * menu_length
        start = 1 / float(2 * n_grids)
        stop = 1
        step = 1 / float(n_grids)
        grids = np.arange(start, stop, step)
        count = 0
        for item in gridded_menu:
                if False == (item - grids[0]).any():
                    coded_menu[count] = "d"
                elif False == (item - grids[1]).any():
                    coded_menu[count] = "c"
                elif False == (item - grids[2]).any():
                    coded_menu[count] = "b"
                count += 1

        return coded_menu, target


    def _griding(self, menu, target, n_levels):
        start = 1 / float(2 * n_levels)
        stop = 1
        step = 1 / float(n_levels)
        np_menu = np.array(menu)[np.newaxis]
        griding_semantic_levels = np.arange(start, stop, step)
        temp_levels = abs(griding_semantic_levels - np_menu.T)

        if target != None:
            min_index = temp_levels.argmin(axis=1)
            gridded_menu = griding_semantic_levels[min_index]
            gridded_menu[target] = 1
        else:
            min_index = temp_levels.argmin(axis=2)
            gridded_menu = griding_semantic_levels[min_index]

        return gridded_menu.T


    def _get_semantic_menu(self, n_groups, n_each_group, n_grids, p_absent):
        assert(n_groups > 0)
        assert(n_each_group > 0)
        assert(n_grids > 0)
        menu, target = self._semantic(n_groups, n_each_group, p_absent)
        #print menu
        gridded_menu = self._griding(menu, target, n_grids)
        menu_length = n_each_group*n_groups
        coded_menu = ["d"]*menu_length
        start = 1 / float(2 * n_grids)
        stop = 1
        step = 1 / float(n_grids)
        grids = np.arange(start, stop, step)
        count = 0
        for item in gridded_menu:
                if False == (item - grids[0]).any():
                    coded_menu[count] = "d"
                elif False == (item - grids[1]).any():
                    coded_menu[count] = "c"
                elif False == (item - grids[2]).any():
                    coded_menu[count] = "b"
                count += 1

        return coded_menu, target

