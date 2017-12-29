import numpy as np

from sdirl.choicemodel.mdp import ChoiceEnvironment, ChoiceTask, Action
from sdirl.rl.simulator import RLSimulator, RLParams
from sdirl.model import SDIRLModel, SDIRLModelFactory, ObservationDataset
import elfi
from matplotlib import pyplot as pl

import logging
logger = logging.getLogger(__name__)

class Observation():
    def __init__(self, pair_index, decoy_target, decoy_type, last_action, duration):
        self.pair_index = pair_index
        self.decoy_target = decoy_target
        self.decoy_type = decoy_type
        self.last_action = last_action
        self.duration = duration

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.pair_index, self.decoy_target, self.decoy_type, self.last_action, self.duration).__hash__()

    def __repr__(self):
        return "O({},{},{},{})".format(self.pair_index, self.decoy_target, self.decoy_type, self.last_action, self.duration)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Observation(self.pair_index, self.decoy_target, self.decoy_type, self.last_action, self.duration)


class ChoiceModelFactory(SDIRLModelFactory):
    def __init__(self,
            parameters,
            n_options=3,
            p_alpha=1.0,
            p_beta=1.0,
            v_loc=19.60,
            v_scale=8.08,
            v_df=100,
            reward_type="utility",
            n_training_sets=10000,
            max_number_of_actions_per_session=20,
            rl_params=RLParams(),
            observation=None,
            ground_truth=None):

        env = ChoiceEnvironment(
                    n_options=n_options,
                    p_alpha=p_alpha,
                    p_beta=p_beta,
                    v_loc=v_loc,
                    v_scale=v_scale,
                    v_df=v_df,
                    reward_type=reward_type,
                    n_training_sets=n_training_sets)
        task = ChoiceTask(
                    env=env,
                    max_number_of_actions_per_session=max_number_of_actions_per_session)
        rl = RLSimulator(
                    rl_params=rl_params,
                    parameters=parameters,
                    env=env,
                    task=task)
        super(ChoiceModelFactory, self).__init__(name="ChoiceModel",
                 parameters=parameters,
                 env=env,
                 task=task,
                 rl=rl,
                 klass=ChoiceModel,
                 observation=observation,
                 ground_truth=ground_truth)


class ChoiceModel(SDIRLModel):
    """ Choice model.
    """
    def summary_function(self, obs):
        if obs[0].name is "Wedell":
            return obs
        return np.atleast_1d(ObservationDataset([
                Observation(ses["pair_index"], ses["decoy_target"], ses["decoy_type"], ses["action"][-1], len(ses["action"]))
                for ses in obs[0].data["sessions"]
            ], name="summary"))

    @staticmethod
    def get_observation_dataset():
        """ Returns the Wedell dataset as an observation.
        """
        data = {"R": {
                "AA": 53.4,
                "AB": 20.3,
                "BA": 5.3,
                "BB": 19.0,
                "D": 2.0,
                "X": 0
                },
                "F": {
                "AA": 47.8,
                "AB": 22.2,
                "BA": 8.0,
                "BB": 20.0,
                "D": 2.0,
                "X": 0
                },
                "RF": {
                "AA": 55.5,
                "AB": 17.0,
                "BA": 5.7,
                "BB": 19.8,
                "D": 2.0,
                "X": 0
                },
                }

        return ObservationDataset(data, name="Wedell")

    def calculate_discrepancy(self, observations, sim_observations):
        R_table_obs = self._get_table("R", observations)
        R_table_sim = self._get_table("R", sim_observations)
        F_table_obs = self._get_table("F", observations)
        F_table_sim = self._get_table("F", sim_observations)
        RF_table_obs = self._get_table("RF", observations)
        RF_table_sim = self._get_table("RF", sim_observations)

        disc = self._table_discrepancy(R_table_obs, R_table_sim) + \
               self._table_discrepancy(F_table_obs, F_table_sim) + \
               self._table_discrepancy(RF_table_obs, RF_table_sim)

        return np.atleast_1d([disc])

    def _get_table(self, decoy_type, observation):
        obs = observation[0][0].data
        if observation[0][0].name == "Wedell":
            print("obs", decoy_type, obs[decoy_type])
            return obs[decoy_type]
        table = {
                "AA": 0,
                "AB": 0,
                "BA": 0,
                "BB": 0,
                "D": 0,
                "X": 0
                }
        #durations = list()
        n = 0
        for index in range(10):  # assume wedell pairs
            context_A = list()
            context_B = list()
            for o in obs:
                if o.pair_index == index and o.decoy_type == decoy_type:
                    if o.decoy_target == "A":
                        context_A.append(o.last_action)
                    elif o.decoy_target == "B":
                        context_B.append(o.last_action)
                    else:
                        assert False
                    #durations.append(o.duration)
            assert len(context_A) > 0
            assert len(context_A) == len(context_B)
            # if more than one pair, match in order of execution
            for a, b in zip(context_A, context_B):
                if a == Action.SELECT_1 and b == Action.SELECT_1:
                    table["AA"] += 1
                elif a == Action.SELECT_1 and b == Action.SELECT_2:
                    table["AB"] += 1 # reversal
                elif a == Action.SELECT_2 and b == Action.SELECT_1:
                    table["BA"] += 1 # inverse reversal
                elif a == Action.SELECT_2 and b == Action.SELECT_2:
                    table["BB"] += 1
                elif a == Action.SELECT_3 or b == Action.SELECT_3:
                    table["D"] += 1 # select decoy
                else:
                    logger.warning("End actions were {} and {}".format(a, b))
                    table["X"] += 1 # timeout?
                n += 1
        for k, v in table.items():
            table[k] = v * 100 / float(n)
        print("sim", decoy_type, table) #, np.mean(durations))
        return table

    def _table_discrepancy(self, table1, table2):
        s1 = float(sum([v for v in table1.values()]))
        s2 = float(sum([v for v in table2.values()]))
        rev1 = 100*(table1["AB"] / s1)
        rev2 = 100*(table2["AB"] / s2)
        irev1 = 100*(table1["BA"] / s1)
        irev2 = 100*(table2["BA"] / s2)
        diff1 = rev1 - irev1
        diff2 = rev2 - irev2
        dec1 = 100*(table1["D"] / s1)
        dec2 = 100*(table2["D"] / s2)
        d = (rev1 - rev2) ** 2 + (irev1 - irev2) ** 2 + (dec1 - dec2) ** 2 + (diff1 - diff2) ** 2
        return np.sqrt(d / 4.0)  # RMSE

    def plot_obs(self, obs):
        obs = self.summary_function([obs])
        R_table = self._get_table("R", [obs])
        F_table = self._get_table("F", [obs])
        RF_table = self._get_table("RF", [obs])
        f, (ax1, ax2, ax3) = pl.subplots(3, 1)
        self.plot_table(R_table, ax1)
        ax1.set_title("Range decoy")
        self.plot_table(F_table, ax2)
        ax2.set_title("Frequency decoy")
        self.plot_table(RF_table, ax3)
        ax3.set_title("Range-frequency decoy")

    def plot_table(self, table, ax):
        ax.axis("off")
        ax.table(cellText=[["stable A {:.2f} %".format(table["AA"]),
                            "inv.rev. {:.2f} %".format(table["BA"])],
                           ["reversal {:.2f} %".format(table["AB"]),
                            "stable B {:.2f} %".format(table["BB"])],
                           ["decoy {:.2f} %".format(table["D"]),
                            "none {:.2f} %".format(table["X"])],
                          ],
                 rowLabels=["B decoy -> chose A  ", "B decoy -> chose B  ", "other"],
                 colLabels=["A decoy -> chose A", "A decoy -> chose B"],
                 bbox=[0.2, 0.2, 0.90, 0.7])

