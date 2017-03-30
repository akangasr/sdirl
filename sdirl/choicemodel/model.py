import numpy as np

from sdirl.choicemodel.mdp import ChoiceEnvironment, ChoiceTask, Action
from sdirl.rl.simulator import RLSimulator, RLParams
from sdirl.model import SDIRLModel, SDIRLModelFactory, ObservationDataset
import elfi
from matplotlib import pyplot as pl

import logging
logger = logging.getLogger(__name__)

class Observation():
    def __init__(self, pair_index, decoy_target, decoy_type, last_action):
        self.pair_index = pair_index
        self.decoy_target = decoy_target
        self.decoy_type = decoy_type
        self.last_action = last_action

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.pair_index, self.decoy_target, self.decoy_type, self.last_action).__hash__()

    def __repr__(self):
        return "O({},{},{},{})".format(self.pair_index, self.decoy_target, self.decoy_type, self.last_action)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Observation(self.pair_index, self.decoy_target, self.decoy_type, self.last_action)


class ChoiceModelFactory(SDIRLModelFactory):
    def __init__(self,
            parameters,
            n_options=3,
            p_alpha=1.0,
            p_beta=1.0,
            v_loc=19.60,
            v_scale=8.08,
            v_df=100,
            alpha=1.5,
            calc_sigma=0.35,
            tau_p=0.011,
            tau_v=1.1,
            f_err=0.1,
            n_training_sets=10000,
            max_number_of_actions_per_session=20,
            step_penalty=-0.1,
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
                    alpha=alpha,
                    calc_sigma=calc_sigma,
                    tau_p=tau_p,
                    tau_v=tau_v,
                    f_err=f_err,
                    n_training_sets=n_training_sets)
        task = ChoiceTask(
                    env=env,
                    step_penalty=step_penalty,
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
        return np.atleast_1d(ObservationDataset([
                Observation(ses["pair_index"], ses["decoy_target"], ses["decoy_type"], ses["action"][-1])
                for ses in obs[0].data["sessions"]
            ], name="summary"))

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
        table = {
                "AA": 0,
                "AB": 0,
                "BA": 0,
                "BB": 0,
                "D": 0,
                "X": 0
                }
        for index in range(60):  # assume wedell pairs
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
        return table

    def _table_discrepancy(self, table1, table2):
        s1 = float(sum([v for v in table1.values()]))
        s2 = float(sum([v for v in table1.values()]))
        d = 0.0
        for k in table1.keys():
            d += abs((table1[k] / s1) - (table2[k] / s2))
        return d

    def plot_obs(self, obs):
        assert isinstance(obs, ObservationDataset), type(obs)
        R_table = self._get_table("R", obs)
        F_table = self._get_table("F", obs)
        RF_table = self._get_table("RF", obs)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        self.plot_table(R_table, ax1)
        ax1.set_title("Range decoy")
        self.plot_table(F_table, ax2)
        ax2.set_title("Frequency decoy")
        self.plot_table(RF_table, ax3)
        ax3.set_title("Range-frequency decoy")

    def plot_table(self, table, ax):
        ax.table(cellText=[[table["AA"], table["BA"]], [table["AB"], table["BB"]]],
                 rowLabels=["chose A", "chose B"],
                 colLabels=["chose A", "chose B"])
