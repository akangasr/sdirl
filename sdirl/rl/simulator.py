import numpy as np

from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment

from sdirl.rl.pybrain_extensions import SparseActionValueTable, EpisodeQ, EGreedyExplorer

import logging
logger = logging.getLogger(__name__)

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Generic RL simulator.
"""

class RLParams():
    def __init__(self,
            n_training_episodes=1,
            n_episodes_per_epoch=1,
            n_simulation_episodes=1,
            q_alpha=1.0,
            q_w=1.0,
            q_gamma=0.98,
            exp_epsilon=0.1,
            exp_decay=1.0,
            soft_q=False,
            soft_temp=1.0):
        self.n_training_episodes = n_training_episodes
        self.n_episodes_per_epoch = n_episodes_per_epoch
        self.n_simulation_episodes = n_simulation_episodes
        self.q_alpha = q_alpha
        self.q_w = q_w
        self.q_gamma = q_gamma
        self.exp_epsilon = exp_epsilon
        self.exp_decay = exp_decay
        self.soft_q = soft_q
        self.soft_temp = soft_temp

    def to_dict(self):
        return {
            "n_training_episodes": self.n_training_episodes,
            "n_episodes_per_epoch": self.n_episodes_per_epoch,
            "n_simulation_episodes": self.n_simulation_episodes,
            "q_alpha": self.q_alpha,
            "q_w": self.q_w,
            "q_gamma": self.q_gamma,
            "exp_epsilon": self.exp_epsilon,
            "exp_decay": self.exp_decay,
            "soft_q": self.soft_q,
            "soft_temp": self.soft_temp,
            }


class RLSimulator():

    def __init__(self,
            rl_params,
            parameters,
            env,
            task):
        """

        Parameters
        ----------
        rl_params : RLParams
        parameters : list of ModelParameter
        env : Environment model
        task : EpisodecTask instance
        """
        self.rl_params = rl_params
        self.parameters = parameters
        self.env = env
        self.task = task
        self.agent = None

    def to_dict(self):
        return {
                "rl_params": self.rl_params.to_dict(),
                "parameters": [p.to_dict() for p in self.parameters],
                }

    def train_model(self, parameter_values, random_state=None):
        self._set_parameter_values(parameter_values)
        self._build_model(random_state)
        self._train_model()

    def __call__(self, parameter_values, random_state=None):
        """ Simulates data.
        Interfaces to ELFI as a sequential simulator.

        Parameters
        ----------
        parameter_values : list of model variables
            Length should equal length of parameters
        random_state: random number generator

        Returns
        -------
        Simulated trajectories as a dict encapsulated in 1D numpy array
        """
        self.train_model(parameter_values, random_state=random_state)
        log_dict = self.simulate(random_state)
        return np.atleast_1d([log_dict])

    def get_policy(self):
        """ Returns the current policy of the agent
        """
        return self.agent.get_policy()

    def _set_parameter_values(self, parameter_values):
        """ Parse parameter values
        """
        self.v = dict()
        if len(self.parameters) != len(parameter_values):
            raise ValueError("Number of model variables was {} ({}), expected {}"
                    .format(len(parameter_values), parameter_values, len(self.parameters)))
        for param, val in zip(self.parameters, parameter_values):
            self.v[param.name] = val
            if param.name == "RL_soft_temp":  # hack
                self.rl_params.soft_temp = val
        logger.debug("Model parameters: {}".format(self.v))

    def _build_model(self, random_state):
        """ Initialize the model
        """
        self.env.setup(self.v, random_state)
        self.task.setup(self.v)
        outdim = self.task.env.outdim
        n_actions = self.task.env.numActions
        self.agent = RLAgent(outdim,
                n_actions,
                random_state,
                rl_params=self.rl_params)
        logger.debug("Model initialized")

    def _train_model(self):
        """ Uses reinforcement learning to find the optimal strategy
        """
        self.experiment = EpisodicExperiment(self.task, self.agent)
        n_epochs = int(self.rl_params.n_training_episodes / self.rl_params.n_episodes_per_epoch)
        logger.debug("Fitting user model over {} epochs, each {} episodes, total {} episodes."
                .format(n_epochs, self.rl_params.n_episodes_per_epoch, n_epochs*self.rl_params.n_episodes_per_epoch))
        for i in range(n_epochs):
            logger.debug("RL epoch {}".format(i))
            self.experiment.doEpisodes(self.rl_params.n_episodes_per_epoch)
            self.agent.learn()
            self.agent.reset()  # reset buffers

    def simulate(self, random_state):
        """ Simulates agent behavior in 'n_sim' episodes.
        """
        logger.debug("Simulating user actions ({} episodes)".format(self.rl_params.n_simulation_episodes))
        self.experiment = EpisodicExperiment(self.task, self.agent)

        # set training flag off
        self.task.env.training = False
        # deactivate learning for experiment
        self.agent.learning = False
        # deactivate exploration
        explorer = self.agent.learner.explorer
        self.agent.learner.explorer = EGreedyExplorer(epsilon=0, decay=1, random_state=random_state)
        self.agent.learner.explorer.module = self.agent.module
        # activate logging
        self.task.env.start_logging()

        # simulate behavior
        self.experiment.doEpisodes(self.rl_params.n_simulation_episodes)
        # store log data
        dataset = self.task.env.log

        # deactivate logging
        self.task.env.end_logging()
        # reactivate exploration
        self.agent.learner.explorer = explorer
        # reactivate learning for experiment
        self.agent.learning = True
        # set training flag back on
        self.task.env.training = True

        return dataset


class RLAgent(LearningAgent):
    def __init__(self, outdim, n_actions, random_state, rl_params):
        """ RL agent
        """
        module = SparseActionValueTable(n_actions,
                                        random_state,
                                        soft_q=rl_params.soft_q,
                                        soft_temp=rl_params.soft_temp)
        module.initialize(0.0)
        learner = EpisodeQ(alpha=rl_params.q_alpha,
                           w=rl_params.q_w,
                           gamma=rl_params.q_gamma)
        learner.explorer = EGreedyExplorer(random_state,
                                           epsilon=rl_params.exp_epsilon,
                                           decay=rl_params.exp_decay)
        LearningAgent.__init__(self, module, learner)

    def get_policy(self):
        return Policy(self.module)


class Policy():
    """ Encapsulated policy
    """

    def __init__(self, module):
        self.module = module

    def __call__(self, state, action):
        """ Returns p(action | state) accoring to deterministic policy with ties broken arbitrarily
        """
        s = float(state.__hash__())  # pybrain secretly casts state to float when we do rl
        a = int(action)
        qvalues = self.module.getActionValues(s)
        maxq = max(qvalues)
        if qvalues[a] == maxq:
            n_max = sum([1 if q == maxq else 0 for q in qvalues])
            return 1.0 / n_max
        return 0

