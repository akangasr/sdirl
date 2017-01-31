import numpy as np

from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment

from sdirl.rl.pybrain_extensions import SparseActionValueTable, EpisodeQ, EGreedyExplorer

import logging
logger = logging.getLogger(__name__)

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Generic RL simulator.
"""

class RLSimulator():

    def __init__(self,
            n_training_episodes,
            n_episodes_per_epoch,
            n_simulation_episodes,
            var_names,
            env,
            task):
        """

        Parameters
        ----------
        n_training_episodes : int
            Number of episodes to train before simulating data
        n_episodes_per_epochs : int
            Number of training episodes between offline learning
        n_simulation_episodes : int
            Number of episodes to simulate after training
        var_names : list of strings
            Names of variables, in order
        env : Environment model
        task : EpisodecTask instance
        softq : bool
            True : uses soft q values for actions
            False : uses hard q values for actions
        """
        self.n_training_episodes = n_training_episodes
        self.n_episodes_per_epoch = n_episodes_per_epoch
        self.n_simulation_episodes = n_simulation_episodes
        self.var_names = var_names
        self.env = env
        self.task = task
        self.softq = False
        self.agent = None

    def to_dict(self):
        return {
                "n_training_episodes": self.n_training_episodes,
                "n_episodes_per_epoch": self.n_episodes_per_epoch,
                "n_simulation_episodes": self.n_simulation_episodes,
                "var_names": self.var_names,
                }

    def train_model(self, *variables, random_state=None):
        self._set_variables(variables)
        self._build_model(random_state)
        self._train_model()

    def __call__(self, *args, random_state=None):
        """ Simulates data.
        Interfaces to ELFI as a sequential simulator.

        Parameters
        ----------
        args : list of model variables
            Length should equal length of var_names
        random_state: random number generator

        Returns
        -------
        Simulated trajectories as a dict encapsulated in 1D numpy array
        """
        self.train_model(*args, random_state=random_state)
        log_dict = self.simulate(random_state)
        return np.atleast_1d([log_dict])

    def get_policy(self):
        """ Returns the current policy of the agent
        """
        return self.agent.get_policy()

    def _set_variables(self, args):
        """ Parse variable values
        """
        self.variables = dict()
        if len(self.var_names) != len(args):
            raise ValueError("Number of model variables was {}, expected {}"
                    .format(len(args), len(self.var_names)))
        for name, val in zip(self.var_names, args):
            self.variables[name] = val
        logger.debug("Model parameters: {}".format(self.variables))

    def _build_model(self, random_state):
        """ Initialize the model
        """
        self.env.setup(self.variables, random_state)
        self.task.setup(self.variables)
        outdim = self.task.env.outdim
        n_actions = self.task.env.numActions
        self.agent = RLAgent(outdim, n_actions, random_state, self.softq)
        logger.debug("Model initialized")

    def _train_model(self):
        """ Uses reinforcement learning to find the optimal strategy
        """
        self.experiment = EpisodicExperiment(self.task, self.agent)
        n_epochs = int(self.n_training_episodes / self.n_episodes_per_epoch)
        logger.debug("Fitting user model over {} epochs, each {} episodes, total {} episodes."
                .format(n_epochs, self.n_episodes_per_epoch, n_epochs*self.n_episodes_per_epoch))
        for i in range(n_epochs):
            self.experiment.doEpisodes(self.n_episodes_per_epoch)
            self.agent.learn()
            self.agent.reset()  # reset buffers

    def simulate(self, random_state):
        """ Simulates agent behavior in 'n_sim' episodes.
        """
        logger.debug("Simulating user actions ({} episodes)".format(self.n_simulation_episodes))
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
        self.experiment.doEpisodes(self.n_simulation_episodes)
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
    def __init__(self, outdim, n_actions, random_state, softq=False):
        """ RL agent
        """
        module = SparseActionValueTable(n_actions, random_state, softq=softq)
        module.initialize(0.0)
        learner = EpisodeQ(alpha=0.3, gamma=0.998)
        learner.explorer = EGreedyExplorer(random_state, epsilon=0.1, decay=1.0)
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

