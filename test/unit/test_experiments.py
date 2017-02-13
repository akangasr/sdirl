from unittest.mock import Mock, MagicMock

from elfi.posteriors import Posterior
from elfi.methods import BOLFI

from sdirl.experiments import *

class TestExperimentPhase():
    def test_runs_parents_when_run(self):
        p1 = Mock(ExperimentPhase)
        p2 = Mock(ExperimentPhase)
        ep = ExperimentPhase(parents=[p1, p2])
        ep.run(PlotParams())
        assert p1.run.call_count == 1
        assert p2.run.call_count == 1


class TestComputeBolfiPosterior():
    def run_phase_with_mock(self, has_store=False):
        self.bolfi = Mock(BOLFI)
        self.bolfi.store = None
        self.retval = Posterior()
        self.bolfi.infer = MagicMock(return_value=self.retval)
        self.ep = ComputeBolfiPosterior(bolfi=self.bolfi)
        self.ep.run(PlotParams())

    def test_runs_bolfi_inference_when_run(self):
        self.run_phase_with_mock()
        assert self.bolfi.infer.call_count == 1

    def test_stores_final_posterior_if_no_store(self):
        self.run_phase_with_mock()
        assert len(self.ep.results.posteriors) == 1
        assert self.ep.results.posteriors[0] == self.retval
        assert self.ep.results.duration > 0


class TestExperiment():
    def test_calls_all_leaf_phases(self):
        e = Experiment(plot_params=PlotParams())
        p1 = Mock(ExperimentPhase)
        p1.parents = []
        e.add_phase(p1)
        p2 = Mock(ExperimentPhase)
        p2.parents = [p1]
        e.add_phase(p2)
        p3 = Mock(ExperimentPhase)
        p3.parents = []
        e.add_phase(p3)
        e.run()
        p1.run.assert_not_called()
        assert p2.run.call_count == 1
        assert p3.run.call_count == 1

