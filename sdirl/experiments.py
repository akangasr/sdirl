import os
import numpy as np
import json
import time

from sdirl.elfi_utils import *
from sdirl.model import ObservationDataset
from elfi.async import wait

from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

import logging
logger = logging.getLogger(__name__)


class ExperimentResults():
    def to_dict(self):
        return dict()


class BolfiResults(ExperimentResults):
    def __init__(self):
        self.posteriors = list()
        self.duration = None

    def to_dict(self):
        return {
                "posteriors": [p.to_dict() for p in self.posteriors],
                "duration": self.duration,
                }

class Errors(ExperimentResults):
    def __init__(self):
        self.errors = dict()

    def to_dict(self):
        return {
                "errors": self.errors,
                }


class PlotParams():
    def __init__(self, pdf_file=None, pdf=None, figsize=(8.27, 11.69)):  # A4 portrait
        self.pdf_file = pdf_file
        self.pdf = pdf
        self.figsize = figsize


class ExperimentPhase():

    def __init__(self, parents=list()):
        self.name = "Experiment Phase"
        self.parents = parents
        self.results = ExperimentResults()

    def run(self, plot_params):
        """ Runs the experiment phase
        """
        for p in self.parents:
            p.run(plot_params)
        self._run()
        self.plot(plot_params)

    def _run(self):
        pass

    def plot(self, plot_params):
        if plot_params.pdf is None:
            return
        self._plot(plot_params)

    def _plot(self, plot_params):
        pass

    def _plot_text_page(self, text, plot_params):
        if plot_params.pdf is not None:
            fig = pl.figure(figsize=plot_params.figsize)
            fig.text(0.02, 0.01, text)
            plot_params.pdf.savefig()
            pl.close()

    def to_dict(self):
        return {
                "name": self.name,
                "results": self.results.to_dict()
                }


class ComputeBolfiPosterior(ExperimentPhase):

    def __init__(self, *args, bolfi=None, **kwargs):
        super(ComputeBolfiPosterior, self).__init__(*args, **kwargs)
        self.name = "Compute Bolfi Posterior"
        self.bolfi = bolfi
        self.results = BolfiResults()
        self._posterior_class = SerializableBolfiPosterior

    def _run(self):
        logger.info("Computing Bolfi posterior")
        start_time = time.time()
        posterior = self.bolfi.infer()
        posterior.__class__ = self._posterior_class  # hacky
        end_time = time.time()
        self.results.duration = end_time - start_time
        if self.bolfi.store is not None:
            self.results.posteriors = self._posterior_class.from_store(self.bolfi.store)
        else:
            self.results.posteriors = [posterior]
        final_posterior = self.results.posteriors[-1]
        if hasattr(final_posterior, "ML"):
            logger.info("Final ML estimate: {}".format(final_posterior.ML))
        if hasattr(final_posterior, "MAP"):
            logger.info("Final MAP estimate: {}".format(final_posterior.MAP))

    def _plot(self, plot_params):
        for posterior in self.results.posteriors:
            self._plot_posterior(posterior, plot_params)

    def _plot_posterior(self, posterior, plot_params):
        if posterior.model.gp is None:
            self._plot_text_page("No model to plot", plot_params)
            return
        fig = pl.figure(figsize=plot_params.figsize)
        try:
            posterior.model.gp.plot()
        except:
            fig.text(0.02, 0.01, "Was not able to plot model")
        plot_params.pdf.savefig()
        pl.close()


class ComputeBolfiErrors(ExperimentPhase):
    def __init__(self, *args, error_measures=list(), **kwargs):
        super(ComputeBolfiErrors, self).__init__(*args, **kwargs)
        self.name = "Compute Bolfi Errors"
        self.error_measures = error_measures
        self.results = Errors()

    def _run(self):
        bolfi_results = self.parents[0].results
        for error_measure in self.error_measures:
            self.results.errors["{}".format(error_measure)] = error_measure(bolfi_results)

    def _plot(self, plot_params):
        for e, errors in self.results.errors.items():
            self._print_error(e, errors)
            self._plot_error(e, errors, plot_params)
        for em in self.error_measures:
            em.plot(plot_params)

    def _print_error(self, e, errors, grid=30):
        logger.info("{}".format(e))
        lim = max(errors) / float(grid)
        for n in reversed(range(grid+1)):
            st = ["{: >+5.3f}".format(n*lim)]
            for e in errors:
                if e >= n*lim:
                    st.append("*")
                else:
                    st.append(" ")
            logger.info("".join(st))

    def _plot_error(self, e, errors, plot_params):
        fig = pl.figure(figsize=plot_params.figsize)
        t = range(len(errors))
        pl.plot(t, errors)
        pl.xlabel("BO samples")
        pl.ylabel(str(e))
        pl.ylim(min(errors)-0.1, max(errors)+0.1)
        pl.title("Reduction {} over time".format(e))
        plot_params.pdf.savefig()
        pl.close()

    def to_dict(self):
        ret = super(ComputeBolfiErrors, self).to_dict()
        for e in self.error_measures:
            ret["{}".format(e)] = e.to_dict()
        return ret


class ErrorMeasure():
    def __init__(self, *args, inference_type=InferenceType.ML, **kwargs):
        self.inference_type = inference_type

    def __call__(self, bolfi_results):
        estimates = self._get_bolfi_estimates(bolfi_results)
        return [float(self._error(est)) for est in estimates]

    def _error(self, value):
        return float("NaN")

    def _get_bolfi_estimates(self, bolfi_results):
        if self.inference_type == InferenceType.ML:
            estimates = [p.ML for p in bolfi_results.posteriors]
        elif self.inference_typdde == InferenceType.MAP:
            estimates = [p.MAP for p in bolfi_results.posteriors]
        else:
            raise NotImplementedError("Error not implemented for {} inference"\
                    .format(self.inference_type))
        return estimates

    def plot(self, plot_params):
        pass

    def __repr__(self):
        return "Error"

    def __hash__(self):
        return (self.inference_type).__hash__()

    def to_dict(self):
        return {
                "inference_type": self.inference_type
                }


class DiscrepancyError(ErrorMeasure):
    """ Calculates discrepancy to observations (assumed to be already inside the itask)

    Parameters
    ----------
    itask : elfi.InferenceTask
    """
    def __init__(self, *args, model=None, itask=None, client=None, **kwargs):
        super(DiscrepancyError, self).__init__(*args, **kwargs)
        self.model = model
        self.itask = itask
        self.client = client
        self._plot_store = list()  # hacky
        self._obs = None  # hacky
        self.called = False

    def _error(self, value):
        # hacky, depends on elfi and dask implementation details
        discrepancy = self._compute_discrepancy(value)
        sim = self._get_simulator()
        if self._obs is None:
            self._obs = sim.observed[0][0]
        data = self._get_last_sim_data(sim)
        self._plot_store.append((value, data))
        return discrepancy

    def _compute_discrepancy(self, value):
        wv_dict = {param.name: np.atleast_2d(value[i])
                               for i, param in enumerate(self.itask.parameters)}
        logger.info("Simulating data with values {}..".format(value))
        future = self.itask.discrepancy.generate(1, with_values=wv_dict)
        result, _a, _b = elfi.wait([future], self.client)
        logger.info("Simulated")
        return float(result)

    def _get_simulator(self):
        return self.itask._find_by_class(elfi.Simulator)[0]

    def _get_last_sim_data(self, sim):
        i = 9999  # assume larger than any sample id we have
        data = None
        logger.info("Finding last sim data..")
        while True:
            data = sim[i].compute()
            if len(data) > 0:
                break
            i -= 1
        logger.info("Found at idx {}".format(i))
        assert isinstance(data[0][0], ObservationDataset), data
        return data[0][0]

    def plot(self, plot_params):
        # hacky
        long_fig = (8.27, 25)
        fig = pl.figure(figsize=long_fig)
        fig.text(0.02, 0.01, "Observed data")
        self.model.plot_obs(self._obs)
        plot_params.pdf.savefig()
        pl.close()
        for v, data in self._plot_store:
            fig = pl.figure(figsize=long_fig)
            fig.text(0.02, 0.01, "Simulated at {}".format(v))
            self.model.plot_obs(data)
            plot_params.pdf.savefig()
            pl.close()

    def to_dict(self):
        ret = super(DiscrepancyError, self).to_dict()
        ret["itask"] = "inference task object" if self.itask is not None else None
        ret["itask"] = "client object" if self.client is not None else None
        return ret


class GroundTruthError(ErrorMeasure):
    """ Calculates error to ground_truth
    """
    def __init__(self, *args, ground_truth=[], **kwargs):
        super(GroundTruthError, self).__init__(*args, **kwargs)
        self.ground_truth = np.atleast_1d(ground_truth)

    def to_dict(self):
        ret = super(GroundTruthError, self).to_dict()
        ret["ground_truth"] = [str(v) for v in self.ground_truth]
        return ret


class L2Error(GroundTruthError):
    """ Calculates error using L2 distance
    """
    def _error(self, est):
        return np.linalg.norm(self.ground_truth - est, ord=2)

    def __repr__(self):
        return "L2 error to {}".format(self.ground_truth)

    def __hash__(self):
        return (("L2", self.inference_type) + tuple(self.ground_truth)).__hash__()


class OrderError(GroundTruthError):
    """ Calculates ordering-based error (hamming distance of ranks)
    """
    def _error(self, est):
        order_gt = np.argsort(self.ground_truth)
        order = np.argsort(est)
        return sum([o1 != o2 for o1, o2 in zip(order_gt, order)])

    def __repr__(self):
        return "Order error to {}".format(self.ground_truth)

    def __hash__(self):
        return (("Order", self.inference_type) + tuple(self.ground_truth)).__hash__()


class ProportionError(GroundTruthError):
    """ Calculates proportion-based error (log10 L2 distance between vectors of parameter proportions)
    """
    def _error(self, est):
        prop_gt = list()
        prop = list()
        for i in range(len(self.ground_truth)):
            for j in range(i+1, len(self.ground_truth)):
                prop_gt.append(float(self.ground_truth[i])/float(self.ground_truth[j]))
                prop.append(float(est[i])/float(est[j]))
        return np.log10(float(np.linalg.norm(np.array(prop_gt) - np.array(prop), ord=2)))

    def __repr__(self):
        return "Proportion error to {}".format(self.ground_truth)

    def __hash__(self):
        return (("Proportion", self.inference_type) + tuple(self.ground_truth)).__hash__()


class Experiment():
    def __init__(self, plot_params):
        self.name = "Experiment"
        self.phases = set()
        self.plot_params = plot_params

    def run(self):
        logger.info("Running {}".format(self.name))
        if self.plot_params.pdf_file is not None:
            with PdfPages(self.plot_params.pdf_file) as pdf:
                self.plot_params.pdf = pdf
                self._run()
            logger.info("Plotted results to {}".format(self.plot_params.pdf_file))
        else:
            logger.info("Not plotting results")
            self._run()

    def _run(self):
        parents = set().union(*[set(p.parents) for p in self.phases])
        for phase in self.phases:
            if phase not in parents:
                phase.run(self.plot_params)

    def add_phase(self, phase):
        self.phases.add(phase)

    def to_dict(self):
        return {
                "name": self.name,
                "phases": [p.to_dict() for p in self.phases]
                }


class InferenceExperiment(Experiment):
    def __init__(self,
            model,
            bolfi_params,
            ground_truth=None,
            plot_params=PlotParams(),
            error_classes=[L2Error, OrderError, ProportionError]):
        super(InferenceExperiment, self).__init__(plot_params)
        self.name = "Inference Experiment "
        assert model is not None
        if model.observation is not None:
            self.name += "with observation dataset"
        elif ground_truth is not None:
            self.name += "with ground truth = {}".format(ground_truth)
        else:
            raise ValueError("Need either ground truth or observation")
        self.model = model
        itf = InferenceTaskFactory(self.model)
        itask = itf.get_new_instance()
        self.bolfi_params = bolfi_params
        bf = BolfiFactory(itask, self.bolfi_params)
        bolfi = bf.get_new_instance()
        phase1 = ComputeBolfiPosterior(parents = [],
                                       bolfi = bolfi)
        self.add_phase(phase1)

        error_measures = []
        for klass in error_classes:
            if issubclass(klass, GroundTruthError):
                error_measures.append(klass(ground_truth=ground_truth,
                                            inference_type=self.bolfi_params.inference_type))
            elif issubclass(klass, DiscrepancyError):
                error_measures.append(klass(model=model,
                                            itask=itask,
                                            client=self.bolfi_params.client,
                                            inference_type=self.bolfi_params.inference_type))
            else:
                raise ValueError("Unknown error class {}".format(klass))

        self.add_phase(ComputeBolfiErrors(parents = [phase1],
                                      error_measures = error_measures))

    def to_dict(self):
        ret = super(InferenceExperiment, self).to_dict()
        ret["model"] = self.model.to_dict()
        ret["bolfi params"] = self.bolfi_params.to_dict()
        return ret


def write_json_file(filename, data):
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
    logger.info("Wrote {}".format(filename))

def read_json_file(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    logger.info("Read {}".format(filename))
    return data

