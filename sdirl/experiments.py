import os
import numpy as np
import json
import time

from sdirl.elfi_utils import *

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

class MLErrors(ExperimentResults):
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


class ComputeBolfiMLErrors(ExperimentPhase):
    def __init__(self, *args, error_measures=list(), **kwargs):
        super(ComputeBolfiMLErrors, self).__init__(*args, **kwargs)
        self.error_measures = error_measures
        self.results = MLErrors()

    def _run(self):
        bolfi_results = self.parents[0].results
        for error_measure in self.error_measures:
            self.results.errors["{}".format(error_measure)] = error_measure(bolfi_results)

    def _plot(self, plot_params):
        for e, errors in self.results.errors.items():
            self._print_error(e, errors)
            self._plot_error(e, errors, plot_params)

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
        pl.title("Reduction {} over time".format(e))
        plot_params.pdf.savefig()
        pl.close()

    def to_dict(self):
        ret = super(ComputeBolfiMLErrors, self).to_dict()
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

    def __repr__(self):
        return "Error"

    def __hash__(self):
        return (self.inference_type).__hash__()

    def to_dict(self):
        return {
                "inference_type": self.inference_type
                }


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
    """ Calculates proportion-based error (L2 distance between vectors of parameter proportions)
    """
    def _error(self, est):
        prop_gt = list()
        prop = list()
        for i in range(len(self.ground_truth)):
            for j in range(i+1, len(self.ground_truth)):
                prop_gt.append(float(self.ground_truth[i])/float(self.ground_truth[j]))
                prop.append(float(est[i])/float(est[j]))
        return np.linalg.norm(np.array(prop_gt) - np.array(prop), ord=2)

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


class GroundTruthInferenceExperiment(Experiment):
    def __init__(self,
            model,
            bolfi_params,
            ground_truth,
            plot_params=PlotParams(),
            error_classes=[L2Error, OrderError, ProportionError]):
        super(GroundTruthInferenceExperiment, self).__init__(plot_params)
        self.name = "Ground Truth Inference Experiment"
        self.model = model
        itf = InferenceTaskFactory(self.model)
        itask = itf.get_new_instance()
        self.bolfi_params = bolfi_params
        bf = BolfiFactory(itask, self.bolfi_params)
        bolfi = bf.get_new_instance()
        phase1 = ComputeBolfiPosterior(parents = [],
                                       bolfi = bolfi)
        self.add_phase(phase1)

        error_measures = [klass(ground_truth=ground_truth,
                                bolfi_params=self.bolfi_params.inference_type)
                          for klass in error_classes]
        self.add_phase(ComputeBolfiMLErrors(parents = [phase1],
                                      error_measures = error_measures))

    def to_dict(self):
        ret = super(GroundTruthInferenceExperiment, self).to_dict()
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

