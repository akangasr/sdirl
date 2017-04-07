import os
import sys

import matplotlib
matplotlib.use('Agg')

from sdirl.experiments import *
from sdirl.model import *
from sdirl.choicemodel.model import ChoiceModelFactory, ChoiceModel
from sdirl.elfi_utils import BolfiParams
from sdirl.rl.simulator import RLParams

import logging
logger = logging.getLogger(__name__)

def get_model(parameters, ground_truth=None, observation=None):
    rl_params = RLParams(
                 n_training_episodes=50000000,
                 n_episodes_per_epoch=10000,
                 n_simulation_episodes=18000,
                 q_alpha=0.01,
                 q_gamma=1.0,
                 exp_epsilon=0.2,
                 exp_decay=1.0)
    cmf = ChoiceModelFactory(
                 parameters,
                 n_options=3,
                 p_alpha=1.0,
                 p_beta=1.0,
                 v_loc=19.60,
                 v_scale=8.08,
                 v_df=100,
                 n_training_sets=100000,
                 max_number_of_actions_per_session=20,
                 rl_params=rl_params,
                 ground_truth=ground_truth,
                 observation=observation)
    return cmf.get_new_instance(approximate=True)

def get_bolfi_params(parameters):
    params = BolfiParams()
    params.bounds = tuple([p.bounds for p in parameters])
    params.sync = True
    params.n_surrogate_samples = 40
    params.batch_size = 10
    params.noise_var = 0.5
    params.kernel_var = 2.00  # 50% of emp.max
    params.kernel_scale = 0.1  # 20% of smallest bounds
    params.kernel_class = GPy.kern.RBF
    params.gp_params_optimizer = "scg"
    params.gp_params_max_opt_iters = 100
    params.exploration_rate = 1.0
    params.acq_opt_iterations = 1000
    params.batches_of_init_samples = 1  # 20%
    params.inference_type = InferenceType.ML
    params.use_store = False  # because of discerror measure
    return params

def run_inference_experiment(parameters, bolfi_params, model, ground_truth=None):
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    results_file = os.path.join(file_dir_path, "results.pdf")

    if ground_truth is not None:
        error_classes=[L2Error]
    else:
        error_classes=[DiscrepancyError]
    experiment = InferenceExperiment(model,
            bolfi_params,
            ground_truth,
            plot_params = PlotParams(pdf_file=results_file),
            error_classes=error_classes)
    experiment.run()

    experiment_file = os.path.join(file_dir_path, "experiment.json")
    write_json_file(experiment_file, experiment.to_dict())

if __name__ == "__main__":
    env = Environment(sys.argv)

    inf_p = [
            #"alpha",
            #"calc_sigma",
            #"tau_p",
            #"tau_v",
            #"tau_r",
            #"tau_c",
            #"f_err",
            "step_penalty",
            ]

    vals = [("alpha", 1.5, 0, 2, "uniform", 0, 2),  # 1.5
            ("calc_sigma", 0, 0, 2, "uniform", 0, 2),  # 0.35
            ("tau_p", 0.03, 0, 0.2, "uniform", 0, 0.2),  # 0.011
            ("tau_v", 1.0, 0, 5, "uniform", 0, 5),  # 1.1
            ("tau_r", 1.0, 0, 5, "uniform", 0, 5),  # 2.0
            ("tau_c", 1.0, 0, 5, "uniform", 0, 5),  # 0.5
            ("f_err", 0.0, 0, 0.5, "uniform", 0, 0.5),  # 0.1
            ("step_penalty", -0.2, -1, 0, "uniform", -1, 1)]  # -0.2
    parameters = list()
    inf_parameters = list()
    for i in range(len(vals)):
        if vals[i][0] not in inf_p:
            bounds = (vals[i][1], vals[i][1])
            prior = ParameterPrior("uniform", bounds)
        else:
            bounds = (vals[i][2], vals[i][3])
            prior = ParameterPrior(vals[i][4], vals[i][5:])
        p = ModelParameter(name=vals[i][0], bounds=bounds, prior=prior)
        if vals[i][0] not in inf_p:
            print("fixed {}".format(p.to_dict()))
        else:
            print("infer {}".format(p.to_dict()))
            inf_parameters.append(p)
        parameters.append(p)
    ground_truth = None
    observation = None

    bolfi_params = get_bolfi_params(inf_parameters)
    bolfi_params.client = env.client

    #ground_truth = [0.5]
    observation = ChoiceModel.get_observation_dataset()

    model = get_model(parameters, ground_truth=ground_truth, observation=observation)
    run_inference_experiment(parameters, bolfi_params, model, ground_truth)
