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
                 n_training_episodes=10000000,
                 n_episodes_per_epoch=1000,
                 n_simulation_episodes=1800,
                 q_alpha=0.01,
                 q_gamma=1.0,
                 exp_epsilon=1.0,
                 exp_decay=0.999999)
    cmf = ChoiceModelFactory(
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
                 n_training_sets=50000,
                 max_number_of_actions_per_session=20,
                 step_penalty=-1.0,
                 rl_params=rl_params,
                 ground_truth=ground_truth,
                 observation=observation)
    return cmf.get_new_instance(approximate=True)

def get_bolfi_params(parameters):
    params = BolfiParams()
    params.bounds = tuple([p.bounds for p in parameters])
    params.sync = True
    params.n_surrogate_samples = 10
    params.batch_size = 2
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

    fix_params = (   1, 2, 3, 4, 5, 6)
    #fix_params = (0,    2, 3, 4, 5, 6)
    #fix_params = (0, 1,    3, 4, 5, 6)
    #fix_params = (0, 1, 2,    4, 5, 6)
    #fix_params = (0, 1, 2, 3,    5, 6)
    #fix_params = (0, 1, 2, 3, 4,    6)
    #fix_params = (0, 1, 2, 3, 4, 5   )

    vals = [("alpha", 1.5, 0.1, 1.9, "uniform", 0.1, 1.9),
            ("calc_sigma", 0.35, 0.01, 5.0, "uniform", 0.01, 5.0),
            ("tau_p", 0.011, 0.001, 0.5, "uniform", 0.001, 0.5),
            ("tau_v", 1.1, 0.1, 5.0, "uniform", 0.1, 5.0),
            ("tau_r", 2.0, 0.1, 5.0, "uniform", 0.1, 5.0),
            ("f_err", 0.1, 0.01, 0.5, "uniform", 0.01, 0.5),
            ("step_penalty", -1.0, -2.0, 0.0, "uniform", -2.0, 0.0)]
    parameters = list()
    inf_parameters = list()
    for i in range(7):
        if i in fix_params:
            bounds = (vals[i][1], vals[i][1])
            prior = ParameterPrior("uniform", bounds)
        else:
            bounds = (vals[i][2], vals[i][3])
            prior = ParameterPrior(vals[i][4], vals[i][5:])
        p = ModelParameter(name=vals[i][0], bounds=bounds, prior=prior)
        if i in fix_params:
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
