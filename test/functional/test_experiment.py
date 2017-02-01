import pytest
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run")

import os
from sdirl.run_test import run

@slow
def test_test_experiment_can_be_run():
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    run(file_dir_path)
    experiment_file = os.path.join(file_dir_path, "experiment.json")
    experiment_file_exists = True
    if os.path.isfile(experiment_file):
        os.remove(experiment_file)
    else:
        experiments_file_exists = False
    results_file = os.path.join(file_dir_path, "results.pdf")
    results_file_exists = True
    if os.path.isfile(results_file):
        os.remove(results_file)
    else:
        results_file_exists = False
    assert experiment_file_exists is True
    assert results_file_exists is True

