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
    assert os.path.isfile(experiment_file)
    os.remove(experiment_file)

