import numpy as np
import json

import pytest

from sdirl.model import *

def dummy_function(*args):
    return None

class DummyToDictable():
    def __init__(self, contents=dict()):
        self.contents = contents

    def to_dict(self):
        return self.contents

class TestParameterPrior():
    def test_can_be_serialized(self):
        distribution_name = "uniform"
        params = (0,1)
        p = ParameterPrior(distribution_name, params)
        d = p.to_dict()
        assert d["distribution_name"] == distribution_name
        assert d["params"] == params
        j = json.dumps(d)

class TestModelParameter():
    def test_can_be_serialized(self):
        name = "param1"
        prior = ParameterPrior()
        bounds = (0, 1)
        p = ModelParameter(name, prior, bounds)
        d = p.to_dict()
        assert d["name"] == name
        assert str(d["prior"]) == str(prior.to_dict())
        assert d["bounds"] == bounds
        j = json.dumps(d)

class TestObservationSummary():
    def test_can_be_serialized(self):
        name = "summary"
        function = dummy_function
        p = ObservationSummary(name, function)
        d = p.to_dict()
        assert d["name"] == name
        assert d["function"] == function.__name__
        j = json.dumps(d)

class TestObservationDataset():
    def test_can_be_serialized_with_value(self):
        data = 1
        parameter_values = [1,2]
        name = "name"
        p = ObservationDataset(data, parameter_values, name)
        d = p.to_dict()
        assert d["data"] == "1"
        assert d["parameter_values"] == ["1", "2"]
        assert d["name"] == name
        j = json.dumps(d)

    def test_can_be_serialized_with_list(self):
        data = [1]
        p = ObservationDataset(data)
        d = p.to_dict()
        assert d["data"] == ["1"]
        j = json.dumps(d)

    def test_can_be_serialized_with_to_dict_object(self):
        val = {"a": 1}
        data = [DummyToDictable(val)]
        p = ObservationDataset(data)
        d = p.to_dict()
        assert d["data"] == [val]
        j = json.dumps(d)


class TestModelBase():
    def test_can_be_serialized(self):
        name = "model"
        prior = ParameterPrior()
        param = ModelParameter("param1", prior, (0,1))
        parameters = [param]
        simulator = dummy_function
        summary = ObservationSummary("sum1", dummy_function)
        discrepancy = dummy_function
        observation = ObservationDataset(2)
        ground_truth = [1]
        m = ModelBase(name, parameters, simulator, [summary],
                      discrepancy, observation, ground_truth)
        d = m.to_dict()
        assert d["name"] == name
        assert str(d["parameters"]) == str([param.to_dict()])
        assert d["simulator"] == simulator.__name__
        assert str(d["summaries"]) == str([summary.to_dict()])
        assert d["discrepancy"] == discrepancy.__name__
        assert str(d["observation"]) == str(observation.to_dict())
        assert d["ground_truth"] == ["1"]
        j = json.dumps(d)


class TestSDIRLModel():
    def test_can_be_serialized(self):
        name = "model"
        prior = ParameterPrior()
        param = ModelParameter("param1", prior, (0,1))
        parameters = [param]
        observation = ObservationDataset(2)
        ground_truth = [1]
        env = DummyToDictable({"env": 1})
        task = DummyToDictable({"task": 1})
        rl = DummyToDictable({"rl": 1})
        goal_state = DummyToDictable({"state": 1})
        path_max_len = 3
        f = SDIRLModelFactory(name, parameters, env,
                 task, rl, goal_state, path_max_len,
                 SDIRLModel, observation, ground_truth)
        m1 = f.get_new_instance(approximate=True)
        d1 = m1.to_dict()
        assert d1["name"] == name
        assert str(d1["parameters"]) == str([param.to_dict()])
        assert d1["simulator"] == m1.simulate_observations.__name__
        assert d1["summaries"] == [{"name": "summary", "function": m1.summary_function.__name__}]
        assert d1["discrepancy"] == m1.calculate_discrepancy.__name__
        assert str(d1["observation"]) == str(observation.to_dict())
        assert d1["ground_truth"] == ["1"]
        assert d1["env"] == env.contents
        assert d1["task"] == task.contents
        assert d1["rl"] == rl.contents
        assert d1["goal_state"] == goal_state.contents
        assert d1["path_max_len"] == 3
        j1 = json.dumps(d1)
        m2 = f.get_new_instance(approximate=False)
        d2 = m2.to_dict()
        assert d2["name"] == d1["name"]
        assert d2["parameters"] == d1["parameters"]
        assert d2["simulator"] == m2.dummy_simulator.__name__
        assert d2["summaries"] == [{"name": "passthrough", "function": m2.passthrough_summary_function.__name__}]
        assert d2["discrepancy"] == m2.logl_discrepancy.__name__
        assert str(d2["observation"]) == str(observation.to_dict())
        assert d2["ground_truth"] == ["1"]
        assert d2["env"] == env.contents
        assert d2["task"] == task.contents
        assert d2["rl"] == rl.contents
        assert d2["goal_state"] == goal_state.contents
        assert d2["path_max_len"] == 3
        j2 = json.dumps(d2)

