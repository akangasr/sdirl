import numpy as np

import elfi

import logging
logger = logging.getLogger(__name__)

"""An implementation of the Menu search model used in Kangasraasio et al. CHI 2017 paper.

Extract summary data of observations.
"""

@elfi.tools.vectorize
def feature_extraction(d):
    """ ELFI interface function.
    Sequential function.

    Parameters
    ----------
    d : np.ndarray containing dict
        Simulated data or processed observation data

    Returns
    -------
    Features of data as dict wrapped in np.ndarray
    """
    assert type(d) is np.ndarray, type(d)
    assert len(d) == 1, len(d)
    data = d[0]
    assert type(data) is dict, type(data)
    features = {
        "00_task_completion_time": _get_value(data, _get_task_completion_time, target_present=None),
        "01_task_completion_time_target_absent": _get_value(data, _get_task_completion_time, target_present=False),
        "02_task_completion_time_target_present": _get_value(data, _get_task_completion_time, target_present=True),
        #"03_fixation_duration": _get_list(data, _get_fixation_durations, target_present=None),
        "03_fixation_duration_target_absent": _get_list(data, _get_fixation_durations, target_present=False),
        "04_fixation_duration_target_present": _get_list(data, _get_fixation_durations, target_present=True),
        "05_saccade_duration_target_absent": _get_list(data, _get_saccade_durations, target_present=False),
        "06_saccade_duration_target_present": _get_list(data, _get_saccade_durations, target_present=True),
        "07_number_of_saccades_target_absent": _get_value(data, _get_number_of_saccades, target_present=False),
        "08_number_of_saccades_target_present": _get_value(data, _get_number_of_saccades, target_present=True),
        "09_fixation_locations_target_absent": _get_list(data, _get_fixation_locations, target_present=False),
        "10_fixation_locations_target_present": _get_list(data, _get_fixation_locations, target_present=True),
        "11_length_of_skips_target_absent": _get_list(data, _get_length_of_skips, target_present=False),
        "12_length_of_skips_target_present": _get_list(data, _get_length_of_skips, target_present=True),
        "13_location_of_gaze_to_target": _get_list(data, _get_location_of_gaze_to_target, target_present=True),
        "14_proportion_of_gaze_to_target": _get_value(data, _get_proportion_of_gaze_to_target, target_present=True),
        }
    return np.atleast_1d([features])

def _get_list(data, extr_fun, target_present):
    ret = list()
    for session in data["sessions"]:
        if _is_valid_condition(session, target_present) == True:
            ret.extend(extr_fun(session))
    return ret

def _get_value(data, extr_fun, target_present):
    return [
            extr_fun(session)
            for session in data["sessions"]
            if _is_valid_condition(session, target_present) == True
            ]

def _get_task_completion_time(session):
    return sum(session["duration_saccade_ms"]) + sum(session["duration_focus_ms"])

def _get_location_of_gaze_to_target(session):
    ret = list()
    for gaze_location in session["gaze_location"]:
        if gaze_location == session["target"]:
            ret.append(gaze_location)
    return ret

def _get_proportion_of_gaze_to_target(session):
    n_gazes_to_target = 0
    n_gazes = 0
    if len(session["gaze_location"]) < 1:
        return (session["target"], 0)
    for gaze_location in session["gaze_location"][:-1]: # last gaze location is end action, not a fixation
        n_gazes += 1
        if gaze_location == session["target"]:
            n_gazes_to_target += 1
    if n_gazes == 0:
        return (session["target"], 0)
    return (session["target"], n_gazes_to_target / n_gazes)

def _get_fixation_durations(session):
    if len(session["duration_focus_ms"]) < 1:
        return list()
    ret = session["duration_focus_ms"][:-1]  # assume all actions are eye fixations except last action
    if len(ret) < 1:
        # only end action
        ret = session["duration_focus_ms"]
    else:
        # adds the possible last action duration into the last fixation's duration
        ret[-1] += session["duration_focus_ms"][-1]
    return ret

def _get_length_of_skips(session):
    # assume all actions are eye fixations except last action
    if len(session["gaze_location"]) < 3:
        # need at least 2 fixations and one final action
        return list()
    ret = list()
    prev_loc = session["gaze_location"][0]
    for i in range(1, len(session["gaze_location"])-1):
        cur_loc = session["gaze_location"][i]
        ret.append(abs(cur_loc - prev_loc))
        prev_loc = cur_loc
    return ret

def _get_saccade_durations(session):
    return session["duration_saccade_ms"][:-1]  # assume all actions are eye fixations except last action

def _get_number_of_saccades(session):
    return len(session["action"]) - 1  # assume all actions are eye fixations except last action

def _get_fixation_locations(session):
    return session["action"][:-1]  # assume all actions are eye fixations except last action

def _is_valid_condition(session, target_present):
    if target_present is None:
        # None -> everything is ok
        return True
    if (target_present == True  and session["target"] != None) or \
        (target_present == False and session["target"] == None):
            return True
    return False

