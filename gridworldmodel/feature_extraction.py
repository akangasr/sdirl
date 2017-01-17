import numpy as np
import json

import logging
logger = logging.getLogger("experiment")

def get_feature_set(data):
    return {
        "00_number_of_actions_per_start": get_value(data, get_number_of_actions_per_start),
        "01_total_reward": get_value(data, get_total_reward),
        "02_rewards": get_list(data, get_rewards),
        "03_number_of_visits_per_cell": get_list(data, get_number_of_visits_per_cell),
        }

def get_list(data, extr_fun):
    ret = list()
    for session in data["sessions"]:
        ret.extend(extr_fun(session))
    return ret

def get_value(data, extr_fun):
    return [
            extr_fun(session)
            for session in data["sessions"]
            ]

def get_total_reward(session):
    return sum(session["reward"])

def get_rewards(session):
    return session["reward"]

def get_number_of_actions_per_start(session):
    return (session["start"], len(session["action"]))

def get_number_of_visits_per_cell(session):
    ret = list()
    cells = list()
    for cell in session["observation"]:
        cells.append(int(cell[0]))
    for i in range(len(session["grid"])**2):
        if i in cells:
            ret.append((i, 100))
        else:
            ret.append((i, 0))
    return ret
