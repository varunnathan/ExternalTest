"""
Usage
    python evaluate_sgd.py --ref_dir  path/to/test --pred_dir my_predictions
"""

import argparse, glob, json, sys
from src_may3.main.evaluate.metrics import compare_slot_values


# ---------- util -----------------------------------------------------------
def load_json_dir(path):
    return [json.load(open(p)) for p in glob.glob(f"{path}/*.json")]


# ---------- evaluation -------------------------------------------
def evaluate(ref_dialogues, pred_dialogues):
    """
    Returns dict with keys:
        active_intent_accuracy
        requested_slots_f1
        average_goal_accuracy
        joint_goal_accuracy
    """
    pass
