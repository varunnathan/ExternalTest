import glob, json, sys
from main.evaluate.metrics import compare_slot_values


def load_json_dir(path):
    return [json.load(open(p)) for p in glob.glob(f"{path}/*.json")]


def evaluate(ref_dialogues, pred_dialogues, schemas):
    """
    Returns dict with keys:
        active_intent_accuracy
        requested_slots_f1
        average_goal_accuracy
        joint_goal_accuracy
    """
    intent_correct = total_turns = 0
    req_tp = req_fp = req_fn = 0
    slot_correct = slot_total = 0
    joint_correct = 0

    for ref_dlg, pred_dlg in zip(ref_dialogues, pred_dialogues):
        # iterate turn‑wise
        for ref_turn, pred_turn in zip(ref_dlg["turns"], pred_dlg["turns"]):
            if ref_turn.get("speaker") != "USER":
                continue  # metrics only on USER turns

            total_turns += 1

            # build service→frame map for quick lookup
            ref_frames = {f["service"]: f for f in ref_turn["frames"]}
            pred_frames = {f["service"]: f for f in pred_turn["frames"]}

            # evaluate each service present in reference turn
            turn_joint_ok = True
            for svc, ref_f in ref_frames.items():
                pred_f = pred_frames.get(svc, {
                    "state": {"active_intent":"NONE",
                              "requested_slots":[],
                              "slot_values":{}}})

                ref_state, pred_state = ref_f["state"], pred_f["state"]

                # -- active intent ------------------------------------------------
                if ref_state["active_intent"] == pred_state["active_intent"]:
                    intent_correct += 1
                else:
                    turn_joint_ok = False

                # -- requested slots ---------------------------------------------
                r_req = set(ref_state["requested_slots"])
                p_req = set(pred_state["requested_slots"])
                if r_req or p_req:
                    req_tp += len(r_req & p_req)
                    req_fp += len(p_req - r_req)
                    req_fn += len(r_req - p_req)

                # -- slot value accuracy -----------------------------------------
                for slot, ref_vals in ref_state["slot_values"].items():
                    if not ref_vals:          # consider only non‑empty GT slots
                        continue
                    slot_total += 1
                    pred_vals = pred_state["slot_values"].get(slot, [])
                    if compare_slot_values(ref_vals, pred_vals,
                                           schemas[svc], use_fuzzy_match=True):
                        slot_correct += 1
                    else:
                        turn_joint_ok = False

                # any extra slots predicted that GT doesn't have ⇒ joint fails
                for slot in pred_state["slot_values"]:
                    if slot not in ref_state["slot_values"]:
                        turn_joint_ok = False

            if turn_joint_ok:
                joint_correct += 1

    metrics = {
        "active_intent_accuracy": intent_correct / total_turns,
        "requested_slots_f1": (
            2 * req_tp / (2 * req_tp + req_fp + req_fn)
            if (req_tp + req_fp + req_fn) else 0.0),
        "average_goal_accuracy": slot_correct / slot_total if slot_total else 0.0,
        "joint_goal_accuracy": joint_correct / total_turns,
    }
    return metrics


# ------------------------------------------------------------------- CLI
def main_evaluate(ref_dir, pred_dir, schemas):
    refs  = load_json_dir(ref_dir)
    preds = load_json_dir(pred_dir)

    # align by dialogue_id
    ref_map  = {d["dialogue_id"]: d for tmp_refs in refs for d in tmp_refs if "dialogue_id" in d}
    pred_map = {d["dialogue_id"]: d for d in preds if "dialogue_id" in d}
    missing  = set(ref_map) - set(pred_map)
    if missing:
        print("ERROR: Missing predictions for", len(missing), "dialogues")
        sys.exit(1)

    ordered_refs  = [ref_map[k]  for k in sorted(ref_map)]
    ordered_preds = [pred_map[k] for k in sorted(ref_map)]

    m = evaluate(ordered_refs, ordered_preds, schemas)
    print("—— SGD metrics ——")
    print(f"Active-Intent Acc : {m['active_intent_accuracy'] * 100:.2f}")
    print(f"Requested-Slot F1 : {m['requested_slots_f1'] * 100:.2f}")
    print(f"Average-Goal Acc  : {m['average_goal_accuracy'] * 100:.2f}")
    print(f"Joint-Goal Acc    : {m['joint_goal_accuracy'] * 100:.2f}")
