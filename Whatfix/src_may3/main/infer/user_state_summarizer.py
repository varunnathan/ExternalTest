from collections import defaultdict
from main.data_model.config import InferenceConfig

_THRESH = {
    "categorical":     0.80,
    "free_form":       0.50,
    "requested_slot":  0.90,
    "transfer_in":     0.85,
    "transfer_cross":  0.90,
}


class UserStateSummariser:
    """
    Aggregate raw model outputs into the SGD 'state' object for one service.
    ------------------------------------------------------------------------
    Returns
        new_state      : dict { active_intent, requested_slots, slot_values }
    """

    def __init__(self, service_schema: dict, inference_config: InferenceConfig):
        self.schema = service_schema
        # map slot → role (required/optional) for current intent filled later
        self._intent_slot_role = {}
        self.prob_thresh = inference_config.prob_threshold
    # ---------------------------------------------------------------------
    # public entry
    # ---------------------------------------------------------------------
    def update(self, prev_state: dict, preds: dict) -> dict:
        intent_name = preds["intent"]["pred"]
        intent_prob = preds["intent"]["prob"]

        # ---------- 1. decide active_intent  ------------------------------
        # paper: they always take arg‑max intent; probability is not thresholded
        active_intent = intent_name

        # build required / optional slot sets for this intent
        self._update_intent_slot_roles(active_intent)

        # ---------- 2. start with previous state's slot_values ------------
        slot_values = {k: v[:] for k, v in prev_state.get("slot_values", {}).items()}
        requested_slots = set()

        # ---------- 3. categorical slot values ---------------------------
        self._apply_slot_value_preds(slot_values, preds["categorical"],
                                     self.prob_thresh["categorical_slot_prediction"])

        # ---------- 4. free‑form slot values -----------------------------
        self._apply_slot_value_preds(slot_values, preds["free_form"],
                                     self.prob_thresh["free_form_slot_prediction"])

        # ---------- 5. requested slots -----------------------------------
        for r in preds["requested_slot"]:
            if r["prob"] >= self.prob_thresh["requested_slot_prediction"]:
                requested_slots.add(r["slot"])

        # ---------- 6. transfer predictions ------------------------------
        self._apply_transfer_preds(slot_values, preds["transfer_in"],
                                   self.prob_thresh["in_domain_slot_prediction"])
        self._apply_transfer_preds(slot_values, preds["transfer_cross"],
                                   self.prob_thresh["cross_domain_slot_prediction"])

        # ---------- 7. rule‑2 filter: keep only required ∪ optional ------
        legal = set(self._intent_slot_role)
        slot_values = {s: v for s, v in slot_values.items() if s in legal}

        return {
            "active_intent":  active_intent if active_intent else "NONE",
            "requested_slots": list(requested_slots & legal),
            "slot_values":     slot_values,
        }

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _update_intent_slot_roles(self, intent_name: str):
        self._intent_slot_role = {}
        for it in self.schema["intents"]:
            if it["name"] != intent_name:
                continue
            for s in it["required_slots"]:
                self._intent_slot_role[s] = "required"
            for s in it["optional_slots"]:
                self._intent_slot_role[s] = "optional"
            break

    def _apply_slot_value_preds(self, slot_values, preds, thr):
        """
        preds : list[{slot, value, prob}]
        keep highest-prob per slot above threshold
        """
        best = defaultdict(lambda: (None, -1.0))
        for p in preds:
            if p["prob"] >= thr and p["prob"] > best[p["slot"]][1]:
                best[p["slot"]] = (p["value"], p["prob"])
        for slot, (val, _) in best.items():
            slot_values[slot] = [val]

    def _apply_transfer_preds(self, slot_values, preds, thr):
        """
        preds : list[{slot, value, prob}]
        copy the value if above threshold and the slot currently empty
        """
        for p in preds:
            if p["prob"] >= thr and not slot_values.get(p["slot"]):
                slot_values[p["slot"]] = [p["value"]]
