import glob, json, os
import torch
from typing import Dict, List, Tuple, Any
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from main.train.utils import log, ModuleType
from main.data_model.config import DataConfig, ModelConfig
from main.data_model.initialize import PathConfig


class DataHandler:
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, path_config: PathConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.path_config = path_config
        self.tok = AutoTokenizer.from_pretrained(model_config.model_name, use_fast=True)
        self.schemas = self._load_schema_map(module=ModuleType.CAT_SLOT)

    def prepare_data(self, splits: List[Tuple[str, str]] = [("train", "train"), ("dev", "validation")],
                     module: ModuleType = ModuleType.INTENT, is_train: bool = True,
                     test_split_tag: str = "test"):
        out_dir = self.path_config.PREPARED_DATA_DIR
        data_dir_prefix = self.data_config.data_dir_prefix[module]
        
        if is_train:
            log("Data preparation for Training")
            log("-" * 100)
            if os.path.isdir(os.path.join(str(out_dir), data_dir_prefix.format("train"))):
                log("Data preparation has already been done. Exiting!")
            else:
                log("Create data_dct for all splits")
                ds = {}
                for raw_split, save_split in splits:
                    ds[save_split] = self._load_sgd_flat(raw_split, module)
                ds = DatasetDict(ds)
                
                for save_split in ds:
                    if module == ModuleType.INTENT:
                        packed = ds[save_split].map(self._pack_intents, remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.CAT_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_cat_slot(ex),
                                                    remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.FREE_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_free_slot(ex),
                                                    remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.REQUESTED_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_req_slot(ex),
                                                    remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.IN_DOMAIN_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_xfer_in(ex),
                                                    remove_columns=ds[save_split].column_names)
                    else:
                        packed = ds[save_split].map(lambda ex: self._pack_xfer_cross(ex),
                                                    remove_columns=ds[save_split].column_names)
                    packed.save_to_disk(out_dir / data_dir_prefix.format(save_split))
                    log(f"Saved {len(packed)} packed {save_split} examples")
        else:
            log("Data preparation for Inference")
            log("-" * 100)
            if os.path.isdir(os.path.join(str(out_dir), data_dir_prefix.format(test_split_tag))):
                log("Data preparation has already been done. Exiting!")
            else:
                ds = {}
                ds[test_split_tag] = self._load_sgd_flat(test_split_tag, module)
                ds = DatasetDict(ds)

                for save_split in ds:
                    if module == ModuleType.INTENT:
                        packed = ds[save_split].map(self._pack_intents, remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.CAT_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_cat_slot(ex),
                                                    remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.FREE_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_free_slot(ex),
                                                    remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.REQUESTED_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_req_slot(ex),
                                                    remove_columns=ds[save_split].column_names)
                    elif module == ModuleType.IN_DOMAIN_SLOT:
                        packed = ds[save_split].map(lambda ex: self._pack_xfer_in(ex),
                                                    remove_columns=ds[save_split].column_names)
                    else:
                        packed = ds[save_split].map(lambda ex: self._pack_xfer_cross(ex),
                                                    remove_columns=ds[save_split].column_names)
                    packed.save_to_disk(out_dir / data_dir_prefix.format(save_split))
                    log(f"Saved {len(packed)} packed {save_split} examples")

    def _load_sgd_flat(self, split: str, module: ModuleType) -> Dataset:
        """Return a Dataset of flattened USER turns for the given split."""
        turns = []
        for dlg in self._iter_dialogues(split):
            if module == ModuleType.INTENT:
                turns.extend(self._build_intent_examples(dlg))
            elif module == ModuleType.CAT_SLOT:
                turns.extend(self._build_cat_slot_examples(dlg))
            elif module == ModuleType.FREE_SLOT:
                turns.extend(self._build_free_slot_examples(dlg))
            elif module == ModuleType.REQUESTED_SLOT:
                turns.extend(self._build_req_slot_examples(dlg))
            elif module == ModuleType.IN_DOMAIN_SLOT:
                turns.extend(self._build_xfer_in_examples(dlg))
            else:
                turns.extend(self._build_xfer_cross_examples(dlg))
        log(f"Loaded {len(turns)} USER turns for {split}")
        return Dataset.from_list(turns)

    def _iter_dialogues(self, split: str):
        base_dir = self.path_config.RAW_DATA_DIR
        for fp in glob.glob(f"{base_dir}/{split}/*.json"):
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for dlg in data:
                        yield dlg
                else:
                    yield data

    def _load_schema_map(self, module: ModuleType):
        """Map service_name → schema dict."""
        base_dir = self.path_config.RAW_DATA_DIR
        out = {}
        for fp in glob.glob(f"{base_dir}/*/schema.json"):
            with open(fp, encoding="utf-8") as f:
                for svc in json.load(f):
                    if module == ModuleType.INTENT:
                        out[svc["service_name"]] = svc["intents"]
                    else:
                        out[svc["service_name"]] = svc
        return out

    @staticmethod
    def _offered_from_actions(actions: List[Dict[str, Any]]):
        out = {}
        for a in actions or []:
            if a.get("act") == "OFFER" and a.get("slot") and a.get("values"):
                out[a["slot"]] = a["values"][0]
        return out
    
    def _build_intent_examples(self, dlg: Dict):
        """Yield flattened USER-turn records; skip objects without 'turns'."""
        if "turns" not in dlg:
            return []

        dlg_id = dlg["dialogue_id"]
        history, prev_intent = [], "NONE"
        for turn_idx, t in enumerate(dlg["turns"]):
            spk, utt = t.get("speaker", ""), t.get("utterance", "")
            if spk == "USER":
                sys_utt = next((u for s, u in reversed(history) if s == "SYSTEM"), "")
                if not t.get("frames"):
                    history.append((spk, utt))
                    continue
                frame = t["frames"][0] # considering only the first frame
                state = frame.get("state", {})
                yield {
                    "dlg_id":   dlg_id,
                    "turn_idx": turn_idx,
                    "service": frame.get("service", ""),
                    "sys_utt": sys_utt,
                    "user_utt": utt,
                    "prev_intent": prev_intent,
                    "active_intent": state.get("active_intent", "NONE"),
                }
                prev_intent = state.get("active_intent", "NONE")
            history.append((spk, utt))
    
    def _build_cat_slot_examples(self, dlg: Dict[str, Any]):
        if "turns" not in dlg:
            return []

        dlg_id = dlg["dialogue_id"]
        examples = []
        last_sys_actions = []
        for turn_idx, turn in enumerate(dlg["turns"]):
            speaker = turn["speaker"]

            # ---------- SYSTEM turn: remember its actions for the next user turn
            if speaker == "SYSTEM":
                if turn.get("frames"):
                    last_sys_actions = turn["frames"][0].get("actions", [])
                continue

            # ---------- USER turn ----------
            if speaker != "USER":
                continue

            # build utterance pair (last SYSTEM + current USER)
            sys_ctx = (last_sys_actions and
                       turn["utterance"] != dlg["turns"][0]["utterance"] and
                       dlg["turns"][dlg["turns"].index(turn)-1]["utterance"]) or ""
            usr_ctx = turn["utterance"]
            utter   = f"sys: {sys_ctx} usr: {usr_ctx}".strip()

            for fr in turn.get("frames", []):
                svc = fr["service"]
                schema = self.schemas.get(svc)
                if not schema or "state" not in fr:
                    continue

                requested = set(fr["state"]["requested_slots"])
                offered   = DataHandler._offered_from_actions(last_sys_actions)
                slot_vals = fr["state"]["slot_values"]

                for sl in schema["slots"]:
                    if not sl["is_categorical"]:
                        continue
                    slot      = sl["name"]
                    values    = ["null"] + sl["possible_values"]
                    gold_val  = slot_vals.get(slot, ["null"])[0]
                    label_idx = values.index(gold_val) if gold_val in values else 0
                    ctx_feat  = 2 * int(slot in requested) + int(slot in offered)

                    examples.append({
                        "dlg_id":   dlg_id,
                        "turn_idx": turn_idx,
                        "utterance": utter,
                        "slot_desc": f"{slot}: {sl['description']}",
                        "values":    values,
                        "label":     label_idx,
                        "ctx_feat":  ctx_feat,
                    })

            last_sys_actions = [] # reset until we hit the next SYSTEM turn

        return examples
    
    def _build_free_slot_examples(self, dlg: Dict[str, Any]):
        """Yield one example per free-form slot in each USER turn."""
        if "turns" not in dlg:
            return []

        dlg_id = dlg["dialogue_id"]
        examples, last_sys_actions = [], []
        last_sys_utt = ""
        for turn_idx, turn in enumerate(dlg["turns"]):
            spk = turn["speaker"]

            # remember most recent SYSTEM actions
            if spk == "SYSTEM":
                last_sys_actions = turn.get("frames", [{}])[0].get("actions", [])
                last_sys_utt = turn["utterance"]
                continue
            
            if spk != "USER":
                continue

            sys_ctx = last_sys_utt
            usr_ctx = turn["utterance"]
            utter   = f"sys: {sys_ctx} usr: {usr_ctx}".strip()
            utter_ids = self.tok(utter, add_special_tokens=False,
                                 return_offsets_mapping=True)
            offsets   = utter_ids["offset_mapping"]

            for fr in turn.get("frames", []):
                svc = fr["service"]
                schema = self.schemas.get(svc)
                if not schema or "state" not in fr:
                    continue
                requested = set(fr["state"]["requested_slots"])
                offered   = DataHandler._offered_from_actions(last_sys_actions)

                gold_spans = {s["slot"]: (s["start"], s["exclusive_end"])
                            for s in fr.get("slots", [])}

                for sl in schema["slots"]:
                    if sl["is_categorical"]:
                        continue
                    slot = sl["name"]
                    ctx_feat = 2 * int(slot in requested) + int(slot in offered)

                    # ----- gold start/end token indices --------------------------
                    if slot in gold_spans:
                        char_s, char_e = gold_spans[slot]
                        # find token indices whose char spans overlap
                        tok_start, tok_end = None, None
                        for idx, (cs, ce) in enumerate(offsets):
                            if cs <= char_s < ce:
                                tok_start = idx
                            if cs < char_e <= ce:
                                tok_end = idx
                                break
                        if tok_start is None or tok_end is None:
                            tok_start = tok_end = len(utter_ids["input_ids"])  # null
                    else:
                        tok_start = tok_end = len(utter_ids["input_ids"])      # null

                    examples.append({
                        "dlg_id":   dlg_id,
                        "turn_idx": turn_idx,
                        "utterance": utter,
                        "slot_desc": f"{slot}: {sl['description']}",
                        "ctx_feat":  ctx_feat,
                        "start":     tok_start,
                        "end":       tok_end,
                    })
        return examples
    
    def _build_req_slot_examples(self, dlg: Dict[str, Any]):
        """One example per slot in each USER turn (label=1 if slot is requested)."""
        if "turns" not in dlg:
            return []

        dlg_id = dlg["dialogue_id"]
        examples, last_sys_actions, last_sys_utt = [], [], ""
        for turn_idx, turn in enumerate(dlg.get("turns", [])):
            spk = turn["speaker"]

            # remember system context
            if spk == "SYSTEM":
                last_sys_actions = turn.get("frames", [{}])[0].get("actions", [])
                last_sys_utt = turn["utterance"]
                continue
            
            if spk != "USER":
                continue

            sys_ctx = last_sys_utt
            usr_ctx = turn["utterance"]
            utter   = f"sys: {sys_ctx} usr: {usr_ctx}".strip()

            for fr in turn.get("frames", []):
                svc = fr["service"]
                schema = self.schemas.get(svc)
                if not schema or "state" not in fr:
                    continue

                requested = set(fr["state"]["requested_slots"])
                offered   = DataHandler._offered_from_actions(last_sys_actions)

                for sl in schema["slots"]:
                    slot = sl["name"]
                    ctx_feat = 2 * int(slot in requested) + int(slot in offered)
                    examples.append({
                        "dlg_id":   dlg_id,
                        "turn_idx": turn_idx,
                        "utterance": utter,
                        "slot_desc": f"{slot}: {sl['description']}",
                        "ctx_feat":  ctx_feat,
                        "label":     int(slot in requested),
                    })
        return examples
    
    def _build_xfer_in_examples(self, dlg: Dict[str, Any]):
        """
        In-domain slot-transfer examples per USER turn.
        label = 1  <=>  (slot offered earlier OR in previous state) AND not overwritten now
        """
        if "dialogue_id" not in dlg:
            return []

        dlg_id = dlg["dialogue_id"]
        examples = []
        last_sys_utt   = ""
        last_offered   = {}          # slot -> latest offered value
        prev_user_state = {}         # service -> slot_values dict

        for turn_idx, turn in enumerate(dlg.get("turns", [])):
            spk = turn["speaker"]

            # ----- SYSTEM turn: update memory of offers & system utterance -------
            if spk == "SYSTEM":
                last_sys_utt = turn["utterance"]
                for fr in turn.get("frames", []):
                    for act in fr.get("actions", []):
                        if act.get("act") == "OFFER" and act.get("slot"):
                            last_offered[act["slot"]] = act.get("values", [None])[0]
                continue

            # ----- skip if not USER ---------------------------------------------
            if spk != "USER":
                continue

            sys_ctx = last_sys_utt
            usr_ctx = turn["utterance"]
            utter   = f"sys: {sys_ctx} usr: {usr_ctx}".strip()

            for fr in turn.get("frames", []):
                svc   = fr["service"]
                schema = self.schemas.get(svc)
                state  = fr.get("state", {})
                if not schema or not state:
                    continue

                # intent requirement flags
                act_name = state.get("active_intent", "NONE")
                intent   = next((i for i in schema["intents"] if i["name"] == act_name), None)

                if intent:
                    required = set(intent["required_slots"])
                    optional = set(intent["optional_slots"])
                else:
                    required = optional = set()

                cur_vals = state["slot_values"]
                overwritten = set(k for k, v in cur_vals.items() if v)  # user filled now
                hist_vals   = prev_user_state.get(svc, {})

                for sl in schema["slots"]:
                    slot = sl["name"]

                    ctx_id = (8 * int(slot in required) +
                              4 * int(slot in optional) +
                              2 * int(slot in last_offered) +
                              1 * int(slot in hist_vals))

                    cond_offer = (slot in last_offered)            # offered earlier
                    cond_hist  = (slot in hist_vals)               # in prev state
                    not_over   = (slot not in overwritten)         # untouched now
                    label = int((cond_offer or cond_hist) and not_over)

                    examples.append({
                        "dlg_id":   dlg_id,
                        "turn_idx": turn_idx,
                        "utterance":   utter,
                        "service_desc": schema["description"],
                        "slot_desc":   f"{slot}: {sl['description']}",
                        "ctx_feat":    ctx_id,
                        "labels":      label,
                    })

                # update running user state for next turn
                prev_user_state[svc] = cur_vals

        return examples
    
    def _build_xfer_cross_examples(self, dlg: Dict[str, Any]):
        if "dialogue_id" not in dlg:
            return []

        dlg_id = dlg["dialogue_id"]
        examples = []
        svc_state_hist = {}     # service -> latest slot_values
        prev_turn_svcs = set()  # services that appeared in *any* previous turn

        for turn_idx, turn in enumerate(dlg.get("turns", [])):
            cur_svcs = {fr["service"] for fr in turn.get("frames", [])}

            # --------------------------------------------------------------------
            # USER turn -> create training examples
            # --------------------------------------------------------------------
            if turn["speaker"] == "USER":
                utter = turn["utterance"]
                source_slots = {
                    (svc, slot): vals[0]
                    for svc, slots in svc_state_hist.items()
                    for slot, vals in slots.items() if vals
                }

                for fr in turn["frames"]:
                    tgt_svc = fr["service"]
                    schema  = self.schemas.get(tgt_svc)
                    if not schema or "state" not in fr:
                        continue

                    cont = int(tgt_svc in prev_turn_svcs)
                    state = fr["state"]
                    intent = next((i for i in schema["intents"]
                                if i["name"] == state["active_intent"]), None)
                    required = set(intent["required_slots"]) if intent else set()
                    optional = set(intent["optional_slots"]) if intent else set()
                    tgt_hist = svc_state_hist.get(tgt_svc, {})

                    for sl in schema["slots"]:
                        slot_tgt = sl["name"]
                        tgt_desc = f"{slot_tgt}: {sl['description']}"
                        req_bit  = int(slot_tgt in required)
                        opt_bit  = int(slot_tgt in optional)
                        t_hist   = int(slot_tgt in tgt_hist)

                        for (src_svc, slot_src), src_val in source_slots.items():
                            if src_svc == tgt_svc:
                                continue  # cross‑domain only

                            src_schema = self.schemas.get(src_svc)
                            src_desc_txt = next((s['description'] for s in src_schema['slots']
                                                if s['name'] == slot_src), slot_src)
                            src_desc = f"{slot_src}: {src_desc_txt}"

                            ctx_id = 16 * cont + 8 * req_bit + 4 * opt_bit + 2 * t_hist + 1

                            has_now = bool(state['slot_values'].get(slot_tgt))
                            label = int((not has_now) and src_val not in ('', None))

                            examples.append({
                                'dlg_id':   dlg_id,
                                'turn_idx': turn_idx,
                                'utterance':   utter,
                                'target_slot': tgt_desc,
                                'source_slot': src_desc,
                                'ctx_feat':    ctx_id,
                                'labels':      label,
                            })

                    # update running state for this target service
                    svc_state_hist[tgt_svc] = state['slot_values']

            # --------------------------------------------------------------------
            # after processing (SYSTEM or USER), remember current services
            # --------------------------------------------------------------------
            prev_turn_svcs = cur_svcs

        return examples
    
    def _pack_intents(self, ex):
        schema = self._load_schema_map(module=ModuleType.INTENT)
        intents = schema[ex["service"]]
        utter = f"sys: {ex['sys_utt']} usr: {ex['user_utt']}"
        pieces, seg, ctx, iid = [self.data_config.special_tokens['cls']], [0], [0], [-1]
        
        # utterance tokens
        u_toks = self.tok.tokenize(utter)
        pieces += u_toks
        seg += [0] * len(u_toks)
        ctx += [0] * len(u_toks)
        iid += [-1] * len(u_toks)
        pieces += [self.data_config.special_tokens['sep']]
        seg += [0]
        ctx += [0]
        iid += [-1]
        label_idx = -1
        for j, it in enumerate(intents):
            desc = self.tok.tokenize(it["description"])
            flag = 1 if it["name"] == ex["prev_intent"] else 0
            pieces += desc
            seg += [1] * len(desc)
            ctx += [flag] * len(desc)
            iid += [j] * len(desc)
            pieces += [self.data_config.special_tokens['sep']]
            seg += [1]
            ctx += [flag]
            iid += [-1]
            if it["name"] == ex["active_intent"]:
                label_idx = j
        
        # truncate/pad
        pieces = pieces[:self.model_config.max_seq_len]
        seg = seg[:self.model_config.max_seq_len]
        ctx = ctx[:self.model_config.max_seq_len]
        iid = iid[:self.model_config.max_seq_len]
        input_ids = self.tok.convert_tokens_to_ids(pieces)
        attn_mask = [1]*len(input_ids)
        pad_len = self.model_config.max_seq_len - len(input_ids)
        input_ids += [self.tok.pad_token_id] * pad_len
        seg += [0] * pad_len
        ctx += [0] * pad_len
        iid += [-1] * pad_len
        attn_mask += [0] * pad_len
        intent_mask = [1] * len(intents) + [0] * (32 - len(intents))  # 32 = max intents per service in SGD
        
        return {
            "dlg_id": ex["dlg_id"],
            "turn_idx": ex["turn_idx"],
            "input_ids": input_ids,
            "token_type_ids": seg,
            "context_ids": ctx,
            "intent_ids": iid,
            "attention_mask": attn_mask,
            "intent_mask": intent_mask,
            "label_id": label_idx,
        }
    
    def _pack_cat_slot(self, ex):
        ids_utt  = self.tok(ex["utterance"], add_special_tokens=False)["input_ids"]
        ids_slot = self.tok(ex["slot_desc"], add_special_tokens=False)["input_ids"]
        val_spans, val_ids = [], []
        for v in ex["values"][:self.data_config.max_num_values_cat_slot]:
            start = 1 + len(ids_utt) + 1 + len(ids_slot) + 1 + len(val_ids)
            ids_v = self.tok(v, add_special_tokens=False)["input_ids"]
            val_spans.append((start, start + len(ids_v)))
            val_ids += ids_v + [self.tok.sep_token_id]
        while len(val_spans) < self.data_config.max_num_values_cat_slot:
            val_spans.append((-1, -1))
            val_ids.append(self.tok.sep_token_id)
        seq = [self.tok.cls_token_id] + ids_utt + [self.tok.sep_token_id] \
            + ids_slot + [self.tok.sep_token_id] + val_ids
        seq = seq[:self.model_config.max_seq_len]
        seg = [0] * (len(ids_utt) + len(ids_slot) + 3) + [1] * (len(seq) - len(ids_utt) - len(ids_slot) - 3)
        ctx = [ex["ctx_feat"]] * len(seq)

        # restricted attention mask
        L = len(seq)
        att = torch.zeros((L, L), dtype=torch.long)
        utt_sl  = slice(1, 1 + len(ids_utt))
        slot_sl = slice(2 + len(ids_utt), 2 + len(ids_utt) + len(ids_slot))
        att[utt_sl, :]  = 1
        att[slot_sl, :] = 1
        for s, e in val_spans:
            if s == -1:
                continue
            att[s: e, utt_sl] = 1
            att[s: e, slot_sl] = 1
            att[s: e, s: e] = 1

        return {
            "dlg_id": ex["dlg_id"],
            "turn_idx": ex["turn_idx"],
            "input_ids": seq,
            "token_type_ids": seg,
            "context_ids": ctx,
            "attention_mask": att,
            "val_spans": val_spans,
            "labels": ex["label"]
        }

    def _pack_free_slot(self, ex):
        ids_utt  = self.tok(ex["utterance"], add_special_tokens=False)["input_ids"]
        ids_utt += self.tok("null", add_special_tokens=False)["input_ids"]  # concat null
        null_idx = len(ids_utt) - 1

        ids_slot = self.tok(ex["slot_desc"], add_special_tokens=False)["input_ids"]

        seq = [self.tok.cls_token_id] + ids_utt + [self.tok.sep_token_id] \
            + ids_slot + [self.tok.sep_token_id]
        seq = seq[:self.model_config.max_seq_len]

        seg = [0] * (len(ids_utt) + 2) + [1] * (len(seq) - len(ids_utt) - 2)
        ctx = [ex["ctx_feat"]] * len(seq)

        # start/end label positions (cap at max_len‑1)
        start = min(1 + ex["start"], self.model_config.max_seq_len - 1) if ex["start"] != -1 else null_idx + 1
        end   = min(1 + ex["end"],   self.model_config.max_seq_len - 1) if ex["end"] != -1 else null_idx + 1

        return {
            "dlg_id": ex["dlg_id"],
            "turn_idx": ex["turn_idx"],
            "input_ids": seq,
            "token_type_ids": seg,
            "context_ids": ctx,
            "attention_mask": [1] * len(seq),
            "start_positions": start,
            "end_positions":   end
        }

    def _pack_req_slot(self, ex):
        ids_utt  = self.tok(ex["utterance"], add_special_tokens=False)["input_ids"]
        ids_utt += self.tok("null", add_special_tokens=False)["input_ids"]
        ids_slot = self.tok(ex["slot_desc"], add_special_tokens=False)["input_ids"]

        seq = [self.tok.cls_token_id] + ids_utt + [self.tok.sep_token_id] \
            + ids_slot + [self.tok.sep_token_id]
        seq = seq[:self.model_config.max_seq_len]

        seg = [0] * (len(ids_utt) + 2) + [1] * (len(seq) - len(ids_utt) - 2)
        ctx = [ex["ctx_feat"]] * len(seq)

        return {
            "dlg_id": ex["dlg_id"],
            "turn_idx": ex["turn_idx"],
            "input_ids": seq,
            "token_type_ids": seg,
            "context_ids": ctx,
            "attention_mask": [1] * len(seq),
            "labels": ex["label"]
        }
    
    def _pack_xfer_in(self, ex):
        ids_srv  = self.tok(ex["service_desc"], add_special_tokens=False)["input_ids"]
        ids_utt  = self.tok(ex["utterance"], add_special_tokens=False)["input_ids"]
        ids_slot = self.tok(ex["slot_desc"], add_special_tokens=False)["input_ids"]

        seq = [self.tok.cls_token_id] + ids_srv + [self.tok.sep_token_id] \
            + ids_utt + [self.tok.sep_token_id] + ids_slot + [self.tok.sep_token_id]
        seq = seq[:self.model_config.max_seq_len]

        # segment 0 = [CLS]+service+[SEP]+utterance+[SEP] ; segment 1 = slot+[SEP]
        seg = [0] * (len(ids_srv) + len(ids_utt) + 3) + [1] * (len(ids_slot) + 1)
        ctx = [ex["ctx_feat"]] * len(seq)

        return {
            "dlg_id": ex["dlg_id"],
            "turn_idx": ex["turn_idx"],
            "input_ids": seq,
            "token_type_ids": seg,
            "context_ids": ctx,
            "attention_mask": [1] * len(seq),
            "labels": ex["labels"]
        }

    def _pack_xfer_cross(self, ex):
        ids_utt  = self.tok(ex["utterance"], add_special_tokens=False)["input_ids"]
        ids_tgt  = self.tok(ex["target_slot"], add_special_tokens=False)["input_ids"]
        ids_src  = self.tok(ex["source_slot"], add_special_tokens=False)["input_ids"]

        seq = [self.tok.cls_token_id] + ids_utt + [self.tok.sep_token_id] \
            + ids_tgt + [self.tok.sep_token_id] + ids_src + [self.tok.sep_token_id]
        seq = seq[:self.model_config.max_seq_len]

        seg = [0] * (len(ids_utt) + len(ids_tgt) + 3) + [1] * (len(seq) - len(ids_utt) - len(ids_tgt) - 3)
        ctx = [ex["ctx_feat"]] * len(seq)

        return {
            "dlg_id": ex["dlg_id"],
            "turn_idx": ex["turn_idx"],
            "input_ids": seq,
            "token_type_ids": seg,
            "context_ids": ctx,
            "attention_mask": [1] * len(seq),
            "labels": ex["labels"]
        }
