import torch
import time
import os
import json
import pickle
import glob
import numpy as np
from typing import Dict, Any
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk
from safetensors.torch import load_file
from main.train.model_handler import IntentHead, CatSlotHead, FreeSlotHead, ReqSlotHead, InXferHead, CrossXferHead, build_tokenizer
from main.train.data_collator import collate_intent, XferCollator, CatCollator
from main.data_model.config import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
from main.data_model.initialize import PathConfig
from main.train.utils import log, ModuleType
from main.infer.user_state_summarizer import UserStateSummariser


def load_model(pytorch_bin_fn: str, model_config: ModelConfig, module: ModuleType, map_location="cpu"):
    if module == ModuleType.INTENT:
        model_dir = pytorch_bin_fn.split("model.safetensors")[0]
    else:
        model_dir = pytorch_bin_fn.split("pytorch_model.bin")[0]
    tok = AutoTokenizer.from_pretrained(model_dir)
    if module == ModuleType.INTENT:
        model = IntentHead(tok, model_config)
        collator = collate_intent
    elif module == ModuleType.CAT_SLOT:
        model = CatSlotHead(model_config=model_config)
        collator = CatCollator(pad_id=tok.pad_token_id)
    elif module == ModuleType.FREE_SLOT:
        model = FreeSlotHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id, module=module)
    elif module == ModuleType.REQUESTED_SLOT:
        model = ReqSlotHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id)
    elif module == ModuleType.IN_DOMAIN_SLOT:
        model = InXferHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id)
    elif module == ModuleType.CROSS_DOMAIN_SLOT:
        model = CrossXferHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id)
    
    if module != ModuleType.INTENT:
        state_dict = torch.load(pytorch_bin_fn, map_location=map_location)
    else:
        state_dict = load_file(pytorch_bin_fn)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(map_location)
    return model, collator


def prediction_helper(module: str, data_config: DataConfig, model_config: ModelConfig,
                      training_config: TrainingConfig, path_config: PathConfig, device: str,
                      n_steps: int = 150, sampling: bool = False, n_samples: int = 1000):
    start = time.time()

    log("Load Test dataset")
    start_module = time.time()
    data_dir = path_config.PREPARED_DATA_DIR
    data_dir_prefix = data_config.data_dir_prefix[module]
    prediction_dir = path_config.PREDICTION_DIR
    test = load_from_disk(data_dir / data_dir_prefix.format("test"))
    if sampling:
        test = test.shuffle(seed=training_config.seed).take(n_samples)
    log(f"Time taken for loading dataset containing {len(test)} examples: {time.time() - start_module}")
    log("-" * 100)

    log("Load model")
    start_module = time.time()
    model_dir = os.path.join(str(path_config.RAW_DATA_DIR), training_config.output_dir[module], f"checkpoint-{n_steps}")
    if module != ModuleType.INTENT:
        model_dir = os.path.join(model_dir, "pytorch_model.bin")
    else:
        model_dir = os.path.join(model_dir, "model.safetensors")
    model, collator = load_model(model_dir, model_config, module, device)
    trainer = Trainer(model=model, data_collator=collator,
                      args=TrainingArguments("tmp", per_device_eval_batch_size=training_config.per_device_eval_batch_size))
    log(f"Time taken for loading model: {time.time() - start_module}")
    log("-" * 100)
    
    log("Prediction begins...")
    start_module = time.time()
    preds = trainer.predict(test).predictions
    log(f"Time taken for getting predictions: {time.time() - start_module}")
    log("-" * 100)

    log("Format predictions")
    look = {}
    for i, ex in enumerate(test):
        dlg_id = ex["dlg_id"]
        turn_id = ex["turn_idx"]
        key = (dlg_id, turn_id)
        if isinstance(preds, tuple):
            row_pred = tuple(p[i] for p in preds)
        else:
            row_pred = preds[i]
        if str(key) not in look:
            look[str(key)] = []
        look[str(key)].append((ex, row_pred))
    
    log("Save predictions")
    start_module = time.time()
    with open(os.path.join(prediction_dir, f"{module}_{n_steps}.pkl"), "wb") as f:
        pickle.dump(look, f)
    log(f"Time taken for saving predictions: {time.time() - start_module}")
    log("-" * 100)

    log(f"Total time taken: {time.time() - start}")


def run_inference(n_steps: int, schema: Dict[str, Any], path_config: PathConfig,
                  model_config: ModelConfig, test_split_tag: str,
                  inference_config: InferenceConfig):
    log("Read the saved predictions")
    look = {}
    for module in ModuleType:
        with open(os.path.join(path_config.PREDICTION_DIR, f"{module.value}_{n_steps}.pkl"), "rb") as f:
            look[module.value] = pickle.load(f)

    log("Load tokenizer")
    tok = build_tokenizer(model_config)

    out_dir = path_config.DLG_LEVEL_PREDICTION_DIR
    log("Iterate over raw test dialogues")
    for fp in glob.glob(str(path_config.RAW_DATA_DIR / f"{test_split_tag}/*.json")):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for dlg in data:
            if "dialogue_id" in dlg:
                dlg_id = dlg["dialogue_id"]
                pred_turns = []
                last_sys_actions = []
                svc_states = {}
                
                for t_idx, turn in enumerate(dlg.get("turns", [])):
                    if turn["speaker"] != "USER":
                        pred_turns.append(turn)
                        if turn.get("frames"):
                            last_sys_actions = turn["frames"][0].get("actions", [])
                        continue

                    key = (dlg_id, t_idx)
                    frames_out = []
                    for fr in turn["frames"]:
                        svc  = fr["service"]
                        sch = schema[svc]

                        # -------- intent logits --------------------------------------
                        if key in look["intent_prediction"]:
                            intent_logits   = look["intent_prediction"][key][0][1] # one per turn
                            intent_id = int(np.argmax(intent_logits))
                            intent_prob = float(torch.max(torch.softmax(torch.tensor(intent_logits), -1)))
                            intent_pred = sch["intents"][intent_id]["name"]
                        else:
                            intent_pred = "N/A"
                            intent_prob = 0.0
                        # -------- categorical slot ----------------------------------
                        cat_preds = []
                        for ex, item in look["categorical_slot_prediction"].get(key, []):
                            prob = float(torch.softmax(torch.tensor(item), -1).max())
                            if prob >= inference_config.prob_threshold["categorical_slot_prediction"] and ex["labels"] != 0:
                                cat_preds.append({"slot": ex["slot_desc"].split(':')[0].strip(),
                                                "value": ex["values"][ex["labels"]],
                                                "prob": prob})

                        # -------- free‑form, requested, transfers -------------------
                        free_preds = []
                        for ex, (start_vec, end_vec) in look["free_form_slot_prediction"].get(key, []):
                            start_idx = int(np.argmax(start_vec))
                            end_idx   = int(np.argmax(end_vec))
                            p_start   = torch.softmax(torch.tensor(start_vec), -1)[start_idx].item()
                            p_end     = torch.softmax(torch.tensor(end_vec),   -1)[end_idx].item()
                            prob      = (p_start * p_end) ** 0.5

                            if start_idx <= end_idx and prob >= inference_config.prob_threshold["free_form_slot_prediction"]:
                                span_ids = ex["input_ids"][start_idx: end_idx + 1]
                                text = tok.decode(span_ids).strip()
                                free_preds.append({"slot": ex["slot_desc"].split(':')[0].strip(),
                                                "value": text,
                                                "prob":  prob})
                        
                        # -------- requested slot ----------------------------------
                        req_preds = []
                        for ex, item in look["requested_slot_prediction"].get(key, []):
                            prob = float(torch.softmax(torch.tensor(item), -1).max())
                            if prob >= inference_config.prob_threshold["requested_slot_prediction"] and ex["labels"] != 0:
                                req_preds.append({"slot": ex["slot_desc"].split(':')[0].strip(),
                                                "prob": prob})
                                
                        # -------- in‑domain transfer ----------------------------------
                        tin_preds = []
                        for ex, item in look["in_domain_slot_prediction"].get(key, []):
                            prob = float(torch.softmax(torch.tensor(item), -1).max())
                            if prob >= inference_config.prob_threshold["in_domain_slot_prediction"] and ex["labels"] != 0:
                                tin_preds.append({"slot": ex["slot_desc"].split(':')[0].strip(),
                                                "prob": prob})
                                
                        # -------- cross‑domain transfer ----------------------------------
                        tcross_preds = []
                        for ex, item in look["cross_domain_slot_prediction"].get(key, []):
                            prob = float(torch.softmax(torch.tensor(item), -1).max())
                            if prob >= inference_config.prob_threshold["cross_domain_slot_prediction"] and ex["labels"] != 0:
                                tcross_preds.append({"source_slot": ex["source_slot"].split(':')[0].strip(),
                                                    "target_slot": ex["target_slot"].split(':')[0].strip(),
                                                    "prob": prob})

                        preds_all = {"intent": {"pred": intent_pred, "prob": intent_prob},
                                     "categorical": cat_preds,
                                     "free_form": free_preds,
                                     "requested_slot": req_preds,
                                     "transfer_in": tin_preds,
                                     "transfer_cross": tcross_preds}

                        summariser = UserStateSummariser(sch, inference_config)
                        new_state = summariser.update(svc_states.get(svc, {}), preds_all)
                        svc_states[svc] = new_state
                        frames_out.append({"service": svc, "state": new_state})

                    pred_turns.append({"speaker": "USER",
                                       "utterance": turn["utterance"],
                                       "frames": frames_out})
                json.dump({"dialogue_id": dlg_id, "turns": pred_turns}, open(out_dir / f"{dlg_id}.json", "w"), indent=2)
