import torch
from main.train.utils import ModuleType


def collate_intent(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class XferCollator:
    def __init__(self, pad_id: int = 0, module: ModuleType = ModuleType.IN_DOMAIN_SLOT):
        self.pad_id = pad_id
        self.module = module
    
    def __call__(self, feats):
        max_len = max(len(f["input_ids"]) for f in feats)
        for f in feats:
            pad = max_len - len(f["input_ids"])
            for k in ("input_ids", "token_type_ids", "context_ids", "attention_mask"):
                f[k] += [0 if k == "attention_mask" else self.pad_id] * pad
        if self.module != ModuleType.FREE_SLOT:
            batch = {k: torch.tensor([f[k] for f in feats]) if k != "attention_mask" else
                    torch.tensor([f[k] for f in feats], dtype=torch.bool)
                    for k in ("input_ids", "token_type_ids", "context_ids",
                            "attention_mask", "labels")}
        else:
            batch = {k: torch.tensor([f[k] for f in feats]) if k != "attention_mask" else
                 torch.tensor([f[k] for f in feats], dtype=torch.bool)
                 for k in ("input_ids", "token_type_ids", "context_ids",
                           "attention_mask", "start_positions", "end_positions")}
        return batch


class CatCollator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, feats):
        max_len = max(len(f["input_ids"]) for f in feats)

        padded_masks = []
        for f in feats:
            pad = max_len - len(f["input_ids"])

            # -------- pad 1‑D fields -----------------------------------------
            for key in ("input_ids", "token_type_ids", "context_ids"):
                f[key] += [self.pad_id] * pad

            # -------- attention mask (convert list → tensor, then pad) -------
            att = torch.tensor(f["attention_mask"], dtype=torch.long)  # (L,L)
            L = att.size(0)

            # pad rows
            if pad:
                pad_row = torch.zeros((pad, L), dtype=torch.long)
                att = torch.cat([att, pad_row], dim=0)                 # (L+pad, L)
                pad_col = torch.zeros((L+pad, pad), dtype=torch.long)
                att = torch.cat([att, pad_col], dim=1)                 # (L+pad, L+pad)

            padded_masks.append(att)

        # -------- stack tensors into batch -----------------------------------
        batch = {
            "input_ids":      torch.tensor([f["input_ids"]      for f in feats]),
            "token_type_ids": torch.tensor([f["token_type_ids"] for f in feats]),
            "context_ids":    torch.tensor([f["context_ids"]    for f in feats]),
            "labels":          torch.tensor([f["labels"]        for f in feats]),
            "attention_mask": torch.stack(padded_masks).bool(),  # (B, max_len, max_len)
            "val_spans":      [f["val_spans"] for f in feats],
        }
        return batch
