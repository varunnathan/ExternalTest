import sys
from enum import Enum


def log(*msg: tuple[str, ...]):
    print("[SGP-DST]", *msg, file=sys.stderr, flush=True)


class ModuleType(str, Enum):
    INTENT = "intent_prediction"
    CAT_SLOT = "categorical_slot_prediction"
    FREE_SLOT = "free_form_slot_prediction"
    REQUESTED_SLOT = "requested_slot_prediction"
    IN_DOMAIN_SLOT = "in_domain_slot_prediction"
    CROSS_DOMAIN_SLOT = "cross_domain_slot_prediction"
