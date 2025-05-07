from collections import defaultdict


_THRESH = {
    "categorical":     0.80,
    "free_form":       0.50,
    "requested_slot":  0.90,
    "transfer_in":     0.85,
    "transfer_cross":  0.90,
}


class UserStateSummariser:
    """
    Aggregate raw model outputs into the SGD 'state' object.
    """

    def __init__(self, service_schema: dict):
        self.schema = service_schema
        pass
