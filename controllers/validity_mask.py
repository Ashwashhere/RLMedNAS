import torch

def apply_action_mask(logits, mask_idx, mask_value=-1e10):
    """
    Suppresses specific action indices in the logit distribution
    to enforce structural domain constraints.
    """
    logits.data[0, mask_idx] = mask_value
    return logits