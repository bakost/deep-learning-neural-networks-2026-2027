"""ЛР6а: три задачи о самовнимании и softmax."""

from .attention import AttentionParams, init_params, permutation_matrix, self_attention
from .counts import parameter_counts
from .softmax import TASK_LOGITS, decimal_reference, softmax, softmax_jacobian

__all__ = [
    "AttentionParams", "TASK_LOGITS", "decimal_reference", "init_params", "parameter_counts",
    "permutation_matrix", "self_attention", "softmax", "softmax_jacobian",
]
