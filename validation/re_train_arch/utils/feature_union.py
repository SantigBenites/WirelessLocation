# utils/data_processing.py
import re
from typing import List, Sequence, Union
from utils.feature_lists import DATASET_TO_FEATURE, union_feature_lists

_UNION_RE = re.compile(r"^union\[(.+)\]$")

def get_feature_list(dataset: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(dataset, (list, tuple)):
        out, seen = [], set()
        for f in dataset:
            if f not in seen:
                out.append(f); seen.add(f)
        return out

    if isinstance(dataset, str):
        m = _UNION_RE.match(dataset)
        if m:
            names = [p.strip() for p in m.group(1).split(",")]
            return union_feature_lists(*names)
        if dataset in DATASET_TO_FEATURE:
            return DATASET_TO_FEATURE[dataset]

    raise ValueError("Unknown feature selection. "
                     "Use a preset name or union[presetA,presetB,...].")

def _default_value_for(name: str):
    n = name.lower()
    if n.endswith("_rssi"):                   return -100.0
    if "_over_" in n:                         return 1.0
    if n.startswith("delta_"):                return 0.0
    if n.endswith("_share"):                  return 0.0
    if n.endswith("_residual"):               return 0.0
    if n.endswith("_rssi_1m"):                return -30.0
    if n.startswith("ap_intercepts_"):        return 0.0
    if n in {"beta1_log10d","beta2_20log10f","n_est","xyfree_gamma_total_power_db"}: return 0.0
    if n.startswith("xyfree_intercept_"):     return 0.0
    return 0.0
