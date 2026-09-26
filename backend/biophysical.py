"""34 biophysical features for Model1's gram_positive / fungal heads.

Verbatim port of compute_biophysical_features() from Model1/predict.py
(Bio + modlamp are imported lazily so the package imports without them).
Feature order is dictated by model_artifacts/config.json at call time.
"""

from .config import STANDARD_AA


def compute_biophysical_features(seq: str):
    seq = seq.upper().strip()
    if len(seq) == 0 or not set(seq) <= set(STANDARD_AA):
        return None
    features = {}
    for aa in STANDARD_AA:
        features[f"frac_{aa}"] = seq.count(aa) / len(seq)
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        pa = ProteinAnalysis(seq)
        features["gravy"] = pa.gravy()
        helix, turn, sheet = pa.secondary_structure_fraction()
        features["ss_helix_frac"] = helix
        features["ss_turn_frac"] = turn
        features["ss_sheet_frac"] = sheet
    except Exception:
        return None
    try:
        from modlamp.descriptors import GlobalDescriptor

        desc = GlobalDescriptor(seq)
        desc.calculate_all()
        for name, val in zip(desc.featurenames, desc.descriptor[0]):
            features[f"modlamp_{name}"] = val
    except Exception:
        return None
    return features


def biophysical_matrix(sequences: list, feature_names: list):
    """DataFrame with exactly the columns config.json lists, in order.

    Returns (df, failed_indices). A sequence whose features cannot be
    computed is reported, never silently dropped.
    """
    import pandas as pd

    rows = []
    failed = []
    for i, s in enumerate(sequences):
        feats = compute_biophysical_features(s)
        if feats is None:
            failed.append(i)
            continue
        rows.append([feats.get(c, float("nan")) for c in feature_names])
    df = pd.DataFrame(rows, columns=feature_names)
    return df, failed
