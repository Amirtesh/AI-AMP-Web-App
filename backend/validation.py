"""Input validation for peptide sequences (stdlib only).

Model1's predict.py enforces: non-empty + 20-standard-AA alphabet, no length
bound. Model2's predict.py enforces the same plus 5aa <= len <= 100aa.
Both behaviours are preserved here instead of inventing a unified rule.
"""

from .config import MODEL2_MAX_LEN, MODEL2_MIN_LEN, STANDARD_AA


def _check_alphabet(seq: str) -> set:
    return set(seq) - set(STANDARD_AA)


def validate_predict_sequences(
    sequences,
    *,
    enforce_length_range: bool,
    what: str = "sequences",
) -> tuple[list, list]:
    """Returns (clean, errors).

    clean: upper-cased, stripped sequences in input order.
    errors: [{"index", "sequence", "reason"}] — one entry per rejected input,
    with the specific offending character(s) or limit named.
    """
    clean: list = []
    errors: list = []
    for i, raw in enumerate(sequences):
        if not isinstance(raw, str):
            errors.append(
                {
                    "index": i,
                    "sequence": str(raw)[:60],
                    "reason": f"{what}[{i}] must be a string, got "
                    f"{type(raw).__name__}",
                }
            )
            continue
        seq = raw.strip().upper()
        if not seq:
            errors.append(
                {
                    "index": i,
                    "sequence": raw[:60],
                    "reason": f"{what}[{i}] is empty after stripping whitespace",
                }
            )
            continue
        bad = _check_alphabet(seq)
        if bad:
            errors.append(
                {
                    "index": i,
                    "sequence": seq[:60],
                    "reason": f"{what}[{i}] contains non-standard amino acid "
                    f"character(s) {sorted(bad)}; only the 20 standard amino "
                    "acids ACDEFGHIKLMNPQRSTVWY are accepted",
                }
            )
            continue
        if enforce_length_range:
            if len(seq) < MODEL2_MIN_LEN:
                errors.append(
                    {
                        "index": i,
                        "sequence": seq[:60],
                        "reason": f"{what}[{i}] length {len(seq)}aa is below "
                        f"{MODEL2_MIN_LEN}aa — outside Model2's training "
                        "length range",
                    }
                )
                continue
            if len(seq) > MODEL2_MAX_LEN:
                errors.append(
                    {
                        "index": i,
                        "sequence": seq[:60] + ("..." if len(seq) > 60 else ""),
                        "reason": f"{what}[{i}] length {len(seq)}aa exceeds "
                        f"{MODEL2_MAX_LEN}aa — outside Model2's training "
                        "length range",
                    }
                )
                continue
        clean.append(seq)
    return clean, errors


def validation_error_detail(errors: list) -> dict:
    return {
        "message": f"{len(errors)} sequence(s) failed validation",
        "errors": errors,
    }
