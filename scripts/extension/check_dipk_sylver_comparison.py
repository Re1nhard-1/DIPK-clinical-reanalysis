import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
DIPK = ROOT / "method_screen/DIPK"
OUT = DIPK / "extension"
SYLVER = ROOT / "method_screen/SYLVER"
SEED = 202609303
DRAWS = 5000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, score):
    p = int(y.sum())
    n = len(y) - p
    if not p or not n:
        return np.nan, np.nan
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    end = np.r_[np.flatnonzero(np.diff(ordered)), len(y) - 1]
    tp = np.cumsum(y[order])[end]
    fp = 1 + end - tp
    auc = np.sum(np.diff(np.r_[0, fp]) * (tp + np.r_[0, tp[:-1]])) / (2 * p * n)
    ap = np.sum(np.diff(np.r_[0, tp]) * tp / (1 + end)) / p
    return float(auc), float(ap)


def evaluate(frame):
    result = {}
    for group in ["all", "MDACC", "ISPY"]:
        sub = frame if group == "all" else frame.loc[frame.source.eq(group)]
        base = {}
        for target in ["pathology", "proxy"]:
            y = sub[target].to_numpy(dtype=int)
            for method in ["DIPK", "SLMHRD"]:
                a, p = metrics(y, sub[method].to_numpy(dtype=float))
                base[f"auc.{method}.{target}"] = a
                if group == "all":
                    base[f"ap.{method}.{target}"] = p
        for metric in (["auc", "ap"] if group == "all" else ["auc"]):
            for method in ["DIPK", "SLMHRD"]:
                base[f"{metric}.{method}.pathology_minus_proxy"] = (
                    base[f"{metric}.{method}.pathology"] - base[f"{metric}.{method}.proxy"]
                )
            for target in ["pathology", "proxy"]:
                base[f"{metric}.DIPK_minus_SLMHRD.{target}"] = (
                    base[f"{metric}.DIPK.{target}"] - base[f"{metric}.SLMHRD.{target}"]
                )
            base[f"{metric}.proxy_relative_advantage_change"] = (
                base[f"{metric}.DIPK_minus_SLMHRD.proxy"]
                - base[f"{metric}.DIPK_minus_SLMHRD.pathology"]
            )
        result.update({f"{group}.{key}": value for key, value in base.items()})
    return result


def verify(frame, results, direct=False):
    errors = []
    for group in ["all", "MDACC", "ISPY"]:
        sub = frame if group == "all" else frame.loc[frame.source.eq(group)]
        for target in ["pathology", "proxy"]:
            y = sub[target].to_numpy(dtype=int)
            for method in ["DIPK", "SLMHRD"]:
                score = sub[method].to_numpy(dtype=float)
                if len(np.unique(y)) < 2:
                    continue
                a = results[f"{group}.auc.{method}.{target}"]
                errors.append(abs(a - roc_auc_score(y, score)))
                if group == "all":
                    errors.append(abs(results[f"{group}.ap.{method}.{target}"] - average_precision_score(y, score)))
                if direct:
                    diff = score[y == 1, None] - score[None, y == 0]
                    errors.append(abs(a - ((diff > 0) + 0.5 * (diff == 0)).mean()))
    if errors and max(errors) > 1e-12:
        raise AssertionError(max(errors))
    return errors


def main():
    protected = {
        DIPK / "GSE25055_fixed_prediction_crosswalk.csv": "d769210ea5d2be25d840ceea5b667abf4e0fb4a37712cee7c012146245b58c10",
        DIPK / "sensitivity/analysis_input.csv": "1cf667165bb726fcd43beb7909f055b910be599c87e1a68f74c78a177027c276",
        DIPK / "third_cohort/GSE20194_verified_array_crosswalk.csv": "424f6fe537cf230203b9ffdae0bb1f1cb7c7eb08ead5aa9ffced679a8957b3b0",
    }
    assert all(sha(f) == h for f, h in protected.items())
    paths = [
        ROOT / "GSE25055_metadata.csv",
        DIPK / "GSE25055_fixed_prediction_crosswalk.csv",
        SYLVER / "source_figure4.xlsx",
        SYLVER / "hatzis1_released_scores.csv",
    ]
    before = {str(f.relative_to(ROOT)): sha(f) for f in paths}
    meta = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    dipk = pd.read_csv(paths[1], keep_default_na=False, float_precision="round_trip")
    source = pd.read_csv(paths[3], keep_default_na=False, float_precision="round_trip")
    assert source.source_sample_id.str.startswith("H.").all()
    source["sample id"] = source.source_sample_id.str.slice(2)
    assert meta["sample id"].is_unique and meta.GSM.is_unique
    assert source["sample id"].is_unique and dipk.GSM.is_unique
    x = source.merge(meta[["sample id", "GSM", "source", "title", "pathologic_response_pcr_rd", "dlda30_prediction"]], on="sample id", how="left", validate="one_to_one", indicator=True)
    assert x._merge.eq("both").all()
    x = x.drop(columns="_merge").merge(dipk, on="GSM", validate="one_to_one", how="left", indicator=True)
    assert x._merge.eq("both").all()
    assert x["sample id"].eq(x.title).all()
    assert x.pathologic_response_pcr_rd.isin(["pCR", "RD"]).all()
    assert x.dlda30_prediction_x.eq(x.dlda30_prediction_y).all()
    assert x.dlda30_prediction_x.eq(x.archive_group).all()
    assert x.observed_pcr_rd.eq(x.pathologic_response_pcr_rd).all()
    assert x.source_response_group.str.split(" (", regex=False).str[0].eq(x.pathologic_response_pcr_rd).all()
    assert np.isfinite(x[["SLMHRD_score", "score", "ln_ic50"]].to_numpy(float)).all()
    assert np.array_equal(x.score.to_numpy(), -x.ln_ic50.to_numpy())
    x["pathology"] = x.pathologic_response_pcr_rd.eq("pCR").astype(int)
    x["proxy"] = x.dlda30_prediction_x.eq("pCR").astype(int)
    x = x.rename(columns={"score": "DIPK", "SLMHRD_score": "SLMHRD"})
    columns = ["GSM", "source_sample_id", "sample id", "source", "pathology", "proxy", "DIPK", "SLMHRD", "source_response_group", "source_hormone_receptor_status"]
    x = x[columns].sort_values("GSM").reset_index(drop=True)
    omitted = meta.loc[~meta.GSM.isin(x.GSM), ["GSM", "sample id", "pathologic_response_pcr_rd"]]
    assert not omitted.pathologic_response_pcr_rd.isin(["pCR", "RD"]).any()
    assert set(x.source) == {"MDACC", "ISPY"}
    x.to_csv(OUT / "sylver_comparison_input.csv", index=False)
    points = evaluate(x)
    errors = verify(x, points, direct=True)
    rng = np.random.default_rng(SEED)
    draws = []
    selected = [0, 1, 2, 999, DRAWS - 1]
    for b in range(DRAWS):
        sampled = x.iloc[rng.integers(0, len(x), len(x))]
        values = evaluate(sampled)
        draws.append(values)
        if b in selected:
            errors.extend(verify(sampled, values))
    boot = pd.DataFrame(draws)
    boot.index.name = "draw"
    boot.to_csv(OUT / "sylver_comparison_bootstrap.csv")
    estimates = {}
    for key, value in points.items():
        finite = boot[key].dropna()
        estimates[key] = {"estimate": value, "ci95": finite.quantile([0.025, 0.975]).tolist(), "valid_draws": len(finite), "undefined_draws": DRAWS - len(finite)}
    assert all(sha(f) == h for f, h in protected.items())
    assert before == {str(f.relative_to(ROOT)): sha(f) for f in paths}
    result = {
        
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        
        
        
        "identity": {
            "source_rows": len(source), "matched_patients": len(x), "unmatched_source_rows": 0,
            
            
            "source_pathology_agreement": int(x.source_response_group.str.startswith("pCR").astype(int).eq(x.pathology).sum()),
            "source_proxy_agreement": int(x.source_response_group.str.startswith("pCR").astype(int).eq(x.proxy).sum()),
            "omitted_geo_records": omitted.to_dict(orient="records"),
        },
        "counts": {g: {"n": len(t), "pathology_pcr": int(t.pathology.sum()), "proxy_pcr": int(t.proxy.sum()), "discordance": int(t.pathology.ne(t.proxy).sum())} for g, t in [("all", x), ("MDACC", x.loc[x.source.eq("MDACC")]), ("ISPY", x.loc[x.source.eq("ISPY")])]},
        "estimates": estimates,
        "verification": {"standard_and_direct_pair_comparisons": len(errors), "max_abs_metric_error": max(errors), "explicit_duplicate_patient_bootstrap_draws_checked": selected, "frozen_inputs_unchanged": True, "source_inputs_and_plan_unchanged": True},
        "input_sha256": before,
        "code_sha256": sha(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    (OUT / "sylver_comparison_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key, value in estimates.items():
        if key.startswith("all.auc."):
            print(key, value)
    print("verified", result["verification"])


if __name__ == "__main__":
    main()
