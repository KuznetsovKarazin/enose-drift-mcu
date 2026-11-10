"""Summary of the E7 board captures (results/e7_20261110) for the manuscript."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "e7_20261110"
BOARDS = {"Mega2560": ["mega_e7b", "mega_e7ds"], "ESP32-C3": ["c3_e7"], "ESP32-S3": ["s3_e7"]}


def main():
    out = {}
    for b, dirs in BOARDS.items():
        rows = [x for d in dirs for x in json.loads((R / d / "results.json").read_text())]
        d1 = [x for x in rows if x["kind"] == "d1_stream"]
        ds = [x for x in rows if x["kind"] == "ds_fused"]
        ex = [x for x in rows if x["kind"] == "extractor"]
        per = lambda m, d, key: next((x[f"us_{key}"] / x[f"n_{key}"] for x in d1 if x["method"] == m and x["d"] == d), None)
        out[b] = dict(
            streams=len(d1), predictions=sum(x["n"] for x in d1), disagreements=sum(x["n"] - x["agree"] for x in d1),
            disagreements_non_rls=sum(x["n"] - x["agree"] for x in d1 if x["method"] != "M6"),
            ds_pairs=len(ds), ds_matches=sum(x["accuracy_board"] == x["accuracy_host_fused32"] for x in ds),
            ds_fit_fuse_us=sum(x["us_end"] for x in ds) / len(ds), ds_apply_us=sum(x["us_obs"] / x["n_obs"] for x in ds) / len(ds),
            extractor_samples=sum(x["samples"] for x in ex), extractor_us=sum(x["us_push"] for x in ex) / sum(x["n_push"] for x in ex),
            extractor_max_rel=max(x["max_rel_diff"] for x in ex),
            stack_peak=max(x["stack_peak_bytes"] for x in rows), arena_max=max(x["init_ram_bytes"] for x in d1 + ds),
            static16_us=per("M0", 16, "obs"), rls16_ref_us=per("M6", 16, "ref"), ols16_refit_us=next(x["us_end"] / x["n_end"] for x in d1 if x["method"] == "M7"),
            static128_us=per("M0", 128, "obs"), rls128_ref_us=per("M6", 128, "ref"))
    (R / "summary.json").write_text(json.dumps(out, indent=1))
    for b, v in out.items():
        print(b, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})


if __name__ == "__main__":
    main()
