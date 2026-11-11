"""Reproduce grouped, out-of-animal Sturman OFT pose-classifier evaluation.

Run: .venv/bin/python 공개영상_행동검증.py DATA_DIR --output 결과.json
DATA_DIR contains labels.csv, manifest.json and all verified source MP4/DLC files.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import gzip
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from behavtrack.analysis.common import drop_short_runs, frame_durations
from behavtrack.classifier import features
from behavtrack.pose_import import body_center, read_dlc_csv
from behavtrack.tracks import Tracks

FPS = 25.0
RATERS = ("Jin", "Oliver", "Furkan")
TARGETS = {"rearing": {"Supported", "Unsupported"}, "grooming": {"Grooming"}}
ARENA_PARTS = {"tl", "tr", "bl", "br", "centre"}


def verified_files(root, manifest):
    labels = (root / "labels.csv").read_bytes()
    label_sha = hashlib.sha1(b"blob " + str(len(labels)).encode() + b"\0" + labels).hexdigest()
    if label_sha != "599a0a01976e24f8b89b794c8728f203bcb4958b":
        raise ValueError("Label checksum mismatch")
    for item in manifest["files"]:
        path = root / item["name"]
        data = path.read_bytes()
        actual = (hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                  if item["type"] == "git" else hashlib.md5(data).hexdigest())
        if actual != item["expected"]:
            raise ValueError(f"Checksum mismatch: {path}")


def labels_for(rows, rater, times, target):
    rr = [r for r in rows if r["Experimenter"] == rater]
    markers = [r for r in rr if r["type"] in ("StartEnd", "Start/End") and r["from"] and r["to"]]
    markers.sort(key=lambda r: float(r["from"]))
    if len(markers) < 2:
        if rater != "Jin":
            return np.zeros(len(times), bool), np.zeros(len(times), bool)
        raise ValueError(f"Missing start/end markers: {rr[0]['ID']} {rater}")
    # Markers bound the experiment; their own and other unspecified intervals are not scored.
    valid = (times > float(markers[0]["to"])) & (times < float(markers[-1]["from"]))
    y = np.zeros(len(times), dtype=bool)
    for r in rr:
        if not r["from"] or not r["to"] or r["to"] == "NA":
            continue
        start, end = float(r["from"]), float(r["to"])
        inside = (times >= start) & (times <= end)
        if r["type"] in target:
            y |= inside
        elif r["type"] in ("StartEnd", "Start/End", "_DEFAULT", "NA"):
            valid &= ~inside
    return y, valid


def counts(truth, pred, valid):
    y, p = truth[valid], pred[valid]
    return {"tp": int(np.sum(y & p)), "fn": int(np.sum(y & ~p)),
            "fp": int(np.sum(~y & p)), "tn": int(np.sum(~y & ~p))}


def scores(c):
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn
    div = lambda a, b: a / b if b else None
    return {"precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "accuracy": div(tp + tn, total),
            "specificity": div(tn, tn + fp), "frames": total, "positive_frames": tp + fn}


def aggregate(rows, field):
    return {k: sum(r[field][k] for r in rows) for k in ("tp", "fp", "fn", "tn")}


def bootstrap(rows, seed=0):
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("precision", "recall", "f1", "accuracy")}
    for _ in range(5000):
        chosen = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        values = scores(aggregate(chosen, "counts"))
        for key in samples:
            if values[key] is not None:
                samples[key].append(values[key])
    return {k: [float(x) for x in np.quantile(v, [0.025, 0.975])] if v else None for k, v in samples.items()}


def alignment_diagnostic(path, points, corners, n):
    """Coarse independent video/pose check; a centroid is not a body-centre ground truth."""
    cap = cv2.VideoCapture(str(path))
    images = []
    for i in np.linspace(300, n - 300, 60, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            images.append((i, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    cap.release()
    if len(images) < 40:
        raise ValueError(f"Could not sample video: {path}")
    bg = np.median(np.stack([im for _, im in images]), axis=0).astype(np.uint8)
    mask = np.zeros_like(bg)
    cv2.fillPoly(mask, [np.array(corners, dtype=np.int32)], 255)
    samples = []
    for i, gray in images:
        _, binary = cv2.threshold(cv2.subtract(bg, gray), 20, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, mask)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 150 < cv2.contourArea(c) < 8000]
        if contours:
            m = cv2.moments(max(contours, key=cv2.contourArea))
            samples.append((i, np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])))
    if len(samples) < 20:
        return {"samples": len(samples), "best_offset": None}
    errors = {}
    for offset in range(-10, 11):
        d = [np.linalg.norm(pt - points["bodycentre"][i + offset, :2]) for i, pt in samples]
        errors[offset] = float(np.mean(np.minimum(d, 40)))
    return {"samples": len(samples), "best_offset": min(errors, key=errors.get),
            "mean_capped_error_at_zero_px": errors[0], "mean_capped_error_at_best_px": min(errors.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--download-from", type=Path, help="이전 결과 JSON의 출처 목록으로 원본을 다시 받기")
    args = p.parse_args()
    root = args.data_dir
    if args.download_from:
        previous = json.loads(args.download_from.read_text(encoding="utf-8"))
        root.mkdir(parents=True, exist_ok=True)
        source_list = previous["sources"]
        manifest = {"ids": [x["id"] for x in previous["videos"]], "files": source_list}
        label_url = "https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/cfdc34a85b6ce961bbddd8905a1a62b714dbc11a/data/OFT/Labels/AllLabDataOFT_final.csv"
        def download(item):
            name, url = item
            if Path(name).name != name or not (url.startswith("https://raw.githubusercontent.com/ETHZ-INS/DLCAnalyzer/") or url.startswith("https://zenodo.org/")):
                raise ValueError("Unexpected source URL or file name")
            request = urllib.request.Request(url, headers={"User-Agent": "behavtrack-validation/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, (root / name).open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(download, [(x["name"], x["url"]) for x in source_list] + [("labels.csv", label_url)]))
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text())
    verified_files(root, manifest)
    with (root / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f, delimiter=";"))
    animals = []
    for id in manifest["ids"]:
        rows = [r for r in raw if r["ID"] == id]
        filenames = {r["DLCFile"] for r in rows}
        if len(filenames) != 1:
            raise ValueError(f"Ambiguous pose mapping: {id}")
        pose_file = next(iter(filenames))
        fr, data = read_dlc_csv(root / pose_file)
        points = data["single"]
        video = cv2.VideoCapture(str(root / (id + ".mp4")))
        n, fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT)), float(video.get(cv2.CAP_PROP_FPS))
        opened = video.isOpened()
        video.release()
        if not opened or abs(fps - FPS) > 1e-6 or not np.array_equal(fr, np.arange(len(fr))) or n <= 0 or abs(len(fr) - n) > 10:
            raise ValueError(f"Frame count, rate or frame sequence mismatch: {id}: video={n}, pose={len(fr)}, fps={fps}")
        n = min(n, len(fr))
        bodyparts = {k: v[:n] for k, v in points.items() if k not in ARENA_PARTS}
        center = body_center(bodyparts, 0.6)
        all_confident = np.logical_and.reduce([np.isfinite(v[:, :2]).all(axis=1) & (v[:, 2] >= 0.6) for v in bodyparts.values()])
        body_valid = np.isfinite(center).all(axis=1)
        t = np.arange(n) / FPS
        tr = Tracks(np.arange(n), t, center[:, 0], center[:, 1], {}, bodyparts)
        corners = [np.median(points[k][:, :2], axis=0) for k in ("tl", "tr", "br", "bl")]
        alignment = alignment_diagnostic(root / (id + ".mp4"), points, corners, n)
        if alignment["best_offset"] is None or abs(alignment["best_offset"]) > 2:
            raise ValueError(f"Video/pose alignment needs manual investigation: {id}: {alignment}")
        perimeter = sum(np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4))
        scale = perimeter / 180.0  # published 45 x 45 cm open-field arena
        names, X = features(tr, scale, 0.6)
        if animals and names != animals[0]["names"]:
            raise ValueError(f"Incompatible keypoints: {id}")
        manual = {target: {rater: labels_for(rows, rater, t, classes) for rater in RATERS}
                  for target, classes in TARGETS.items()}
        animals.append({"id": id, "pose": pose_file, "n": n, "pose_frames": len(fr), "fps": fps,
                        "scale_px_cm": scale, "names": names, "X": X.astype(np.float32), "manual": manual,
                        "times": t, "alignment": alignment,
                        "predictions": {},
                        "all_keypoints_confident_pct": 100 * float(np.mean(all_confident)),
                        "body_center_valid_pct": 100 * float(np.mean(body_valid))})
        print(f"준비: {id}, 영상 {n}프레임, 포즈 {len(fr)}프레임, 특징 {len(names)}개, 시차 {alignment['best_offset']}", flush=True)

    split = list(GroupKFold(5).split(np.arange(len(animals)), groups=[a["id"] for a in animals]))
    results = {k: [] for k in TARGETS}
    for fold, (tr_ids, te_ids) in enumerate(split, 1):
        for behavior in TARGETS:
            XX, yy = [], []
            for i in tr_ids:
                a = animals[i]
                y, valid = a["manual"][behavior]["Jin"]
                sampled = valid & (np.arange(a["n"]) % 5 == 0)
                XX.append(a["X"][sampled]); yy.append(y[sampled])
            X_train, y_train = np.concatenate(XX), np.concatenate(yy)
            model = RandomForestClassifier(n_estimators=50, class_weight="balanced_subsample", min_samples_leaf=2,
                                           n_jobs=4, random_state=0)
            model.fit(X_train, y_train)
            del X_train, y_train, XX, yy
            for i in te_ids:
                a = animals[i]
                prob = model.predict_proba(a["X"])[:, 1]
                predicted = drop_short_runs(prob >= 0.5, frame_durations(a["times"]), 0.2)
                a["predictions"][behavior] = (prob, predicted, fold)
                truth, valid = a["manual"][behavior]["Jin"]
                row = {"id": a["id"], "fold": fold, "counts": counts(truth, predicted, valid),
                       "truth_s": float(np.sum(truth & valid) / FPS),
                       "predicted_s": float(np.sum(predicted & valid) / FPS),
                       "label_valid_s": float(np.sum(valid) / FPS)}
                row["offset_sensitivity"] = {}
                for offset in (-1, 1):
                    shifted = np.roll(truth, offset)
                    valid_shift = valid & np.roll(valid, offset)
                    valid_shift[0 if offset > 0 else -1] = False
                    row["offset_sensitivity"][str(offset)] = counts(shifted, predicted, valid_shift)
                row["human_comparison"] = {}
                for rater in RATERS[1:]:
                    other, other_valid = a["manual"][behavior][rater]
                    row["human_comparison"][rater] = counts(truth, other, valid & other_valid)
                results[behavior].append(row)
            print(f"완료: fold {fold}/5, {behavior}, test {[animals[i]['id'] for i in te_ids]}", flush=True)

    summary = {}
    for behavior, rows in results.items():
        pooled = aggregate(rows, "counts")
        human = {}
        for rater in RATERS[1:]:
            c = {k: sum(r["human_comparison"][rater][k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
            human[rater] = {"counts": c, "scores": scores(c)}
        summary[behavior] = {"counts": pooled, "scores": scores(pooled), "bootstrap_95": bootstrap(rows),
                             "individuals": rows, "human_comparison": human,
                             "one_frame_offset_sensitivity": {
                                 offset: scores({k: sum(r["offset_sensitivity"][offset][k] for r in rows)
                                                 for k in ("tp", "fp", "fn", "tn")})
                                 for offset in ("-1", "1")}}
    out = {"dataset": "10.5281/zenodo.3608658", "labels_git_blob": "599a0a01976e24f8b89b794c8728f203bcb4958b",
           "fps": FPS, "method": {"split": "GroupKFold 5, one mouse per video", "reference": "Jin",
              "training_stride_frames": 5, "estimators": 50, "min_samples_leaf": 2, "seed": 0,
              "threshold": 0.5, "min_bout_s": 0.2, "keypoint_min_p": 0.6, "excluded_keypoints": sorted(ARENA_PARTS)},
           "sources": manifest["files"], "videos": [{k: a[k] for k in ("id", "pose", "n", "pose_frames", "fps", "scale_px_cm", "alignment", "all_keypoints_confident_pct", "body_center_valid_pct")} for a in animals],
           "results": summary}
    frame_file = args.output.with_name("행동인식_프레임별.csv.gz")
    with gzip.open(frame_file, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frame", "time_s", "fold", "reference_valid", "rearing_truth", "rearing_probability",
                    "rearing_prediction", "grooming_truth", "grooming_probability", "grooming_prediction"])
        for a in animals:
            rt, valid = a["manual"]["rearing"]["Jin"]
            gt, valid_g = a["manual"]["grooming"]["Jin"]
            rp, rl, fold = a["predictions"]["rearing"]
            gp, gl, _ = a["predictions"]["grooming"]
            if not np.array_equal(valid, valid_g):
                raise ValueError("Behavior reference-valid masks differ")
            for i in range(a["n"]):
                w.writerow([a["id"], i, f"{i / FPS:.2f}", fold, int(valid[i]), int(rt[i]) if valid[i] else "",
                            f"{rp[i]:.6f}", int(rl[i]), int(gt[i]) if valid[i] else "", f"{gp[i]:.6f}", int(gl[i])])
    out["frame_output"] = {"file": frame_file.name, "sha256": hashlib.sha256(frame_file.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
