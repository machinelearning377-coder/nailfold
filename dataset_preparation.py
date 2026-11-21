from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

REPO_ID = "HanaNguyen/Capillary-Dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MORPHOLOGY_NAMES = {0: "Bushy", 1: "Crossing", 2: "Hairpin", 3: "Tortuous"}
SEED = 42


def normalize_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return text.lower()


def decoded_pixel_hash(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return hashlib.sha1(rgb.tobytes()).hexdigest()


def compact_hashes(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(str(path))
    x = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(x))
    block = dct[:8, :8]
    median = np.median(block[1:, :])
    phash = int.from_bytes(np.packbits((block > median).ravel()).tobytes(), "big")
    x = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = x[:, 1:] > x[:, :-1]
    dhash = int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")
    return phash, dhash


def normalized_thumbnail(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(str(path))
    gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray -= gray.mean()
    return gray / (gray.std() + 1e-6)


def classification_participant(path, class_root):
    rel = path.relative_to(class_root)
    if len(rel.parts) > 1:
        return normalize_id(rel.parts[0])
    match = re.match(r"(?i)(?:hea|dia)[_-]([^_]+)", path.stem)
    if match:
        return normalize_id(match.group(1))
    raise ValueError(f"Participant identity could not be recovered for {path}")


def download_dataset(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(data_dir), max_workers=8)


def audit_inventory(data_dir, audit_dir):
    rows = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or ".cache" in path.parts:
            continue
        rows.append({"path": str(path.resolve()), "relative_path": str(path.relative_to(data_dir)), "size_bytes": path.stat().st_size, "extension": path.suffix.lower()})
    frame = pd.DataFrame(rows)
    frame.to_csv(audit_dir / "file_inventory.csv", index=False)
    summary = {
        "repository": REPO_ID,
        "files": int(len(frame)),
        "size_gb": float(frame.size_bytes.sum() / 1024**3) if len(frame) else 0.0,
        "zero_byte_files": int((frame.size_bytes == 0).sum()) if len(frame) else 0,
    }
    (audit_dir / "dataset_audit.json").write_text(json.dumps(summary, indent=2))


def build_classification_manifest(data_dir, split_dir):
    roots = {
        "T2DM": data_dir / "classification_original" / "diabetic_patients",
        "Healthy": data_dir / "classification_original" / "healthy_subjects",
    }
    rows = []
    for source, root in roots.items():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(path) as image:
                width, height = image.size
            participant = classification_participant(path, root)
            rows.append({
                "source": source,
                "label": 1 if source == "T2DM" else 0,
                "participant": participant,
                "subject_key": f"{source}|{participant}",
                "raw_path": str(path.resolve()),
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "pixel_hash": decoded_pixel_hash(path),
            })
    frame = pd.DataFrame(rows)
    cross = frame.groupby("pixel_hash")["subject_key"].nunique()
    ambiguous = set(cross[cross > 1].index)
    frame = frame[~frame.pixel_hash.isin(ambiguous)].copy()
    frame = frame.drop_duplicates(["subject_key", "pixel_hash"], keep="first").reset_index(drop=True)
    participants = frame[["source", "label", "participant", "subject_key"]].drop_duplicates().reset_index(drop=True)
    train_subjects, temporary_subjects = train_test_split(participants, test_size=0.30, stratify=participants.label, random_state=SEED)
    val_subjects, test_subjects = train_test_split(temporary_subjects, test_size=0.50, stratify=temporary_subjects.label, random_state=SEED)
    split_map = {}
    for split_name, subset in [("train", train_subjects), ("val", val_subjects), ("test", test_subjects)]:
        for subject in subset.subject_key:
            split_map[subject] = split_name
    frame["split"] = frame.subject_key.map(split_map)
    participants["split"] = participants.subject_key.map(split_map)
    assert frame.split.notna().all()
    assert len(set(train_subjects.subject_key) & set(val_subjects.subject_key)) == 0
    assert len(set(train_subjects.subject_key) & set(test_subjects.subject_key)) == 0
    assert len(set(val_subjects.subject_key) & set(test_subjects.subject_key)) == 0
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert len(set(frame.loc[frame.split.eq(a), "pixel_hash"]) & set(frame.loc[frame.split.eq(b), "pixel_hash"])) == 0
    frame.to_csv(split_dir / "classification_image_manifest.csv", index=False)
    participants.to_csv(split_dir / "classification_participant_splits.csv", index=False)
    summary = frame.groupby(["split", "source"]).agg(participants=("subject_key", "nunique"), images=("raw_path", "size")).reset_index()
    summary.to_csv(split_dir / "classification_split_summary.csv", index=False)
    strict = frame[frame.resolution.eq("640x480")].copy()
    strict.to_csv(split_dir / "classification_strict_640_manifest.csv", index=False)
    return frame, participants


def morphology_label_path(data_dir, image_path):
    image_root = data_dir / "Morphology_detection" / "images"
    label_root = data_dir / "Morphology_detection" / "labels"
    rel = image_path.relative_to(image_root)
    return label_root / rel.with_suffix(".txt")


def read_morphology_counts(label_path):
    counts = Counter()
    bad = 0
    if not label_path.exists():
        return counts, 0, 1
    for line in label_path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        values = line.split()
        try:
            cid, xc, yc, width, height = map(float, values[:5])
            cid = int(cid)
            valid = cid in MORPHOLOGY_NAMES and 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1
            if valid:
                counts[cid] += 1
            else:
                bad += 1
        except Exception:
            bad += 1
    return counts, sum(counts.values()), bad


def parse_morphology_subject(path, participant_table):
    stem = path.stem
    lower = stem.lower()
    known = {}
    for source in ["Healthy", "T2DM"]:
        values = participant_table.loc[participant_table.source.eq(source), "participant"].astype(str).unique().tolist()
        known[source] = sorted(values, key=len, reverse=True)
    if lower.startswith("hea_"):
        rest = stem[4:]
        for participant in known["Healthy"]:
            if rest == participant or rest.startswith(participant + "_"):
                return "Healthy", participant
    if lower.startswith("dia_"):
        rest = stem[4:]
        for participant in known["T2DM"]:
            if rest == participant or rest.startswith(participant + "_"):
                return "T2DM", participant
    for source in ["Healthy", "T2DM"]:
        for participant in known[source]:
            if stem == participant or stem.startswith(participant + "_"):
                return source, participant
    return None, None


def build_morphology_manifest(data_dir, split_dir, classification, participants):
    image_root = data_dir / "Morphology_detection" / "images"
    exact_map = classification[["pixel_hash", "source", "participant", "subject_key"]].drop_duplicates("pixel_hash")
    exact_map = exact_map.set_index("pixel_hash").to_dict("index")
    rows = []
    invalid_rows = 0
    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(image_root)
        repository_split = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        label_path = morphology_label_path(data_dir, path)
        counts, objects, bad = read_morphology_counts(label_path)
        invalid_rows += bad
        pixel_hash = decoded_pixel_hash(path)
        exact = exact_map.get(pixel_hash)
        parsed_source, parsed_participant = parse_morphology_subject(path, participants)
        if exact is not None:
            final_source = exact["source"]
            final_participant = str(exact["participant"])
            identity_source = "exact_pixel"
        elif parsed_participant is not None:
            final_source = parsed_source
            final_participant = str(parsed_participant)
            identity_source = "filename"
        else:
            final_source = None
            final_participant = None
            identity_source = "unresolved"
        phash, dhash = compact_hashes(path)
        rows.append({
            "repository_split": repository_split,
            "raw_path": str(path.resolve()),
            "label_path": str(label_path.resolve()),
            "pixel_hash": pixel_hash,
            "phash": format(phash, "016x"),
            "dhash": format(dhash, "016x"),
            "clinical_label": final_source,
            "participant": final_participant,
            "final_subject": f"{final_source}|{final_participant}" if final_participant is not None else None,
            "identity_source": identity_source,
            "bushy": counts[0],
            "crossing": counts[1],
            "hairpin": counts[2],
            "tortuous": counts[3],
            "objects": objects,
        })
    frame = pd.DataFrame(rows)
    classification_hashes = []
    for row in classification.itertuples():
        phash, dhash = compact_hashes(row.raw_path)
        classification_hashes.append((row.raw_path, row.source, row.participant, row.subject_key, phash, dhash))
    classification_hashes = pd.DataFrame(classification_hashes, columns=["raw_path", "source", "participant", "subject_key", "phash", "dhash"])
    thumb_cache = {}
    def thumb(path):
        if path not in thumb_cache:
            thumb_cache[path] = normalized_thumbnail(path)
        return thumb_cache[path]
    unresolved_indexes = frame.index[frame.final_subject.isna()].tolist()
    for index in unresolved_indexes:
        row = frame.loc[index]
        name = Path(row.raw_path).name.lower()
        if name.startswith("hea_"):
            candidates = classification_hashes[classification_hashes.source.eq("Healthy")]
        elif name.startswith("dia_"):
            candidates = classification_hashes[classification_hashes.source.eq("T2DM")]
        else:
            candidates = classification_hashes
        phash = int(row.phash, 16)
        dhash = int(row.dhash, 16)
        scored = []
        for candidate in candidates.itertuples():
            pdistance = (phash ^ int(candidate.phash)).bit_count()
            ddistance = (dhash ^ int(candidate.dhash)).bit_count()
            scored.append((pdistance + ddistance, pdistance, ddistance, candidate.Index))
        scored.sort(key=lambda value: value[0])
        query = normalized_thumbnail(row.raw_path)
        checked = []
        for total_distance, pdistance, ddistance, candidate_index in scored[:10]:
            candidate = classification_hashes.loc[candidate_index]
            similarity = float(np.mean(query * thumb(candidate.raw_path)))
            checked.append((similarity, total_distance, pdistance, ddistance, candidate_index))
        checked.sort(reverse=True, key=lambda value: value[0])
        if len(checked) < 2:
            continue
        best, second = checked[0], checked[1]
        similarity, total_distance, _, _, candidate_index = best
        margin = similarity - second[0]
        accepted = similarity >= 0.995 or (similarity >= 0.985 and total_distance <= 10 and margin >= 0.005)
        if accepted:
            candidate = classification_hashes.loc[candidate_index]
            frame.loc[index, "clinical_label"] = candidate.source
            frame.loc[index, "participant"] = str(candidate.participant)
            frame.loc[index, "final_subject"] = candidate.subject_key
            frame.loc[index, "identity_source"] = "perceptual_match"
    unresolved = frame[frame.final_subject.isna()].copy().reset_index()
    parent = list(range(len(unresolved)))
    rank = [0] * len(unresolved)
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
    phashes = [int(value, 16) for value in unresolved.phash]
    dhashes = [int(value, 16) for value in unresolved.dhash]
    for i in range(len(unresolved)):
        for j in range(i + 1, len(unresolved)):
            pdistance = (phashes[i] ^ phashes[j]).bit_count()
            if pdistance > 6:
                continue
            ddistance = (dhashes[i] ^ dhashes[j]).bit_count()
            if ddistance <= 6 and pdistance + ddistance <= 10:
                union(i, j)
    root_ids = {root: f"visual_{number:04d}" for number, root in enumerate(sorted({find(i) for i in range(len(unresolved))}), 1)}
    frame["hybrid_group"] = None
    known = frame.final_subject.notna()
    frame.loc[known, "hybrid_group"] = "subject_" + frame.loc[known, "final_subject"].astype(str)
    for local_index, original_index in enumerate(unresolved["index"]):
        frame.loc[original_index, "hybrid_group"] = root_ids[find(local_index)]
    assert frame.hybrid_group.notna().all()
    participant_split = participants.set_index("subject_key")["split"].to_dict()
    group_rows = []
    for group, group_frame in frame.groupby("hybrid_group"):
        subjects = group_frame.final_subject.dropna().unique().tolist()
        subject = subjects[0] if subjects else None
        group_rows.append({
            "hybrid_group": group,
            "final_subject": subject,
            "forced_split": participant_split.get(subject) if subject is not None else None,
            "images": len(group_frame),
            "bushy": int(group_frame.bushy.sum()),
            "crossing": int(group_frame.crossing.sum()),
            "hairpin": int(group_frame.hairpin.sum()),
            "tortuous": int(group_frame.tortuous.sum()),
        })
    groups = pd.DataFrame(group_rows)
    features = ["images", "bushy", "crossing", "hairpin", "tortuous"]
    matrix = groups[features].to_numpy(float)
    totals = matrix.sum(axis=0)
    ratios = np.array([0.70, 0.15, 0.15])
    targets = ratios[:, None] * totals[None, :]
    weights = np.array([3.0, 1.5, 1.0, 1.0, 1.5])
    split_names = ["train", "val", "test"]
    split_index = {name: index for index, name in enumerate(split_names)}
    fixed = np.full(len(groups), -1, dtype=int)
    current = np.zeros((3, len(features)), dtype=float)
    for i, row in groups.iterrows():
        if pd.notna(row.forced_split):
            split_id = split_index[row.forced_split]
            fixed[i] = split_id
            current[split_id] += matrix[i]
    unresolved_group_indexes = np.where(fixed == -1)[0]
    impact = matrix[:, 0] / max(totals[0], 1) + (matrix[:, 1:] / np.maximum(totals[1:], 1)).sum(axis=1)
    rng = np.random.default_rng(SEED)
    best_score = np.inf
    best_assignment = None
    for _ in range(250):
        trial_current = current.copy()
        assignment = fixed.copy()
        jitter = rng.random(len(unresolved_group_indexes)) * 0.03
        order = unresolved_group_indexes[np.argsort(-(impact[unresolved_group_indexes] + jitter))]
        for index in order:
            best_local = np.inf
            best_split = None
            for split_id in range(3):
                candidate = trial_current.copy()
                candidate[split_id] += matrix[index]
                relative = (candidate - targets) / np.maximum(targets, 1)
                cost = np.sum((relative ** 2) * weights)
                over = max(0, candidate[split_id, 0] - targets[split_id, 0])
                cost += 10.0 * (over / max(targets[split_id, 0], 1)) ** 2
                if cost < best_local:
                    best_local = cost
                    best_split = split_id
            assignment[index] = best_split
            trial_current[best_split] += matrix[index]
        relative = (trial_current - targets) / np.maximum(targets, 1)
        score = np.sum((relative ** 2) * weights)
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()
    groups["split"] = [split_names[index] for index in best_assignment]
    group_map = dict(zip(groups.hybrid_group, groups.split))
    frame["split"] = frame.hybrid_group.map(group_map)
    known_frame = frame[frame.final_subject.notna()].copy()
    known_frame["classification_split"] = known_frame.final_subject.map(participant_split)
    assert (known_frame.split == known_frame.classification_split).all()
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert len(set(frame.loc[frame.split.eq(a), "hybrid_group"]) & set(frame.loc[frame.split.eq(b), "hybrid_group"])) == 0
        assert len(set(frame.loc[frame.split.eq(a), "pixel_hash"]) & set(frame.loc[frame.split.eq(b), "pixel_hash"])) == 0
    frame.to_csv(split_dir / "morphology_image_manifest.csv", index=False)
    groups.to_csv(split_dir / "morphology_group_splits.csv", index=False)
    summary = frame.groupby("split").agg(images=("raw_path", "size"), objects=("objects", "sum"), bushy=("bushy", "sum"), crossing=("crossing", "sum"), hairpin=("hairpin", "sum"), tortuous=("tortuous", "sum")).reindex(["train", "val", "test"])
    summary.to_csv(split_dir / "morphology_split_summary.csv")
    (split_dir / "morphology_audit.json").write_text(json.dumps({"images": int(len(frame)), "objects": int(frame.objects.sum()), "invalid_annotation_rows": int(invalid_rows), "participant_resolved_images": int(frame.final_subject.notna().sum()), "unresolved_images": int(frame.final_subject.isna().sum()), "assignment_score": float(best_score)}, indent=2))
    return frame


def prepare_morphology_yolo(data_dir, prepared_dir, morphology):
    root = prepared_dir / "morphology_yolo"
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    rows = []
    for row in morphology.itertuples():
        source_image = Path(row.raw_path)
        source_label = Path(row.label_path)
        name = f"{source_image.stem}__{row.pixel_hash[:8]}{source_image.suffix.lower()}"
        image_destination = root / "images" / row.split / name
        label_destination = root / "labels" / row.split / f"{Path(name).stem}.txt"
        if not image_destination.exists():
            shutil.copy2(source_image, image_destination)
        if not label_destination.exists():
            shutil.copy2(source_label, label_destination)
        rows.append({"split": row.split, "image_path": str(image_destination.resolve()), "label_path": str(label_destination.resolve()), "source_image": str(source_image.resolve())})
    pd.DataFrame(rows).to_csv(prepared_dir / "morphology_yolo_manifest.csv", index=False)
    yaml_text = "path: " + str(root.resolve()) + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: Bushy\n  1: Crossing\n  2: Hairpin\n  3: Tortuous\n"
    (prepared_dir / "morphology_yolo.yaml").write_text(yaml_text)


def extract_morphology_crops(prepared_dir, morphology):
    crop_root = prepared_dir / "morphology_crops"
    rows = []
    for row in morphology.itertuples():
        with Image.open(row.raw_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for object_id, line in enumerate(Path(row.label_path).read_text(errors="ignore").splitlines()):
                if not line.strip():
                    continue
                cid, xc, yc, bw, bh = map(float, line.split()[:5])
                cid = int(cid)
                center_x, center_y = xc * width, yc * height
                box_width, box_height = bw * width, bh * height
                margin = 0.25 * max(box_width, box_height)
                x1 = max(0, int(round(center_x - box_width / 2 - margin)))
                y1 = max(0, int(round(center_y - box_height / 2 - margin)))
                x2 = min(width, int(round(center_x + box_width / 2 + margin)))
                y2 = min(height, int(round(center_y + box_height / 2 + margin)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image.crop((x1, y1, x2, y2))
                class_name = MORPHOLOGY_NAMES[cid]
                destination_dir = crop_root / row.split / class_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{Path(row.raw_path).stem}__{row.pixel_hash[:8]}__obj{object_id:03d}.jpg"
                crop.save(destination, quality=97, subsampling=0)
                rows.append({"split": row.split, "class_id": cid, "class_name": class_name, "crop_path": str(destination.resolve()), "source_image": row.raw_path, "subject_key": row.final_subject, "hybrid_group": row.hybrid_group, "pixel_hash": row.pixel_hash})
    crops = pd.DataFrame(rows)
    crops.to_csv(prepared_dir / "morphology_crop_manifest.csv", index=False)
    return crops


def clahe_image(source, destination):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(str(source))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lightness)
    output = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), output, [cv2.IMWRITE_JPEG_QUALITY, 97])


def build_inspection_manifest(prepared_dir, classification, morphology):
    inspection_dir = prepared_dir / "inspection_clahe"
    rows = []
    for source in ["T2DM", "Healthy"]:
        sample = classification[classification.source.eq(source)].sort_values(["participant", "raw_path"]).iloc[0]
        destination = inspection_dir / f"{source.lower()}.jpg"
        clahe_image(sample.raw_path, destination)
        rows.append({"type": source, "raw_path": sample.raw_path, "processed_path": str(destination.resolve()), "label_path": None})
    for cid, class_name in MORPHOLOGY_NAMES.items():
        column = class_name.lower()
        sample = morphology[morphology[column] > 0].sort_values("raw_path").iloc[0]
        destination = inspection_dir / f"morphology_{class_name.lower()}.jpg"
        clahe_image(sample.raw_path, destination)
        rows.append({"type": class_name, "raw_path": sample.raw_path, "processed_path": str(destination.resolve()), "label_path": sample.label_path})
    pd.DataFrame(rows).to_csv(prepared_dir / "inspection_manifest.csv", index=False)


def video_duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], text=True).strip())


def prepare_video_frames(data_dir, prepared_dir, participants):
    video_root = data_dir / "Video"
    t2dm = participants[participants.source.eq("T2DM")].copy()
    t2dm["normalized_participant"] = t2dm.participant.map(normalize_id)
    participant_map = t2dm.set_index("normalized_participant")[["participant", "subject_key", "split"]].to_dict("index")
    audit_rows = []
    for video in sorted(video_root.rglob("*.mpg")):
        candidate = normalize_id(video.parent.name)
        match = participant_map.get(candidate)
        audit_rows.append({"video_path": str(video.resolve()), "participant": match["participant"] if match else None, "subject_key": match["subject_key"] if match else None, "split": match["split"] if match else None, "matched": match is not None})
    video_audit = pd.DataFrame(audit_rows)
    video_audit.to_csv(prepared_dir / "video_participant_audit.csv", index=False)
    frame_rows = []
    for split in ["train", "val", "test"]:
        subset = video_audit[video_audit.matched & video_audit.split.eq(split)].reset_index(drop=True)
        for video_index, row in subset.iterrows():
            duration = video_duration(row.video_path)
            times = np.linspace(0.05 * duration, 0.95 * duration, 8)
            folder = prepared_dir / "video_frames" / split / f"video_{video_index:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            for frame_id, timestamp in enumerate(times, 1):
                destination = folder / f"frame_{frame_id:02d}.jpg"
                if not destination.exists():
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{timestamp:.3f}", "-i", row.video_path, "-frames:v", "1", "-q:v", "2", str(destination)], check=True)
                frame_rows.append({"video_path": row.video_path, "subject_key": row.subject_key, "participant": row.participant, "split": split, "frame_id": frame_id, "timestamp_sec": float(timestamp), "frame_path": str(destination.resolve())})
    frames = pd.DataFrame(frame_rows)
    frames.to_csv(prepared_dir / "video_frame_manifest.csv", index=False)
    for split in ["train", "val", "test"]:
        frames[frames.split.eq(split)].to_csv(prepared_dir / f"video_{split}_frame_manifest.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Capillary-Dataset")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    audit_dir = output_dir / "audit"
    split_dir = output_dir / "splits"
    prepared_dir = output_dir / "prepared"
    for directory in [audit_dir, split_dir, prepared_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_dataset(data_dir)
    audit_inventory(data_dir, audit_dir)
    classification, participants = build_classification_manifest(data_dir, split_dir)
    morphology = build_morphology_manifest(data_dir, split_dir, classification, participants)
    prepare_morphology_yolo(data_dir, prepared_dir, morphology)
    extract_morphology_crops(prepared_dir, morphology)
    build_inspection_manifest(prepared_dir, classification, morphology)
    prepare_video_frames(data_dir, prepared_dir, participants)
    print(f"Classification images: {len(classification):,}")
    print(f"Participants: {participants.subject_key.nunique():,}")
    print(f"Morphology images: {len(morphology):,}")
    print(f"Morphology objects: {morphology.objects.sum():,}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
