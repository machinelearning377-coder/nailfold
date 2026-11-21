from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from methodology import MORPHOLOGY_CLASSES, MorphologyDataset, DiseaseDataset, MorphologyModel, DiseaseModel, TemporalModel, apply_lora, morphology_eval_transform, disease_eval_transform, encode_video_frames


def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def draw_boxes(axis, image_path, label_path):
    if not label_path or pd.isna(label_path) or not Path(label_path).exists():
        return
    with Image.open(image_path) as image:
        width, height = image.size
    for line in Path(label_path).read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        cid, xc, yc, bw, bh = map(float, line.split()[:5])
        x = (xc - bw / 2) * width
        y = (yc - bh / 2) * height
        rectangle = patches.Rectangle((x, y), bw * width, bh * height, fill=False, linewidth=1.1)
        axis.add_patch(rectangle)
        axis.text(x, max(0, y - 4), MORPHOLOGY_CLASSES[int(cid)], fontsize=6, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))


def figure03_dataset_samples(artifact_dir, figure_dir):
    classification = pd.read_csv(artifact_dir / "splits" / "classification_image_manifest.csv")
    morphology = pd.read_csv(artifact_dir / "splits" / "morphology_image_manifest.csv")
    video = pd.read_csv(artifact_dir / "prepared" / "video_train_frame_manifest.csv")
    t2dm = classification[classification.source.eq("T2DM")].sort_values(["participant", "raw_path"]).drop_duplicates("subject_key").head(4)
    healthy = classification[classification.source.eq("Healthy")].sort_values(["participant", "raw_path"]).drop_duplicates("subject_key").head(4)
    morphology_examples = []
    for class_name in MORPHOLOGY_CLASSES:
        column = class_name.lower()
        morphology_examples.append(morphology[morphology[column] > 0].sort_values("raw_path").iloc[0])
    first_video = video.video_path.iloc[0]
    video_examples = video[video.video_path.eq(first_video)].sort_values("frame_id").iloc[[0, 2, 5, 7]]
    figure, axes = plt.subplots(4, 4, figsize=(12, 11))
    for index, row in enumerate(t2dm.itertuples()):
        axes[0, index].imshow(Image.open(row.raw_path).convert("RGB"))
        axes[0, index].set_title(f"T2DM | Participant {row.participant}", fontsize=8)
        axes[0, index].axis("off")
    for index, row in enumerate(healthy.itertuples()):
        axes[1, index].imshow(Image.open(row.raw_path).convert("RGB"))
        axes[1, index].set_title(f"Healthy | Participant {row.participant}", fontsize=8)
        axes[1, index].axis("off")
    for index, row in enumerate(morphology_examples):
        axes[2, index].imshow(Image.open(row.raw_path).convert("RGB"))
        draw_boxes(axes[2, index], row.raw_path, row.label_path)
        axes[2, index].set_title(f"Morphology: {MORPHOLOGY_CLASSES[index]}", fontsize=8)
        axes[2, index].axis("off")
    for index, row in enumerate(video_examples.itertuples()):
        axes[3, index].imshow(Image.open(row.frame_path).convert("RGB"))
        axes[3, index].set_title(f"Video frame {row.frame_id}", fontsize=8)
        axes[3, index].axis("off")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig03_dataset_samples.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def figure04_classification_split(artifact_dir, figure_dir):
    summary = pd.read_csv(artifact_dir / "splits" / "classification_split_summary.csv")
    split_order = ["train", "val", "test"]
    class_order = ["T2DM", "Healthy"]
    participants = summary.pivot(index="split", columns="source", values="participants").reindex(split_order)[class_order]
    images = summary.pivot(index="split", columns="source", values="images").reindex(split_order)[class_order]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    participants.plot(kind="bar", stacked=True, ax=axes[0])
    images.plot(kind="bar", stacked=True, ax=axes[1])
    axes[0].set_title("Participant-Level Split")
    axes[0].set_xlabel("Dataset Split")
    axes[0].set_ylabel("Number of Participants")
    axes[1].set_title("Image Distribution Across Splits")
    axes[1].set_xlabel("Dataset Split")
    axes[1].set_ylabel("Number of Images")
    for axis in axes:
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig04_classification_split.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def figure05_morphology_split(artifact_dir, figure_dir):
    summary = pd.read_csv(artifact_dir / "splits" / "morphology_split_summary.csv", index_col=0).reindex(["train", "val", "test"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    summary["images"].plot(kind="bar", ax=axes[0])
    summary[["bushy", "crossing", "hairpin", "tortuous"]].plot(kind="bar", stacked=True, ax=axes[1])
    axes[0].set_title("Final Morphology Images by Split")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Images")
    axes[1].set_title("Final Morphology Object Distribution")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Annotated Capillaries")
    axes[1].legend(title="Morphology")
    for axis in axes:
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig05_morphology_split.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def figure06_preprocessing(artifact_dir, figure_dir):
    inspection = pd.read_csv(artifact_dir / "prepared" / "inspection_manifest.csv")
    order = ["T2DM", "Healthy", "Bushy", "Crossing", "Hairpin", "Tortuous"]
    inspection = inspection.set_index("type").loc[order].reset_index()
    figure, axes = plt.subplots(2, 6, figsize=(15, 5.2))
    for column, row in enumerate(inspection.itertuples()):
        raw = Image.open(row.raw_path).convert("RGB")
        processed = Image.open(row.processed_path).convert("RGB")
        axes[0, column].imshow(raw)
        axes[1, column].imshow(processed)
        if row.type not in {"T2DM", "Healthy"}:
            draw_boxes(axes[0, column], row.raw_path, row.label_path)
            draw_boxes(axes[1, column], row.raw_path, row.label_path)
        axes[0, column].set_title(row.type, fontsize=8)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Raw")
    axes[1, 0].set_ylabel("Preprocessed")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig06_preprocessing.png", dpi=600, bbox_inches="tight")
    plt.close(figure)


def morphology_predictions(model, frame, device):
    loader = DataLoader(MorphologyDataset(frame, morphology_eval_transform()), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    labels = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for images, target in loader:
            logits = model(images.to(device))
            labels.extend(target.numpy())
            probabilities.append(logits.softmax(1).cpu().numpy())
    labels = np.asarray(labels)
    probabilities = np.concatenate(probabilities)
    return labels, probabilities


def morphology_metrics(labels, probabilities):
    predictions = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=np.arange(4), zero_division=0)
    binary = label_binarize(labels, classes=np.arange(4))
    auroc = np.array([roc_auc_score(binary[:, index], probabilities[:, index]) for index in range(4)])
    return pd.DataFrame({"Class": MORPHOLOGY_CLASSES, "Precision": precision, "Recall": recall, "F1": f1, "AUROC": auroc, "Support": support})


def figures07_to_10_morphology(artifact_dir, figure_dir, results_dir, device):
    history = pd.read_csv(artifact_dir / "histories" / "morphology_history.csv")
    metadata = json.loads((artifact_dir / "histories" / "morphology_meta.json").read_text())
    crops = pd.read_csv(artifact_dir / "prepared" / "morphology_crop_manifest.csv")
    model = MorphologyModel().to(device)
    model.load_state_dict(torch.load(artifact_dir / "models" / "morphology_dinov2.pt", map_location=device, weights_only=True))
    val = crops[crops.split.eq("val")].copy()
    test = crops[crops.split.eq("test")].copy()
    val_labels, val_probabilities = morphology_predictions(model, val, device)
    test_labels, test_probabilities = morphology_predictions(model, test, device)
    val_metrics = morphology_metrics(val_labels, val_probabilities)
    test_metrics = morphology_metrics(test_labels, test_probabilities)
    val_metrics.to_csv(results_dir / "morphology_validation_metrics.csv", index=False)
    test_metrics.to_csv(results_dir / "morphology_test_metrics.csv", index=False)
    figure = plt.figure(figsize=(8, 5))
    plt.plot(history.epoch, history.train_loss, linewidth=2, label="Training Loss")
    plt.plot(history.epoch, history.val_loss, linewidth=2, label="Validation Loss")
    plt.axvline(metadata["best_epoch"], linestyle="--", linewidth=1.2, label=f"Best Epoch = {metadata['best_epoch']}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("DINOv2 Morphology Classifier — Training and Validation Loss")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig07_morphology_loss.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    figure = plt.figure(figsize=(7.2, 5.8))
    val_binary = label_binarize(val_labels, classes=np.arange(4))
    for index, class_name in enumerate(MORPHOLOGY_CLASSES):
        fpr, tpr, _ = roc_curve(val_binary[:, index], val_probabilities[:, index])
        auc = roc_auc_score(val_binary[:, index], val_probabilities[:, index])
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUROC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("DINOv2 Morphology Classifier — Validation ROC Curves")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig08_morphology_validation_roc.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    axis = val_metrics.set_index("Class")[["Precision", "Recall", "F1", "AUROC"]].plot(kind="bar", figsize=(9, 5), width=0.78)
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_title("Validation Performance Across Four Capillary Morphologies")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig09_morphology_validation_performance.png", dpi=600, bbox_inches="tight")
    plt.close()
    axis = test_metrics.set_index("Class")[["Precision", "Recall", "F1", "AUROC"]].plot(kind="bar", figsize=(9, 5), width=0.78)
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_title("Independent Test Performance Across Four Capillary Morphologies")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig10_morphology_test_performance.png", dpi=600, bbox_inches="tight")
    plt.close()
    return val_metrics, test_metrics


def disease_predictions(model, frame, device):
    loader = DataLoader(DiseaseDataset(frame, disease_eval_transform()), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    labels = []
    probabilities = []
    subjects = []
    model.eval()
    with torch.no_grad():
        for images, target, subject in loader:
            logits = model(images.to(device))
            labels.extend(target.numpy())
            probabilities.extend(logits.softmax(1)[:, 1].cpu().numpy())
            subjects.extend(subject)
    image_frame = pd.DataFrame({"subject": subjects, "label": labels, "prob": probabilities})
    image_frame["pred"] = (image_frame.prob >= 0.5).astype(int)
    participant_frame = image_frame.groupby("subject", as_index=False).agg(label=("label", "first"), prob=("prob", "mean"), images=("prob", "size"))
    participant_frame["pred"] = (participant_frame.prob >= 0.5).astype(int)
    return image_frame, participant_frame


def per_class_binary_metrics(frame):
    precision, recall, f1, support = precision_recall_fscore_support(frame.label, frame.pred, labels=[0, 1], zero_division=0)
    return pd.DataFrame({"Class": ["Healthy", "T2DM"], "Precision": precision, "Recall": recall, "F1": f1, "Support": support})


def figures11_to_15_stage1(artifact_dir, figure_dir, results_dir, device):
    history = pd.read_csv(artifact_dir / "histories" / "stage1_history.csv")
    strict = pd.read_csv(artifact_dir / "splits" / "classification_strict_640_manifest.csv")
    model = DiseaseModel(dropout=0.25).to(device)
    model.load_state_dict(torch.load(artifact_dir / "models" / "stage1_disease.pt", map_location=device, weights_only=True))
    val_image, val_participant = disease_predictions(model, strict[strict.split.eq("val")].copy(), device)
    test_image, test_participant = disease_predictions(model, strict[strict.split.eq("test")].copy(), device)
    val_image.to_csv(results_dir / "stage1_validation_image_predictions.csv", index=False)
    val_participant.to_csv(results_dir / "stage1_validation_participant_predictions.csv", index=False)
    test_image.to_csv(results_dir / "stage1_test_image_predictions.csv", index=False)
    test_participant.to_csv(results_dir / "stage1_test_participant_predictions.csv", index=False)
    val_per_class = per_class_binary_metrics(val_participant)
    test_per_class = per_class_binary_metrics(test_participant)
    val_per_class.to_csv(results_dir / "stage1_validation_per_class.csv", index=False)
    test_per_class.to_csv(results_dir / "stage1_test_per_class.csv", index=False)
    best_epoch = int(history.loc[history.val_loss.idxmin(), "epoch"])
    figure = plt.figure(figsize=(8, 5))
    plt.plot(history.epoch, history.train_loss, linewidth=2, label="Training Loss")
    plt.plot(history.epoch, history.val_loss, linewidth=2, label="Validation Loss")
    plt.axvline(best_epoch, linestyle="--", linewidth=1.2, label=f"Best Epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage-1 Image Model — Training and Validation Loss")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig11_stage1_loss.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    val_image_auc = roc_auc_score(val_image.label, val_image.prob)
    val_participant_auc = roc_auc_score(val_participant.label, val_participant.prob)
    fpr_i, tpr_i, _ = roc_curve(val_image.label, val_image.prob)
    fpr_p, tpr_p, _ = roc_curve(val_participant.label, val_participant.prob)
    figure = plt.figure(figsize=(7.2, 5.5))
    plt.plot(fpr_i, tpr_i, linewidth=2, label=f"Image ROC (AUC = {val_image_auc:.3f})")
    plt.plot(fpr_p, tpr_p, linewidth=2, label=f"Participant ROC (AUC = {val_participant_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stage-1 Image Model — Validation ROC Curves")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig12_stage1_validation_roc.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    axis = val_per_class.set_index("Class")[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(8, 5), width=0.72)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.set_title("Stage-1 Image Model — Validation Per-Class Performance")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig13_stage1_validation_performance.png", dpi=600, bbox_inches="tight")
    plt.close()
    test_image_auc = roc_auc_score(test_image.label, test_image.prob)
    test_participant_auc = roc_auc_score(test_participant.label, test_participant.prob)
    fpr_i, tpr_i, _ = roc_curve(test_image.label, test_image.prob)
    fpr_p, tpr_p, _ = roc_curve(test_participant.label, test_participant.prob)
    figure = plt.figure(figsize=(7.2, 5.5))
    plt.plot(fpr_i, tpr_i, linewidth=2, label=f"Image AUC = {test_image_auc:.3f}")
    plt.plot(fpr_p, tpr_p, linewidth=2, label=f"Participant AUC = {test_participant_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stage-1 Image Model — Independent Test ROC")
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig14_stage1_test_roc.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    axis = test_per_class.set_index("Class")[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(8, 5), width=0.72)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.set_title("Stage-1 Image Model — Independent Test Performance")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig15_stage1_test_performance.png", dpi=600, bbox_inches="tight")
    plt.close()
    return {"val_image_auc": val_image_auc, "val_participant_auc": val_participant_auc, "test_image_auc": test_image_auc, "test_participant_auc": test_participant_auc}


def temporal_scores(sequences, temporal, device):
    position_scores = [[] for _ in range(8)]
    video_rows = []
    temporal.eval()
    with torch.no_grad():
        for sequence, video_path, subject_key in sequences:
            sequence = sequence.unsqueeze(0).to(device)
            scores = []
            for position in range(8):
                mask = torch.zeros(1, 8, dtype=torch.bool, device=device)
                mask[:, position] = True
                prediction = temporal(sequence, mask)
                score = F.cosine_similarity(prediction[:, position], sequence[:, position], dim=-1).item()
                scores.append(score)
                position_scores[position].append(score)
            video_rows.append({"video_path": video_path, "subject_key": subject_key, "mean_similarity": float(np.mean(scores)), "min_similarity": float(np.min(scores)), "max_similarity": float(np.max(scores))})
    position_frame = pd.DataFrame({"Frame": [f"Frame {index}" for index in range(1, 9)], "Cosine_Similarity": [float(np.mean(values)) for values in position_scores]})
    return position_frame, pd.DataFrame(video_rows)


def figures16_to_20_temporal(artifact_dir, figure_dir, results_dir, device):
    history = pd.read_csv(artifact_dir / "histories" / "temporal_history.csv")
    image_model = DiseaseModel(dropout=0.25).to(device)
    image_model.load_state_dict(torch.load(artifact_dir / "models" / "stage1_disease.pt", map_location=device, weights_only=True))
    image_model.eval()
    for parameter in image_model.parameters():
        parameter.requires_grad = False
    temporal = TemporalModel().to(device)
    temporal.load_state_dict(torch.load(artifact_dir / "models" / "temporal_model.pt", map_location=device, weights_only=True))
    val_frames = pd.read_csv(artifact_dir / "prepared" / "video_val_frame_manifest.csv")
    test_frames = pd.read_csv(artifact_dir / "prepared" / "video_test_frame_manifest.csv")
    val_sequences = encode_video_frames(val_frames, image_model, disease_eval_transform(), device)
    test_sequences = encode_video_frames(test_frames, image_model, disease_eval_transform(), device)
    val_positions, val_videos = temporal_scores(val_sequences, temporal, device)
    test_positions, test_videos = temporal_scores(test_sequences, temporal, device)
    val_positions.to_csv(results_dir / "temporal_validation_frame_position.csv", index=False)
    val_videos.to_csv(results_dir / "temporal_validation_video.csv", index=False)
    test_positions.to_csv(results_dir / "temporal_test_frame_position.csv", index=False)
    test_videos.to_csv(results_dir / "temporal_test_video.csv", index=False)
    figure = plt.figure(figsize=(8, 5))
    plt.plot(history.epoch, history.train_loss, linewidth=2, label="Training Loss")
    plt.plot(history.epoch, history.val_loss, linewidth=2, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked Reconstruction Loss")
    plt.title("Video Temporal Model — Training and Validation Loss")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig16_temporal_loss.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    axis = val_positions.plot(x="Frame", y="Cosine_Similarity", kind="bar", figsize=(8.5, 5), legend=False)
    axis.axhline(val_positions.Cosine_Similarity.mean(), linestyle="--", label=f"Mean = {val_positions.Cosine_Similarity.mean():.3f}")
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_ylabel("Cosine Similarity")
    axis.set_title("Video Temporal Model — Validation Reconstruction Performance")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig17_temporal_validation_frame.png", dpi=600, bbox_inches="tight")
    plt.close()
    figure = plt.figure(figsize=(9, 5))
    plt.bar(range(1, len(val_videos) + 1), val_videos.mean_similarity)
    plt.axhline(val_videos.mean_similarity.mean(), linestyle="--", label=f"Mean = {val_videos.mean_similarity.mean():.3f}")
    plt.ylim(0, 1)
    plt.xlabel("Validation Video")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Video Temporal Model — Video-Level Validation Robustness")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig18_temporal_validation_video.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    axis = test_positions.plot(x="Frame", y="Cosine_Similarity", kind="bar", figsize=(8.5, 5), legend=False)
    axis.axhline(test_positions.Cosine_Similarity.mean(), linestyle="--", label=f"Mean = {test_positions.Cosine_Similarity.mean():.3f}")
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_ylabel("Cosine Similarity")
    axis.set_title("Video Temporal Model — Held-Out Test Performance")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig19_temporal_test_frame.png", dpi=600, bbox_inches="tight")
    plt.close()
    figure = plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(test_videos) + 1), test_videos.mean_similarity)
    plt.axhline(test_videos.mean_similarity.mean(), linestyle="--", label=f"Mean = {test_videos.mean_similarity.mean():.3f}")
    plt.ylim(0, 1)
    plt.xlabel("Test Video")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Video Temporal Model — Held-Out Video Robustness")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig20_temporal_test_video.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    return float(val_videos.mean_similarity.mean()), float(test_videos.mean_similarity.mean())


def figures21_to_22_final(artifact_dir, figure_dir, results_dir, device):
    strict = pd.read_csv(artifact_dir / "splits" / "classification_strict_640_manifest.csv")
    test = strict[strict.split.eq("test")].copy()
    model = DiseaseModel(dropout=0.25)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = apply_lora(model, rank=4, alpha=8, dropout=0.10)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model.load_state_dict(torch.load(artifact_dir / "models" / "fusion_lora.pt", map_location="cpu", weights_only=True))
    model = model.to(device)
    image_predictions, participant_predictions = disease_predictions(model, test, device)
    image_predictions.to_csv(results_dir / "fusion_test_image_predictions.csv", index=False)
    participant_predictions.to_csv(results_dir / "fusion_test_participant_predictions.csv", index=False)
    image_auc = roc_auc_score(image_predictions.label, image_predictions.prob)
    participant_auc = roc_auc_score(participant_predictions.label, participant_predictions.prob)
    accuracy = accuracy_score(image_predictions.label, image_predictions.pred)
    balanced_accuracy = balanced_accuracy_score(image_predictions.label, image_predictions.pred)
    precision = precision_score(image_predictions.label, image_predictions.pred, zero_division=0)
    recall = recall_score(image_predictions.label, image_predictions.pred, zero_division=0)
    f1 = f1_score(image_predictions.label, image_predictions.pred, zero_division=0)
    metrics = pd.DataFrame({"Metric": ["Image AUROC", "Participant AUROC", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"], "Value": [image_auc, participant_auc, accuracy, balanced_accuracy, precision, recall, f1]})
    metrics.to_csv(results_dir / "fusion_final_metrics.csv", index=False)
    metadata = json.loads((artifact_dir / "histories" / "fusion_meta.json").read_text())
    full_parameters = float(metadata["total_parameters"])
    trainable_parameters = float(metadata["trainable_parameters"])
    figure = plt.figure(figsize=(7.5, 5))
    bars = plt.bar(["Full Fine-Tuning", "LoRA"], [full_parameters / 1e6, trainable_parameters / 1e6])
    plt.ylabel("Trainable Parameters (millions)")
    plt.title("Parameter Efficiency of LoRA Adaptation")
    plt.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, [full_parameters / 1e6, trainable_parameters / 1e6]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}M", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig21_parameter_efficiency.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    labels = ["Image\nAUROC", "Participant\nAUROC", "Accuracy", "Balanced\nAccuracy", "Precision", "Recall", "F1-score"]
    values = metrics.Value.to_numpy()
    figure = plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Performance Score")
    plt.title("Fusion + LoRA — Independent Test Performance")
    plt.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(figure_dir / "Fig22_final_performance.png", dpi=600, bbox_inches="tight")
    plt.close(figure)
    return dict(zip(metrics.Metric, metrics.Value))


def save_principal_results(results_dir, morphology_validation, stage1, temporal_validation, temporal_test, final_metrics):
    rows = []
    for row in morphology_validation.itertuples():
        rows.append({"Experimental Stage": f"Morphology validation AUROC — {row.Class}", "Principal Result": row.AUROC})
    rows.extend([
        {"Experimental Stage": "Stage-1 validation image AUROC", "Principal Result": stage1["val_image_auc"]},
        {"Experimental Stage": "Stage-1 validation participant AUROC", "Principal Result": stage1["val_participant_auc"]},
        {"Experimental Stage": "Stage-1 test image AUROC", "Principal Result": stage1["test_image_auc"]},
        {"Experimental Stage": "Stage-1 test participant AUROC", "Principal Result": stage1["test_participant_auc"]},
        {"Experimental Stage": "Temporal validation cosine similarity", "Principal Result": temporal_validation},
        {"Experimental Stage": "Temporal held-out cosine similarity", "Principal Result": temporal_test},
    ])
    for metric, value in final_metrics.items():
        rows.append({"Experimental Stage": f"Fusion + LoRA {metric}", "Principal Result": value})
    pd.DataFrame(rows).to_csv(results_dir / "principal_experimental_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    figure_dir = ensure_directory(artifact_dir / "paper_figures")
    results_dir = ensure_directory(artifact_dir / "evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figure03_dataset_samples(artifact_dir, figure_dir)
    figure04_classification_split(artifact_dir, figure_dir)
    figure05_morphology_split(artifact_dir, figure_dir)
    figure06_preprocessing(artifact_dir, figure_dir)
    morphology_validation, _ = figures07_to_10_morphology(artifact_dir, figure_dir, results_dir, device)
    stage1 = figures11_to_15_stage1(artifact_dir, figure_dir, results_dir, device)
    temporal_validation, temporal_test = figures16_to_20_temporal(artifact_dir, figure_dir, results_dir, device)
    final_metrics = figures21_to_22_final(artifact_dir, figure_dir, results_dir, device)
    save_principal_results(results_dir, morphology_validation, stage1, temporal_validation, temporal_test, final_metrics)
    print(f"Figures: {figure_dir}")
    print(f"Evaluation outputs: {results_dir}")


if __name__ == "__main__":
    main()
