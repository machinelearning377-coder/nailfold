from pathlib import Path
from contextlib import nullcontext
import argparse
import copy
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import Dinov2Model
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

SEED = 42
MORPHOLOGY_CLASSES = ["Bushy", "Crossing", "Hairpin", "Tortuous"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


class SquarePad:
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, (left, top, right, bottom), fill=self.fill)


class PerImageStandardize:
    def __call__(self, tensor):
        standardized = (tensor - tensor.mean()) / tensor.std().clamp_min(1e-6)
        return (0.20 * standardized + 0.50).clamp(0, 1)


def disease_train_transform():
    return transforms.Compose([
        transforms.Grayscale(3),
        SquarePad(255),
        transforms.Resize((336, 336)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(3.9),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 0.8))], p=0.15),
        transforms.ToTensor(),
        PerImageStandardize(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.06)),
    ])


def disease_eval_transform():
    return transforms.Compose([
        transforms.Grayscale(3),
        SquarePad(255),
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        PerImageStandardize(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def morphology_train_transform():
    return transforms.Compose([
        SquarePad((255, 255, 255)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(3.9),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 0.8))], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.06)),
    ])


def morphology_eval_transform():
    return transforms.Compose([
        SquarePad((255, 255, 255)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MorphologyDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.class_to_id = {name: index for index, name in enumerate(MORPHOLOGY_CLASSES)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(row.crop_path).convert("RGB"))
        return image, self.class_to_id[row.class_name]


class DiseaseDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(row.raw_path).convert("RGB"))
        return image, int(row.label), row.subject_key


class VideoFrameDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.transform(Image.open(row.frame_path).convert("RGB")), row.video_path, int(row.frame_id), row.subject_key


class VideoSequenceDataset(Dataset):
    def __init__(self, frame, transform):
        self.groups = []
        for video_path, group in frame.groupby("video_path", sort=False):
            group = group.sort_values("frame_id")
            if len(group) == 8:
                self.groups.append((video_path, group.reset_index(drop=True)))
        self.transform = transform

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        video_path, group = self.groups[index]
        frames = torch.stack([self.transform(Image.open(path).convert("RGB")) for path in group.frame_path])
        return frames, video_path, group.subject_key.iloc[0]


class MorphologyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Linear(384, 4)

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        return self.classifier(hidden[:, 0])


class DiseaseModel(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Dropout(dropout), nn.Linear(768, 2))

    def representation(self, pixel_values):
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        cls = hidden[:, 0]
        patch_mean = hidden[:, 1:].mean(dim=1)
        return torch.cat([cls, patch_mean], dim=1), F.normalize(cls, dim=1)

    def forward(self, pixel_values):
        representation, _ = self.representation(pixel_values)
        return self.classifier(representation)


class TemporalModel(nn.Module):
    def __init__(self, input_dim=384, temporal_dim=256, frames=8):
        super().__init__()
        self.projection = nn.Linear(input_dim, temporal_dim)
        self.position = nn.Parameter(torch.randn(1, frames, temporal_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, temporal_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=temporal_dim, nhead=8, dim_feedforward=512, dropout=0.10, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.reconstruction = nn.Linear(temporal_dim, input_dim)

    def forward(self, sequence, mask):
        hidden = self.projection(sequence)
        hidden = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(hidden), hidden)
        hidden = self.transformer(hidden + self.position)
        return F.normalize(self.reconstruction(hidden), dim=-1)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8, dropout=0.10):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, values):
        return self.base(values) + self.b(self.dropout(self.a(values))) * self.scale


def apply_lora(model, rank=4, alpha=8, dropout=0.10):
    for block in model.backbone.encoder.layer[-4:]:
        attention = block.attention.attention
        attention.query = LoRALinear(attention.query, rank=rank, alpha=alpha, dropout=dropout)
        attention.value = LoRALinear(attention.value, rank=rank, alpha=alpha, dropout=dropout)
    return model


def participant_balanced_sampler(frame):
    images_per_subject = frame.groupby("subject_key").size()
    subjects_per_class = frame[["subject_key", "label"]].drop_duplicates().groupby("label").size()
    weights = [1.0 / (images_per_subject[row.subject_key] * subjects_per_class[row.label]) for row in frame.itertuples()]
    return WeightedRandomSampler(weights, len(frame), replacement=True)


def random_mask(batch_size, device):
    mask = torch.zeros(batch_size, 8, dtype=torch.bool, device=device)
    indexes = torch.rand(batch_size, 8, device=device).topk(2, largest=False).indices
    mask.scatter_(1, indexes, True)
    return mask


def fixed_mask(batch_size, device):
    mask = torch.zeros(batch_size, 8, dtype=torch.bool, device=device)
    mask[:, [2, 5]] = True
    return mask


def train_yolo(artifact_dir):
    from ultralytics import YOLO
    yaml_path = artifact_dir / "prepared" / "morphology_yolo.yaml"
    output_dir = artifact_dir / "models" / "yolo11s"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11s.pt")
    model.train(data=str(yaml_path), epochs=100, patience=20, imgsz=640, batch=16, optimizer="AdamW", lr0=1e-3, weight_decay=5e-4, pretrained=True, amp=True, cache=False, workers=4, degrees=5, translate=0.05, scale=0.10, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.20, mixup=0.0, seed=SEED, deterministic=True, project=str(output_dir.parent), name=output_dir.name, exist_ok=True, plots=False, save=True, verbose=True)


def train_morphology(artifact_dir, device):
    crop_manifest = pd.read_csv(artifact_dir / "prepared" / "morphology_crop_manifest.csv")
    train_frame = crop_manifest[crop_manifest.split.eq("train")].copy()
    val_frame = crop_manifest[crop_manifest.split.eq("val")].copy()
    counts = train_frame.class_name.value_counts().reindex(MORPHOLOGY_CLASSES)
    class_weights = 1.0 / np.sqrt(counts.values.astype(float))
    sample_weights = train_frame.class_name.map(dict(zip(MORPHOLOGY_CLASSES, class_weights))).values
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(train_frame), replacement=True)
    train_loader = DataLoader(MorphologyDataset(train_frame, morphology_train_transform()), batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(MorphologyDataset(val_frame, morphology_eval_transform()), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model = MorphologyModel().to(device)
    head_parameters = list(model.classifier.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW([{"params": backbone_parameters, "lr": 2e-5}, {"params": head_parameters, "lr": 1e-4}], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model_dir = artifact_dir / "models"
    history_dir = artifact_dir / "histories"
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(1, 26):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(labels)
        model.eval()
        validation_loss = 0.0
        labels_all = []
        probabilities_all = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast_context(device):
                    logits = model(images)
                    loss = criterion(logits, labels)
                validation_loss += loss.item() * len(labels)
                labels_all.extend(labels.cpu().numpy())
                probabilities_all.append(logits.softmax(1).cpu().numpy())
        labels_all = np.asarray(labels_all)
        probabilities_all = np.concatenate(probabilities_all)
        predictions = probabilities_all.argmax(1)
        per_class_f1 = []
        for class_id in range(4):
            true_positive = np.sum((labels_all == class_id) & (predictions == class_id))
            false_positive = np.sum((labels_all != class_id) & (predictions == class_id))
            false_negative = np.sum((labels_all == class_id) & (predictions != class_id))
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            per_class_f1.append(2 * precision * recall / max(precision + recall, 1e-12))
        macro_f1 = float(np.mean(per_class_f1))
        train_loss = total / len(train_frame)
        val_loss = validation_loss / len(val_frame)
        history.append([epoch, train_loss, val_loss, macro_f1])
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_dir / "morphology_dinov2.pt")
        scheduler.step()
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "val_macro_f1"]).to_csv(history_dir / "morphology_history.csv", index=False)
    (history_dir / "morphology_meta.json").write_text(json.dumps({"best_epoch": best_epoch, "selection_metric": "validation_loss"}, indent=2))


def load_morphology_backbone(checkpoint, device):
    morphology = MorphologyModel()
    morphology.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    backbone = copy.deepcopy(morphology.backbone)
    del morphology
    return backbone.to(device)


def evaluate_stage1_validation(model, loader, criterion, frame, device):
    model.eval()
    labels_all = []
    probabilities_all = []
    subjects_all = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels, subjects in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast_context(device):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            labels_all.extend(labels.cpu().numpy())
            probabilities_all.extend(logits.softmax(1)[:, 1].cpu().numpy())
            subjects_all.extend(subjects)
    image_frame = pd.DataFrame({"subject": subjects_all, "label": labels_all, "prob": probabilities_all})
    participant_frame = image_frame.groupby("subject", as_index=False).agg(label=("label", "first"), prob=("prob", "mean"))
    participant_pred = (participant_frame.prob >= 0.5).astype(int)
    return total_loss / len(frame), roc_auc_score(image_frame.label, image_frame.prob), roc_auc_score(participant_frame.label, participant_frame.prob), f1_score(participant_frame.label, participant_pred)


def train_stage1(artifact_dir, device):
    strict = pd.read_csv(artifact_dir / "splits" / "classification_strict_640_manifest.csv")
    train_frame = strict[strict.split.eq("train")].copy()
    val_frame = strict[strict.split.eq("val")].copy()
    train_loader = DataLoader(DiseaseDataset(train_frame, disease_train_transform()), batch_size=64, sampler=participant_balanced_sampler(train_frame), num_workers=4, pin_memory=True)
    val_loader = DataLoader(DiseaseDataset(val_frame, disease_eval_transform()), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = DiseaseModel(dropout=0.25)
    model.backbone.load_state_dict(load_morphology_backbone(artifact_dir / "models" / "morphology_dinov2.pt", torch.device("cpu")).state_dict())
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(1, 9):
        model.train()
        total = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(labels)
        val_loss, image_auc, participant_auc, participant_f1 = evaluate_stage1_validation(model, val_loader, criterion, val_frame, device)
        history.append([epoch, total / len(train_frame), val_loss, image_auc, participant_auc, participant_f1])
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), artifact_dir / "models" / "stage1_disease.pt")
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "image_auc", "participant_auc", "participant_f1"]).to_csv(artifact_dir / "histories" / "stage1_history.csv", index=False)
    (artifact_dir / "histories" / "stage1_meta.json").write_text(json.dumps({"best_epoch": best_epoch, "selection_metric": "validation_loss"}, indent=2))


def encode_video_frames(frame, model, transform, device):
    dataset = VideoFrameDataset(frame, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    records = []
    features = []
    model.eval()
    with torch.no_grad():
        for images, video_paths, frame_ids, subject_keys in loader:
            images = images.to(device)
            _, cls = model.representation(images)
            features.append(cls.cpu())
            for video_path, frame_id, subject_key in zip(video_paths, frame_ids.tolist(), subject_keys):
                records.append((video_path, frame_id, subject_key))
    feature_tensor = torch.cat(features)
    metadata = pd.DataFrame(records, columns=["video_path", "frame_id", "subject_key"])
    sequences = []
    for video_path, group in metadata.reset_index().groupby("video_path", sort=False):
        group = group.sort_values("frame_id")
        if len(group) == 8:
            indexes = group["index"].values
            sequences.append((feature_tensor[indexes], video_path, group.subject_key.iloc[0]))
    return sequences


class FeatureSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index][0]


def train_temporal(artifact_dir, device):
    train_frames = pd.read_csv(artifact_dir / "prepared" / "video_train_frame_manifest.csv")
    val_frames = pd.read_csv(artifact_dir / "prepared" / "video_val_frame_manifest.csv")
    image_model = DiseaseModel(dropout=0.25)
    image_model.load_state_dict(torch.load(artifact_dir / "models" / "stage1_disease.pt", map_location="cpu", weights_only=True))
    image_model = image_model.to(device).eval()
    for parameter in image_model.parameters():
        parameter.requires_grad = False
    train_sequences = encode_video_frames(train_frames, image_model, disease_eval_transform(), device)
    val_sequences = encode_video_frames(val_frames, image_model, disease_eval_transform(), device)
    train_loader = DataLoader(FeatureSequenceDataset(train_sequences), batch_size=16, shuffle=True)
    val_loader = DataLoader(FeatureSequenceDataset(val_sequences), batch_size=16, shuffle=False)
    model = TemporalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = []
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(1, 16):
        model.train()
        train_losses = []
        for sequence in train_loader:
            sequence = sequence.to(device)
            mask = random_mask(len(sequence), device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence, mask)
            cosine = F.cosine_similarity(prediction[mask], sequence[mask], dim=-1)
            loss = (1 - cosine).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        val_cosines = []
        with torch.no_grad():
            for sequence in val_loader:
                sequence = sequence.to(device)
                mask = fixed_mask(len(sequence), device)
                prediction = model(sequence, mask)
                cosine = F.cosine_similarity(prediction[mask], sequence[mask], dim=-1)
                val_losses.append((1 - cosine).mean().item())
                val_cosines.append(cosine.mean().item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_cosine = float(np.mean(val_cosines))
        history.append([epoch, train_loss, val_loss, val_cosine])
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), artifact_dir / "models" / "temporal_model.pt")
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "val_cosine"]).to_csv(artifact_dir / "histories" / "temporal_history.csv", index=False)
    (artifact_dir / "histories" / "temporal_meta.json").write_text(json.dumps({"best_epoch": best_epoch, "selection_metric": "validation_reconstruction_loss"}, indent=2))


def fusion_validation(model, temporal, image_loader, video_loader, criterion, image_frame, device):
    model.eval()
    labels_all = []
    probabilities_all = []
    subjects_all = []
    losses = []
    with torch.no_grad():
        for images, labels, subjects in image_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses.append(criterion(logits, labels).item() * len(labels))
            labels_all.extend(labels.cpu().numpy())
            probabilities_all.extend(logits.softmax(1)[:, 1].cpu().numpy())
            subjects_all.extend(subjects)
        video_cosines = []
        for videos, _, _ in video_loader:
            batch, frames, channels, height, width = videos.shape
            videos = videos.reshape(batch * frames, channels, height, width).to(device)
            _, cls = model.representation(videos)
            sequence = cls.reshape(batch, frames, -1)
            mask = fixed_mask(batch, device)
            prediction = temporal(sequence, mask)
            video_cosines.extend(F.cosine_similarity(prediction[mask], sequence[mask], dim=-1).cpu().numpy())
    image_predictions = pd.DataFrame({"subject": subjects_all, "label": labels_all, "prob": probabilities_all})
    image_class = (image_predictions.prob >= 0.5).astype(int)
    participant_predictions = image_predictions.groupby("subject", as_index=False).agg(label=("label", "first"), prob=("prob", "mean"))
    return sum(losses) / len(image_frame), roc_auc_score(image_predictions.label, image_predictions.prob), roc_auc_score(participant_predictions.label, participant_predictions.prob), balanced_accuracy_score(image_predictions.label, image_class), f1_score(image_predictions.label, image_class), float(np.mean(video_cosines))


def train_fusion(artifact_dir, device):
    strict = pd.read_csv(artifact_dir / "splits" / "classification_strict_640_manifest.csv")
    train_frame = strict[strict.split.eq("train")].copy()
    val_frame = strict[strict.split.eq("val")].copy()
    video_train = pd.read_csv(artifact_dir / "prepared" / "video_train_frame_manifest.csv")
    video_val = pd.read_csv(artifact_dir / "prepared" / "video_val_frame_manifest.csv")
    image_train_loader = DataLoader(DiseaseDataset(train_frame, disease_train_transform()), batch_size=64, sampler=participant_balanced_sampler(train_frame), num_workers=4, pin_memory=True)
    image_val_loader = DataLoader(DiseaseDataset(val_frame, disease_eval_transform()), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    video_train_loader = DataLoader(VideoSequenceDataset(video_train, disease_eval_transform()), batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    video_val_loader = DataLoader(VideoSequenceDataset(video_val, disease_eval_transform()), batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    model = DiseaseModel(dropout=0.25)
    model.load_state_dict(torch.load(artifact_dir / "models" / "stage1_disease.pt", map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = apply_lora(model, rank=4, alpha=8, dropout=0.10)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    model = model.to(device)
    temporal = TemporalModel().to(device)
    temporal.load_state_dict(torch.load(artifact_dir / "models" / "temporal_model.pt", map_location="cpu", weights_only=True))
    temporal.eval()
    for parameter in temporal.parameters():
        parameter.requires_grad = False
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=3e-5, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    history = []
    best_loss = np.inf
    best_epoch = 0
    video_iterator = iter(video_train_loader)
    for epoch in range(1, 16):
        model.train()
        total = 0.0
        for step, (images, labels, _) in enumerate(image_train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if step % 4 == 0:
                try:
                    videos, _, _ = next(video_iterator)
                except StopIteration:
                    video_iterator = iter(video_train_loader)
                    videos, _, _ = next(video_iterator)
                batch, frames, channels, height, width = videos.shape
                videos = videos.reshape(batch * frames, channels, height, width).to(device)
                _, cls = model.representation(videos)
                sequence = cls.reshape(batch, frames, -1)
                mask = random_mask(batch, device)
                reconstruction = temporal(sequence, mask)
                temporal_loss = (1 - F.cosine_similarity(reconstruction[mask], sequence.detach()[mask], dim=-1)).mean()
                loss = loss + 0.05 * temporal_loss
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, 0.5)
            optimizer.step()
            total += loss.item() * len(labels)
        val_loss, image_auc, participant_auc, image_balanced_accuracy, image_f1, video_cosine = fusion_validation(model, temporal, image_val_loader, video_val_loader, criterion, val_frame, device)
        history.append([epoch, total / len(train_frame), val_loss, image_auc, participant_auc, image_balanced_accuracy, image_f1, video_cosine])
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), artifact_dir / "models" / "fusion_lora.pt")
    history_frame = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "image_auc", "participant_auc", "image_balanced_accuracy", "image_f1", "video_cosine"])
    history_frame.to_csv(artifact_dir / "histories" / "fusion_history.csv", index=False)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    (artifact_dir / "histories" / "fusion_meta.json").write_text(json.dumps({"best_epoch": best_epoch, "selection_metric": "validation_classification_loss", "total_parameters": total_parameters, "trainable_parameters": trainable_count, "trainable_fraction_percent": 100 * trainable_count / total_parameters, "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.10, "video_loss_weight": 0.05, "video_every_image_batches": 4}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--stage", choices=["all", "yolo", "morphology", "stage1", "temporal", "fusion"], default="all")
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    (artifact_dir / "models").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "histories").mkdir(parents=True, exist_ok=True)
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stages = [args.stage] if args.stage != "all" else ["yolo", "morphology", "stage1", "temporal", "fusion"]
    for stage in stages:
        if stage == "yolo":
            train_yolo(artifact_dir)
        elif stage == "morphology":
            train_morphology(artifact_dir, device)
        elif stage == "stage1":
            train_stage1(artifact_dir, device)
        elif stage == "temporal":
            train_temporal(artifact_dir, device)
        elif stage == "fusion":
            train_fusion(artifact_dir, device)
    print(f"Completed stages: {', '.join(stages)}")


if __name__ == "__main__":
    main()
