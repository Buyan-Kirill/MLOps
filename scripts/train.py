#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from tqdm import tqdm
import sys
import random
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.encoder import BookEncoderModel, BookEncoderConfig
from src.metrics import jaccard
from src.utils import parse_genre_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ContrastiveBookDataset(Dataset):
    def __init__(
        self,
        book_embeddings: np.ndarray,
        books_meta: pd.DataFrame,
        desc_original: np.ndarray,
        num_negatives: int = 3,
        genre_jaccard_threshold: float = 0.2,
        desc_sim_threshold: float = 0.3,
        seed: int = 42,
    ):
        self.books_meta = books_meta.copy()
        self.books_meta["series_clean"] = (
            self.books_meta["series_clean"].fillna("").astype(str)
        )
        self.books_meta["author_clean"] = (
            self.books_meta["author_clean"].fillna("").astype(str)
        )
        self.books_meta = self.books_meta.reset_index(drop=True)

        self.embeddings = torch.tensor(book_embeddings, dtype=torch.float32)
        self.desc_original = torch.tensor(desc_original, dtype=torch.float32)
        self.num_negatives = num_negatives
        self.genre_jaccard_threshold = genre_jaccard_threshold
        self.desc_sim_threshold = desc_sim_threshold

        self.rng = np.random.default_rng(seed)

        self.author_to_indices = (
            self.books_meta.groupby("author_clean", group_keys=False)
            .apply(lambda x: x.index.tolist(), include_groups=False)
            .to_dict()
        )
        non_empty_series = self.books_meta[
            self.books_meta["series_clean"].str.strip() != ""
        ]
        self.series_to_indices = (
            non_empty_series.groupby("series_clean", group_keys=False)
            .apply(lambda x: x.index.tolist(), include_groups=False)
            .to_dict()
        )
        self.all_indices = list(range(len(self.books_meta)))
        self.genres_sets = [parse_genre_str(g) for g in self.books_meta["genres_list"]]

    def _get_unrelated_negative(self, idx: int) -> int:
        anchor_author = self.books_meta.loc[idx, "author_clean"]
        anchor_series = self.books_meta.loc[idx, "series_clean"]
        anchor_genres = self.genres_sets[idx]
        anchor_desc = self.desc_original[idx]
        n = len(self.books_meta)
        for _ in range(100):
            j = self.rng.integers(0, n)
            if j == idx:
                continue
            if self.books_meta.loc[j, "author_clean"] == anchor_author:
                continue
            j_series = self.books_meta.loc[j, "series_clean"]
            if anchor_series.strip() and j_series.strip() and anchor_series == j_series:
                continue
            j_genres = self.genres_sets[j]
            if jaccard(anchor_genres, j_genres) > self.genre_jaccard_threshold:
                continue
            desc_sim = F.cosine_similarity(
                anchor_desc.unsqueeze(0), self.desc_original[j].unsqueeze(0)
            ).item()
            if desc_sim > self.desc_sim_threshold:
                continue
            return j
        other_author = [
            j
            for j in self.all_indices
            if self.books_meta.loc[j, "author_clean"] != anchor_author
        ]
        return self.rng.choice(other_author) if other_author else idx

    def __len__(self) -> int:
        return len(self.books_meta)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        anchor = self.embeddings[idx]
        author = self.books_meta.loc[idx, "author_clean"]
        series = self.books_meta.loc[idx, "series_clean"]

        pos_series = idx
        if (
            series.strip()
            and series in self.series_to_indices
            and len(self.series_to_indices[series]) > 1
        ):
            candidates = [i for i in self.series_to_indices[series] if i != idx]
            if candidates:
                pos_series = self.rng.choice(candidates)

        pos_author = idx
        if author in self.author_to_indices and len(self.author_to_indices[author]) > 1:
            candidates = [i for i in self.author_to_indices[author] if i != idx]
            if candidates:
                pos_author = self.rng.choice(candidates)

        genre_candidates = [
            j
            for j in self.all_indices
            if j != idx and self.genres_sets[idx] & self.genres_sets[j]
        ]
        pos_genre = self.rng.choice(genre_candidates) if genre_candidates else idx

        desc_sim = F.cosine_similarity(
            self.desc_original[idx : idx + 1], self.desc_original
        )
        desc_candidates = torch.where(desc_sim >= 0.5)[0].cpu().numpy()
        desc_candidates = [j for j in desc_candidates if j != idx]
        pos_desc = self.rng.choice(desc_candidates) if desc_candidates else idx

        neg_idxs = [
            self._get_unrelated_negative(idx) for _ in range(self.num_negatives)
        ]
        negatives = self.embeddings[neg_idxs]
        return (
            anchor,
            self.embeddings[pos_series],
            self.embeddings[pos_author],
            self.embeddings[pos_genre],
            self.embeddings[pos_desc],
            negatives,
        )


def weighted_contrastive_loss(
    anchor: torch.Tensor,
    pos_s: torch.Tensor,
    pos_a: torch.Tensor,
    pos_g: torch.Tensor,
    pos_d: torch.Tensor,
    negatives: torch.Tensor,
    w_series: float = 1.0,
    w_author: float = 0.8,
    w_genre: float = 0.5,
    w_desc: float = 0.6,
    margin: float = 0.2,
) -> torch.Tensor:
    def triplet_loss(pos: torch.Tensor) -> torch.Tensor:
        pos_dist = 1 - F.cosine_similarity(anchor, pos, dim=-1)
        neg_dists = 1 - F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)
        hardest_neg = neg_dists.min(dim=1)[0]
        return F.relu(pos_dist - hardest_neg + margin).mean()

    return (
        w_series * triplet_loss(pos_s)
        + w_author * triplet_loss(pos_a)
        + w_genre * triplet_loss(pos_g)
        + w_desc * triplet_loss(pos_d)
    )


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    data_seed = config.get("data_seed", seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.info("GPU детерминизм включён (cudnn.deterministic=True)")

    logger.info(f"seed={seed}, data_seed={data_seed}")

    INPUT_DIR = config["paths"]["processed_data_dir"]
    OUTPUT_DIR = config["paths"]["outputs_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)

    logger.info("Загрузка данных...")
    book_embeddings = np.load(os.path.join(INPUT_DIR, "book_embeddings_multimodal.npy"))
    books_meta = pd.read_csv(
        os.path.join(INPUT_DIR, "books_meta_multimodal.csv"),
        keep_default_na=False,
        na_values=[],
    )
    desc_original = np.load(os.path.join(INPUT_DIR, "book_descriptions_original.npy"))

    dataset = ContrastiveBookDataset(
        book_embeddings,
        books_meta,
        desc_original,
        num_negatives=config["training"]["num_negatives"],
        genre_jaccard_threshold=config["training"]["genre_jaccard_threshold"],
        desc_sim_threshold=config["training"]["description_sim_threshold"],
        seed=data_seed,
    )

    g = torch.Generator()
    g.manual_seed(data_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")

    config_model = BookEncoderConfig(
        input_dim=book_embeddings.shape[1],
        hidden_dim=config["training"]["hidden_dim"],
        output_dim=config["training"]["output_dim"],
    )
    model = BookEncoderModel(config_model).to(device)

    encoder_path = os.path.join(
        OUTPUT_DIR,
        f"book_encoder_contrastive_{config['training']['output_dim']}",
    )
    if os.path.exists(os.path.join(encoder_path, "pytorch_model.bin")):
        logger.info(f"Загружаем чекпоинт: {encoder_path}")
        model = BookEncoderModel.from_pretrained(encoder_path).to(device)
    else:
        logger.info("Начинаем обучение с нуля.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    total_steps = len(dataloader) * config["training"]["epochs"]
    warmup_steps = len(dataloader) * config["training"]["warmup_epochs"]
    main_steps = total_steps - warmup_steps

    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=float(config["training"]["start_factor"]),
        end_factor=float(config["training"]["end_factor"]),
        total_iters=warmup_steps,
    )
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=main_steps,
        eta_min=float(config["training"].get("eta_min", 1e-6)),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_steps],
    )

    logger.info("Обучение...")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch + 1}")
        for batch in pbar:
            anchor, pos_s, pos_a, pos_g, pos_d, negatives = [
                x.to(device, non_blocking=True) for x in batch
            ]
            optimizer.zero_grad()
            a_emb = model(anchor)
            ps_emb = model(pos_s)
            pa_emb = model(pos_a)
            pg_emb = model(pos_g)
            pd_emb = model(pos_d)
            n_emb = model(negatives)
            w = config["training"]["weights"]
            loss = weighted_contrastive_loss(
                a_emb,
                ps_emb,
                pa_emb,
                pg_emb,
                pd_emb,
                n_emb,
                w_series=w["series"],
                w_author=w["author"],
                w_genre=w["genre"],
                w_desc=w["desc"],
                margin=config["training"]["margin"],
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

    model.eval()
    model.save_pretrained(encoder_path)
    logger.info(f"Модель сохранена: {encoder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
