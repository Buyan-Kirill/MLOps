#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import sys
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List, Set

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import parse_genre_str
from src.metrics import jaccard, spearman_rank_correlation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/compare.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def compute_detailed_stats(
    embeddings: np.ndarray,
    books_meta: pd.DataFrame,
    desc_original: Optional[np.ndarray] = None,
    sample_pairs: int = 2000,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    books_meta = books_meta.reset_index(drop=True)
    n = len(books_meta)
    if n < 2:
        return {
            k: float("nan")
            for k in [
                "author_intra",
                "author_inter",
                "series_intra",
                "series_inter",
                "genre_corr",
                "desc_corr",
            ]
        }

    author_intra = []
    author_groups = list(
        books_meta.groupby("author_clean", group_keys=False)
    )
    for author, group in tqdm(
        author_groups, desc="Авторы (внутри)", leave=False
    ):
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        if len(idxs) > 20:
            idxs = rng.choice(idxs, 20, replace=False)
        sim = cos_sim(
            torch.tensor(embeddings[idxs], dtype=torch.float32),
            torch.tensor(embeddings[idxs], dtype=torch.float32),
        ).cpu().numpy()
        triu = np.triu_indices_from(sim, k=1)
        author_intra.extend(sim[triu])
    avg_author_intra = (
        np.mean(author_intra) if author_intra else float("nan")
    )

    author_inter = []
    authors = books_meta["author_clean"].unique()
    for _ in tqdm(
        range(sample_pairs), desc="Авторы (между)", leave=False
    ):
        if len(authors) < 2:
            break
        a1, a2 = rng.choice(authors, 2, replace=False)
        g1 = books_meta[books_meta["author_clean"] == a1]
        g2 = books_meta[books_meta["author_clean"] == a2]
        if len(g1) == 0 or len(g2) == 0:
            continue
        i = rng.choice(g1.index)
        j = rng.choice(g2.index)
        sim = cos_sim(
            torch.tensor(embeddings[i:i + 1], dtype=torch.float32),
            torch.tensor(embeddings[j:j + 1], dtype=torch.float32),
        ).cpu().item()
        author_inter.append(sim)
    avg_author_inter = (
        np.mean(author_inter) if author_inter else float("nan")
    )

    series_intra = []
    series_books = books_meta[
        books_meta["series_clean"].str.strip() != ""
    ]
    series_groups = list(
        series_books.groupby("series_clean", group_keys=False)
    )
    for series, group in tqdm(
        series_groups, desc="Серии (внутри)", leave=False
    ):
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        if len(idxs) > 10:
            idxs = rng.choice(idxs, 10, replace=False)
        sim = cos_sim(
            torch.tensor(embeddings[idxs], dtype=torch.float32),
            torch.tensor(embeddings[idxs], dtype=torch.float32),
        ).cpu().numpy()
        triu = np.triu_indices_from(sim, k=1)
        series_intra.extend(sim[triu])
    avg_series_intra = (
        np.mean(series_intra) if series_intra else float("nan")
    )

    series_inter = []
    series_list = series_books["series_clean"].unique()
    for _ in tqdm(
        range(sample_pairs), desc="Серии (между)", leave=False
    ):
        if len(series_list) < 2:
            break
        s1, s2 = rng.choice(series_list, 2, replace=False)
        g1 = series_books[series_books["series_clean"] == s1]
        g2 = series_books[series_books["series_clean"] == s2]
        if len(g1) == 0 or len(g2) == 0:
            continue
        i = rng.choice(g1.index)
        j = rng.choice(g2.index)
        sim = cos_sim(
            torch.tensor(embeddings[i:i + 1], dtype=torch.float32),
            torch.tensor(embeddings[j:j + 1], dtype=torch.float32),
        ).cpu().item()
        series_inter.append(sim)
    avg_series_inter = (
        np.mean(series_inter) if series_inter else float("nan")
    )

    genre_orig, genre_emb = [], []
    for _ in tqdm(range(sample_pairs), desc="Жанры", leave=False):
        i, j = rng.choice(len(books_meta), 2, replace=False)
        g1 = parse_genre_str(books_meta.iloc[i]["genres_list"])
        g2 = parse_genre_str(books_meta.iloc[j]["genres_list"])
        genre_orig.append(jaccard(g1, g2))
        emb_sim = cos_sim(
            torch.tensor(embeddings[i:i + 1], dtype=torch.float32),
            torch.tensor(embeddings[j:j + 1], dtype=torch.float32),
        ).cpu().item()
        genre_emb.append(emb_sim)
    corr_genre = float("nan")
    if len(genre_emb) > 5:
        try:
            corr_genre = spearman_rank_correlation(genre_orig, genre_emb)
        except Exception:
            corr_genre = float("nan")

    desc_orig_sims, desc_emb_sims = [], []
    for _ in tqdm(range(sample_pairs), desc="Описание", leave=False):
        i, j = rng.choice(len(books_meta), 2, replace=False)
        if (
            not books_meta.iloc[i]["description_clean"]
            or not books_meta.iloc[j]["description_clean"]
        ):
            continue
        orig_sim = cos_sim(
            torch.tensor(desc_original[i:i + 1], dtype=torch.float32),
            torch.tensor(desc_original[j:j + 1], dtype=torch.float32),
        ).cpu().item()
        emb_sim = cos_sim(
            torch.tensor(embeddings[i:i + 1], dtype=torch.float32),
            torch.tensor(embeddings[j:j + 1], dtype=torch.float32),
        ).cpu().item()
        desc_orig_sims.append(orig_sim)
        desc_emb_sims.append(emb_sim)
    corr_desc = float("nan")
    if len(desc_emb_sims) > 5:
        try:
            corr_desc = spearman_rank_correlation(desc_orig_sims, desc_emb_sims)
        except Exception:
            corr_desc = float("nan")

    return {
        "author_intra": avg_author_intra,
        "author_inter": avg_author_inter,
        "series_intra": avg_series_intra,
        "series_inter": avg_series_inter,
        "genre_corr": corr_genre,
        "desc_corr": corr_desc,
    }


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sample_size = config["compare_embeddings"]["books_number"]

    INPUT_DIR = config["paths"]["processed_data_dir"]
    OUTPUT_DIR = config["paths"]["outputs_dir"]
    OUTPUT_DIM = config["training"]["output_dim"]

    logger.info("Загрузка данных...")
    books_meta = pd.read_csv(
        os.path.join(INPUT_DIR, "books_meta_multimodal.csv"),
        keep_default_na=False,
        na_values=[],
    )
    embeddings_original = np.load(
        os.path.join(INPUT_DIR, "book_embeddings_multimodal.npy")
    )
    embeddings_contrastive_path = os.path.join(
        OUTPUT_DIR,
        f"book_encoder_contrastive_{OUTPUT_DIM}",
        f"book_embeddings_contrastive_{OUTPUT_DIM}.npy"
    )
    if not os.path.exists(embeddings_contrastive_path):
        logger.error(
            "Контрастивные эмбеддинги не найдены: %s",
            embeddings_contrastive_path
        )
        logger.error("Сначала выполните: make train")
        return
    embeddings_contrastive = np.load(embeddings_contrastive_path)
    desc_original = np.load(
        os.path.join(INPUT_DIR, "book_descriptions_original.npy")
    )

    min_len = min(
        len(books_meta),
        len(embeddings_original),
        len(embeddings_contrastive),
        len(desc_original),
    )
    idx = np.random.default_rng(42).choice(
        min_len, size=min(sample_size, min_len), replace=False
    )

    books_meta = books_meta.iloc[idx].reset_index(drop=True)
    embeddings_original = embeddings_original[idx]
    embeddings_contrastive = embeddings_contrastive[idx]
    desc_original = desc_original[idx]
    sample_pairs = config["compare_embeddings"]["pairs_number"]

    logger.info("Оценка оригинальных эмбеддингов...")
    orig = compute_detailed_stats(
        embeddings_original,
        books_meta,
        desc_original,
        sample_pairs=sample_pairs,
    )

    logger.info("Оценка контрастивных эмбеддингов...")
    cont = compute_detailed_stats(
        embeddings_contrastive,
        books_meta,
        desc_original,
        sample_pairs=sample_pairs,
    )

    logger.info("\n" + "=" * 90)
    logger.info("ДЕТАЛЬНОЕ СРАВНЕНИЕ: ВНУТРИГРУППОВОЕ vs МЕЖГРУППОВОЕ СХОДСТВО")
    logger.info("=" * 90)
    logger.info(f"{'Метрика':<25} {'До':<8} {'После':<8} {'Δ':<8}")
    logger.info("-" * 90)

    def fmt(x):
        return f"{x:.4f}" if not np.isnan(x) else "  nan"

    def delta(o, c):
        d = c - o
        return (
            f"{d:+.4f}"
            if not (np.isnan(o) or np.isnan(c))
            else "  nan"
        )

    logger.info(
        f"{'Автор (внутри)':<25} "
        f"{fmt(orig['author_intra']):<8} "
        f"{fmt(cont['author_intra']):<8} "
        f"{delta(orig['author_intra'], cont['author_intra']):<8}"
    )
    logger.info(
        f"{'Автор (между)':<25} "
        f"{fmt(orig['author_inter']):<8} "
        f"{fmt(cont['author_inter']):<8} "
        f"{delta(orig['author_inter'], cont['author_inter']):<8}"
    )
    logger.info(
        f"{'Серия (внутри)':<25} "
        f"{fmt(orig['series_intra']):<8} "
        f"{fmt(cont['series_intra']):<8} "
        f"{delta(orig['series_intra'], cont['series_intra']):<8}"
    )
    logger.info(
        f"{'Серия (между)':<25} "
        f"{fmt(orig['series_inter']):<8} "
        f"{fmt(cont['series_inter']):<8} "
        f"{delta(orig['series_inter'], cont['series_inter']):<8}"
    )
    logger.info(
        f"{'Жанры (корреляция)':<25} "
        f"{fmt(orig['genre_corr']):<8} "
        f"{fmt(cont['genre_corr']):<8} "
        f"{delta(orig['genre_corr'], cont['genre_corr']):<8}"
    )
    logger.info(
        f"{'Описание (корреляция)':<25} "
        f"{fmt(orig['desc_corr']):<8} "
        f"{fmt(cont['desc_corr']):<8} "
        f"{delta(orig['desc_corr'], cont['desc_corr']):<8}"
    )

    logger.info("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml"
    )
    args = parser.parse_args()
    main(args.config)
