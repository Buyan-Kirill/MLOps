#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import (
    clean_html,
    safe_str,
    set_seed,
    parse_genres,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/embed.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    INPUT_DIR = config["paths"]["processed_data_dir"]
    OUTPUT_DIR = config["paths"]["processed_data_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)

    logger.info("Загрузка книг...")
    books = pd.read_parquet(
        os.path.join(INPUT_DIR, "books_clean.parquet")
    )

    books["author_clean"] = books["author"].apply(safe_str)
    books["series_clean"] = books.get("series", "").apply(safe_str)
    books["description_clean"] = books["description"].apply(clean_html)
    books["genres_list"] = books["genres"].apply(parse_genres)

    books = books[
        books["title"].notna() & (books["title"] != "")
    ].copy().reset_index(drop=True)

    logger.info("Кодирование...")
    model = SentenceTransformer(config["embeddings"]["model_name"])
    logger.info("Кодирование серий")
    series_emb = model.encode(
        books["series_clean"].tolist(), show_progress_bar=True
    )
    logger.info("Кодирование авторов")
    author_emb = model.encode(
        books["author_clean"].tolist(), show_progress_bar=True
    )
    logger.info("Кодирование описаний")
    desc_emb = model.encode(
        books["description_clean"].tolist(), show_progress_bar=True
    )

    mlb = MultiLabelBinarizer()
    genre_multihot = mlb.fit_transform(books["genres_list"])

    logger.info(f"   series_embeddings: {series_emb.shape}")
    logger.info(f"   author_embeddings: {author_emb.shape}")
    logger.info(f"   description_embeddings: {desc_emb.shape}")
    logger.info(f"   genre_multihot_embeddings: {genre_multihot.shape}")

    combined = np.hstack([series_emb, author_emb, desc_emb, genre_multihot])
    np.save(
        os.path.join(OUTPUT_DIR, "book_embeddings_multimodal.npy"),
        combined,
    )
    np.save(
        os.path.join(OUTPUT_DIR, "book_descriptions_original.npy"),
        desc_emb,
    )

    meta = books[
        [
            "bookId",
            "title",
            "author_clean",
            "series_clean",
            "genres_list",
            "description_clean",
            "avg_rating",
        ]
    ].copy()
    meta["genres_list"] = meta["genres_list"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )
    meta["author_clean"] = meta["author_clean"].fillna("").astype(str)
    meta["series_clean"] = meta["series_clean"].fillna("").astype(str)
    meta.to_csv(
        os.path.join(OUTPUT_DIR, "books_meta_multimodal.csv"),
        index=False,
        na_rep="",
    )

    logger.info(f"Сохранено {len(books)} эмбеддингов.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml"
    )
    args = parser.parse_args()
    main(args.config)
