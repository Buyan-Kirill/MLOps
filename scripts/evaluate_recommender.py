#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.metrics import spearman_rank_correlation
from src.utils import set_seed
from src.recommender import LightweightRecommender

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def evaluate_user(
    group: pd.DataFrame,
    recommender: LightweightRecommender,
    books_meta: pd.DataFrame,
    history_part: float,
    max_history: int
) -> tuple[float | None, float | None]:
    if len(group) < 3:
        return None, None
    group = group.sample(frac=1, random_state=42)
    split = min(int(history_part * len(group)), max_history)
    history = group.iloc[:split]
    test = group.iloc[split:]
    user_history = list(zip(history["bookId"], history["rating"]))

    y_true, y_pred_model, y_pred_pop = [], [], []
    for _, row in test.iterrows():
        bid = row["bookId"]
        true_rating = row["rating"]
        pred_model = recommender.predict_rating(user_history, bid)
        pred_pop = books_meta.set_index("bookId").loc[bid, "avg_rating"]
        y_true.append(true_rating)
        y_pred_model.append(pred_model)
        y_pred_pop.append(pred_pop)

    corr_model = spearman_rank_correlation(y_true, y_pred_model)
    corr_pop = spearman_rank_correlation(y_true, y_pred_pop)
    return corr_model, corr_pop


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    PROCESSED_DIR = config["paths"]["processed_data_dir"]
    OUTPUT_DIR = config["paths"]["outputs_dir"]
    OUTPUT_DIM = config['training']['output_dim']
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)

    logger.info("Загрузка данных...")
    embeddings = np.load(
        os.path.join(
            OUTPUT_DIR,
            f"book_encoder_contrastive_{OUTPUT_DIM}",
            f"book_embeddings_contrastive_{OUTPUT_DIM}.npy",
        )
    )
    books_meta = pd.read_csv(
        os.path.join(PROCESSED_DIR, "books_meta_multimodal.csv")
    )
    ratings = pd.read_parquet(
        os.path.join(
            config["paths"]["processed_data_dir"], "ratings_clean.parquet"
        )
    )

    valid_books = set(books_meta["bookId"])
    ratings = ratings[ratings["bookId"].isin(valid_books)].copy()

    user_groups = list(ratings.groupby("user_id"))
    if len(user_groups) > config["evaluation"]["max_users"]:
        user_groups = user_groups[: config["evaluation"]["max_users"]]

    logger.info(f"Оценка на {len(user_groups)} пользователях...")

    max_history = config["evaluation"]["history_max_len"]
    history_part = config["evaluation"]["history_part"]
    weights = config["evaluation"]["popularity_weights"]
    results = {w: [] for w in weights}
    pop_results = []

    for user_id, group in tqdm(user_groups, desc="Пользователи"):
        corr_model_pop, corr_pop = evaluate_user(
            group,
            LightweightRecommender(
                embeddings, books_meta, popularity_weight=1.0
            ),
            books_meta,
            history_part,
            max_history,
        )
        if corr_pop is not None and not np.isnan(corr_pop):
            pop_results.append(corr_pop)

        for w in weights:
            recommender = LightweightRecommender(
                embeddings, books_meta, popularity_weight=w
            )
            corr_model, _ = evaluate_user(
                group, recommender, books_meta, history_part, max_history
            )
            if corr_model is not None and not np.isnan(corr_model):
                results[w].append(corr_model)

    avg_model = {
        w: np.mean(results[w]) if results[w] else np.nan for w in weights
    }
    avg_pop = np.mean(pop_results) if pop_results else np.nan

    logger.info("\n" + "=" * 50)
    logger.info("Оптимизация веса популярности")
    logger.info("=" * 50)
    for w in weights:
        logger.info(f"Вес {w}: {avg_model[w]:.4f}")
    logger.info(f"Только популярность: {avg_pop:.4f}")
    logger.info("=" * 50)

    best_weight = max(
        avg_model,
        key=lambda w: avg_model[w] if not np.isnan(avg_model[w]) else -1,
    )
    logger.info(
        f"\nОптимальный вес: {best_weight} "
        f"(корреляция: {avg_model[best_weight]:.4f})"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        weights,
        [avg_model[w] for w in weights],
        "o-",
        label="Модель",
    )
    plt.axhline(
        avg_pop,
        color="red",
        linestyle="--",
        label=f"Популярность ({avg_pop:.3f})",
    )
    plt.xlabel("Вес популярности")
    plt.ylabel("Spearman correlation")
    plt.title("Подбор веса популярности")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PROCESSED_DIR, "popularity_tuning.png"))
    logger.info("График сохранён: popularity_tuning.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml"
    )
    args = parser.parse_args()
    main(args.config)
