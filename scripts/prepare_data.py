#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import pandas as pd
import zipfile
import subprocess
import glob
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import clean_html, normalize_title, safe_str, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/prepare.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dataset_slug: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
    if csv_files:
        logger.info(f"Датасет {dataset_slug} уже загружен: {csv_files[0]}")
        return

    logger.info(f"Загрузка '{dataset_slug}' в {target_dir}...")
    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_slug,
                "-p",
                target_dir,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        zip_files = [f for f in os.listdir(target_dir) if f.endswith(".zip")]
        if not zip_files:
            raise FileNotFoundError("Нет .zip после загрузки")
        zip_path = os.path.join(target_dir, zip_files[0])
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        os.remove(zip_path)
        extracted_csv = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
        logger.info(f"Распаковано: {extracted_csv}")
    except Exception as e:
        logger.error(f"Kaggle: {e}")
        logger.error("Проверьте: 1) kaggle.json, 2) соглашение на Kaggle")
        raise SystemExit(1)


def validate_books_df(df: pd.DataFrame, name: str = "books") -> pd.DataFrame:
    """Проверяет и чистит датафрейм книг. bookId — строка."""
    logger.info(f"Валидация {name} (до: {len(df)} строк)...")

    required = ["bookId", "title"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: отсутствуют колонки {missing}")

    df = df.copy()
    df["bookId"] = df["bookId"].astype(str).apply(safe_str)
    df["title"] = df["title"].astype(str).apply(safe_str)
    df["author"] = (
        df.get("author", "Unknown").fillna("Unknown").astype(str).apply(safe_str)
    )
    df["description"] = (
        df.get("description", "").fillna("").astype(str).apply(clean_html)
    )
    df["genres"] = df.get("genres", "").fillna("").astype(str)

    # Удаление строк с пустыми bookId/title
    before = len(df)
    df = df[df["bookId"].str.strip() != ""]
    df = df[df["title"].str.strip() != ""]
    removed = before - len(df)
    if removed > 0:
        logger.info(f"   Удалено {removed} строк с пустыми bookId/title")

    dup_bookid = df[df.duplicated(subset=["bookId"])]
    if len(dup_bookid) > 0:
        logger.warning(f"   Удалено {len(dup_bookid)} дубликатов по bookId")
        df = df.drop_duplicates(subset=["bookId"]).copy()

    for col in ["rating", "pages", "likedPercent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].fillna(median_val)
                logger.warning(
                    f"   Заполнено {missing} пропусков в '{col}' "
                    f"медианой ({median_val:.2f})"
                )
        else:
            df[col] = df["rating"].median() if col == "rating" else 0.0

    logger.info(f"{name}: {len(df)} книг, {df['bookId'].nunique()} уникальных ID")
    logger.info(
        f"   Средний рейтинг книг: {df['rating'].mean():.2f} ± {df['rating'].std():.2f}"
    )

    return df.reset_index(drop=True)


def validate_ratings_df(df: pd.DataFrame, name: str = "ratings") -> pd.DataFrame:
    """Проверяет и чистит рейтинги."""
    logger.info(f"Валидация {name}. До очистки: {len(df)} строк")

    required = ["ID", "Name", "Rating"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: отсутствуют {missing}")

    df = df.copy()
    df.rename(
        columns={
            "ID": "user_id",
            "Name": "title",
            "Rating": "rating_text",
        },
        inplace=True,
    )

    # Удаляем пустые
    df = df[
        df["rating_text"].notna()
        & (df["rating_text"] != "This user doesn't have any rating")
    ]

    rating_map = {
        "did not like it": 1,
        "it was ok": 2,
        "liked it": 3,
        "really liked it": 4,
        "it was amazing": 5,
    }
    df["rating"] = df["rating_text"].map(rating_map)
    invalid = df["rating"].isna().sum()
    if invalid > 0:
        logger.warning(f"   Удалено {invalid} строк с некорректным рейтингом")
        df = df.dropna(subset=["rating"])

    df["rating"] = df["rating"].astype(int)
    df["user_id"] = df["user_id"].astype(str).apply(safe_str)
    df["title"] = df["title"].astype(str).apply(safe_str)
    df["norm_title"] = df["title"].apply(normalize_title)

    out_of_range = ((df["rating"] < 1) | (df["rating"] > 5)).sum()
    if out_of_range > 0:
        logger.warning(f"   Удалено {out_of_range} рейтингов вне [1,5]")
        df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

    logger.info(f"   После очистки: {len(df)} строк")
    logger.info(
        f"   Средний рейтинг пользователей: {df['rating'].mean():.2f} "
        f"± {df['rating'].std():.2f}"
    )
    logger.info(f"   Уникальных пользователей: {df['user_id'].nunique()}")
    return df.reset_index(drop=True)


def main(config_path: str, limit_ratings: Optional[int] = None) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    logger.info("Проверка датасетов...")
    BEST_DATASET = "thedevastator/comprehensive-overview-of-52478-goodreads-best-b"
    RATING_DATASET = "bahramjannesarr/goodreads-book-datasets-10m"
    BEST_DIR = os.path.join(config["paths"]["dataset_dir"], "best_books_dataset")
    RATING_DIR = os.path.join(config["paths"]["dataset_dir"], "all_books_dataset")

    os.makedirs(config["paths"]["dataset_dir"], exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)
    os.makedirs(RATING_DIR, exist_ok=True)

    if not glob.glob(os.path.join(BEST_DIR, "*.csv")):
        download_kaggle_dataset(BEST_DATASET, BEST_DIR)
    if not any(f.startswith("user_rating") for f in os.listdir(RATING_DIR)):
        download_kaggle_dataset(RATING_DATASET, RATING_DIR)

    best_csv = glob.glob(os.path.join(BEST_DIR, "*.csv"))[0]
    logger.info(f"Чтение {best_csv}...")
    books_raw = pd.read_csv(best_csv, on_bad_lines="skip", low_memory=False)
    books_clean = validate_books_df(books_raw, "best_books")

    books_clean["norm_title"] = books_clean["title"].apply(normalize_title)
    books_clean["norm_author"] = books_clean["author"].apply(normalize_title)
    books_clean = books_clean.drop_duplicates(
        subset=["norm_title", "norm_author"]
    ).copy()
    books_clean = books_clean.rename(columns={"rating": "avg_rating"})
    logger.info(f"Уникальных книг (по title+author): {len(books_clean)}")

    rating_files = [
        f
        for f in os.listdir(RATING_DIR)
        if f.startswith("user_rating") and f.endswith(".csv")
    ]
    if not rating_files:
        raise FileNotFoundError(f"Нет файлов рейтингов в {RATING_DIR}")

    ratings_list = []
    for f in rating_files:
        df = pd.read_csv(
            os.path.join(RATING_DIR, f),
            on_bad_lines="skip",
            low_memory=False,
        )
        ratings_list.append(df)
        if limit_ratings and sum(len(d) for d in ratings_list) >= limit_ratings:
            break

    ratings_raw = pd.concat(ratings_list, ignore_index=True)
    if limit_ratings:
        ratings_raw = ratings_raw.head(limit_ratings)

    ratings_clean = validate_ratings_df(ratings_raw, "user_ratings")

    title_to_meta = books_clean.set_index("norm_title")
    ratings_enriched = ratings_clean.join(
        title_to_meta, on="norm_title", how="inner", rsuffix="_book"
    )
    if "rating_book" in ratings_enriched.columns:
        ratings_enriched = ratings_enriched.drop(columns=["rating_book"])

    logger.info(f"Сопоставлено {len(ratings_enriched)} рейтингов с книгами.")

    if ratings_enriched.empty:
        sample_titles = (
            ratings_clean["norm_title"].sample(min(10, len(ratings_clean))).tolist()
        )
        book_titles = set(books_clean["norm_title"])
        unmatched = [t for t in sample_titles if t not in book_titles]
        logger.warning(f"Несопоставленные названия (примеры): {unmatched}")
        raise ValueError("Ни один рейтинг не сопоставился! Проверьте нормализацию.")

    ratings_enriched = ratings_enriched[
        ratings_enriched["bookId"].notna() & (ratings_enriched["bookId"] != "")
    ]
    book_ids_in_ratings = ratings_enriched["bookId"].unique()
    books_final = books_clean[books_clean["bookId"].isin(book_ids_in_ratings)].copy()

    assert len(books_final) > 0, "Нет книг после фильтрации"
    assert len(ratings_enriched) > 0, "Нет рейтингов после фильтрации"
    assert "bookId" in books_final.columns, "Нет bookId"
    assert "avg_rating" in books_final.columns, "Нет avg_rating"
    assert ratings_enriched["bookId"].notna().all(), "Есть NaN в bookId рейтингов"

    OUTPUT_DIR = config["paths"]["processed_data_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)

    books_final.to_parquet(os.path.join(OUTPUT_DIR, "books_clean.parquet"), index=False)
    ratings_enriched.to_parquet(
        os.path.join(OUTPUT_DIR, "ratings_clean.parquet"), index=False
    )

    logger.info("Обработка датасета завершена. После сопоставления осталось:")
    logger.info(f"   Книг: {len(books_final)}")
    logger.info(f"   Рейтингов: {len(ratings_enriched)}")
    logger.info(f"   Пользователей: {ratings_enriched['user_id'].nunique()}")
    logger.info(f"   Файлы сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--limit-ratings", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.limit_ratings)
