import argparse
import os
import sys
import pandas as pd
import numpy as np
import logging
from difflib import get_close_matches
from typing import Tuple, Optional

sys.path.append(os.getcwd())

try:
    from src.recommender import LightweightRecommender
except ImportError:
    sys.path.append("/app")
    from src.recommender import LightweightRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RecommenderService")


def find_book_id_by_title_author(
    title: str, author: str, books_meta: pd.DataFrame
) -> Tuple[Optional[str], str]:
    title_norm = str(title).lower().strip()
    author_norm = str(author).lower().strip()

    exact = books_meta[
        (books_meta["_title_norm"] == title_norm)
        & (books_meta["_author_norm"] == author_norm)
    ]
    if not exact.empty:
        return str(exact.iloc[0]["bookId"]), "Exact match"

    by_author = books_meta[
        books_meta["_author_norm"].str.contains(author_norm, na=False, regex=False)
    ]
    if len(by_author) > 0:
        titles = by_author["_title_norm"].tolist()
        matches = get_close_matches(title_norm, titles, n=1, cutoff=0.6)
        if matches:
            candidate = by_author[by_author["_title_norm"] == matches[0]].iloc[0]
            return str(candidate["bookId"]), "Fuzzy title match by author"

    all_titles = books_meta["_title_norm"].tolist()
    matches = get_close_matches(title_norm, all_titles, n=1, cutoff=0.7)
    if matches:
        candidate = books_meta[books_meta["_title_norm"] == matches[0]].iloc[0]
        return str(candidate["bookId"]), "Fuzzy title match global"

    return None, "Not found"


def main(
    input_path, output_path, embeddings_path, meta_path, k=5, popularity_weight=0.3
):
    logger.info(f"Loading metadata from {meta_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    books_meta = pd.read_csv(meta_path, keep_default_na=False, na_values=[])
    books_meta["_title_norm"] = books_meta["title"].astype(str).str.lower().str.strip()
    books_meta["_author_norm"] = (
        books_meta["author_clean"].astype(str).str.lower().str.strip()
    )

    logger.info(f"Loading embeddings from {embeddings_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)

    logger.info(f"Reading user history from {input_path}")
    try:
        user_df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        sys.exit(1)

    required_cols = ["title", "author", "rating"]
    for col in required_cols:
        if col not in user_df.columns:
            raise ValueError(f"Input CSV must contain column '{col}'")

    logger.info("Resolving book IDs...")
    user_history = []

    for _, row in user_df.iterrows():
        bid, msg = find_book_id_by_title_author(row["title"], row["author"], books_meta)
        if bid:
            logger.info(f"Found '{row['title']}': {msg} (ID: {bid})")
            user_history.append((bid, float(row["rating"])))
        else:
            logger.warning(
                f"Skipping '{row['title']}' ({row['author']}): Not found in database"
            )

    if not user_history:
        logger.error("No valid books found in input history. Cannot recommend.")
        pd.DataFrame({"error": ["No books found"]}).to_csv(output_path, index=False)
        return

    logger.info(f"Generating top-{k} recommendations...")
    recommender = LightweightRecommender(
        embeddings=embeddings,
        books_meta=books_meta,
        popularity_weight=popularity_weight,
    )

    recs = recommender.recommend(
        user_history=user_history,
        k=k,
        exclude_book_ids=[bid for bid, _ in user_history],
    )

    logger.info(f"Saving recommendations to {output_path}")
    recs_df = pd.DataFrame(recs)

    cols_to_save = [
        "title",
        "author",
        "predicted_rating",
        "bookId",
        "similarity_to_user_profile",
    ]
    actual_cols = [c for c in cols_to_save if c in recs_df.columns]

    recs_df[actual_cols].to_csv(output_path, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input CSV (history)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output CSV (recommendations)",
    )

    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="outputs/book_encoder_contrastive_256/book_embeddings_contrastive_256.npy",
    )
    parser.add_argument(
        "--meta_path", type=str, default="processed_data/books_meta_multimodal.csv"
    )

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--popularity_weight", type=float, default=0.3)

    args = parser.parse_args()
    main(
        args.input_path,
        args.output_path,
        args.embeddings_path,
        args.meta_path,
        args.k,
        args.popularity_weight,
    )
