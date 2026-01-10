import argparse
import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
import yaml
from difflib import get_close_matches
from typing import Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(os.getcwd())
try:
    from src.recommender import LightweightRecommender
    from src.encoder import BookEncoderModel
    from src.utils import clean_html, safe_str, robust_parse_genres
except ImportError:
    sys.path.append("/app")
    from src.recommender import LightweightRecommender
    from src.encoder import BookEncoderModel
    from src.utils import clean_html, safe_str, robust_parse_genres

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("PredictService")


class ColdStartProcessor:
    def __init__(self, config: dict, meta_df: pd.DataFrame, device: str = "cpu"):
        self.device = device
        self.config = config

        base_model_name = config["embeddings"]["model_name"]
        logger.info(f"Loading SentenceTransformer: {base_model_name}...")
        self.base_model = SentenceTransformer(base_model_name, device=device)

        logger.info("Fitting MultiLabelBinarizer on existing metadata...")
        self.mlb = MultiLabelBinarizer()
        genres_list = (
            meta_df["genres_list"].fillna("").astype(str).apply(robust_parse_genres)
        )
        self.mlb.fit(genres_list)
        logger.info(f"MLB learned {len(self.mlb.classes_)} classes.")

        output_dim = config["training"]["output_dim"]
        outputs_dir = config["paths"]["outputs_dir"]

        model_dir = os.path.join(outputs_dir, f"book_encoder_contrastive_{output_dim}")

        logger.info(f"Loading trained neural network from {model_dir}...")
        try:
            self.encoder_model = BookEncoderModel.from_pretrained(model_dir)
            self.encoder_model.to(device)
            self.encoder_model.eval()
        except Exception as e:
            logger.error(f"Critical: Failed to load trained model: {e}")
            raise e

    def process_row(self, row: pd.Series) -> np.ndarray:
        desc_clean = clean_html(str(row.get("description", "")))
        author_clean = safe_str(str(row.get("author", "")))
        series_clean = safe_str(str(row.get("series", "")))

        raw_genres = str(row.get("genres", ""))
        genres_parsed = robust_parse_genres(raw_genres)
        genre_vec = self.mlb.transform([genres_parsed]).astype(np.float32)

        with torch.no_grad():
            series_emb = self.base_model.encode([series_clean])
            author_emb = self.base_model.encode([author_clean])
            desc_emb = self.base_model.encode([desc_clean])

        combined_input = np.hstack([series_emb, author_emb, desc_emb, genre_vec])
        input_tensor = torch.tensor(combined_input, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            final_emb = self.encoder_model(input_tensor).cpu().numpy()

        return final_emb[0]


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
            return str(candidate["bookId"]), "Fuzzy match by author"

    return None, "Not found"


def main(args):
    logger.info(f"Loading config from {args.config}...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = config["paths"]["processed_data_dir"]
    outputs_dir = config["paths"]["outputs_dir"]
    output_dim = config["training"]["output_dim"]

    meta_path = os.path.join(processed_dir, "books_meta_multimodal.csv")

    embeddings_filename = f"book_embeddings_contrastive_{output_dim}.npy"
    embeddings_path = os.path.join(
        outputs_dir, f"book_encoder_contrastive_{output_dim}", embeddings_filename
    )

    logger.info(f"Loading metadata from {meta_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Meta file missing: {meta_path}. Run 'dvc pull' or pipeline."
        )

    books_meta = pd.read_csv(meta_path, keep_default_na=False, na_values=[])
    books_meta["_title_norm"] = books_meta["title"].astype(str).str.lower().str.strip()
    books_meta["_author_norm"] = (
        books_meta["author_clean"].astype(str).str.lower().str.strip()
    )

    logger.info(f"Loading embeddings from {embeddings_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings missing: {embeddings_path}")
    embeddings = np.load(embeddings_path)

    cold_processor = ColdStartProcessor(config, books_meta, device="cpu")

    logger.info(f"Reading input from {args.input_path}")
    try:
        user_df = pd.read_csv(args.input_path)
    except Exception as e:
        logger.error(f"Cannot read input CSV: {e}")
        sys.exit(1)

    required = ["title", "author", "rating"]
    for c in required:
        if c not in user_df.columns:
            raise ValueError(f"Input CSV missing column: {c}")

    user_history = []
    new_embeddings_list = []
    new_meta_rows = []

    for i, row in user_df.iterrows():
        title = row["title"]
        author = row["author"]
        rating = float(row["rating"])

        bid, msg = find_book_id_by_title_author(title, author, books_meta)

        if bid:
            logger.info(f"Found '{title}': {msg} (ID: {bid})")
            user_history.append((bid, rating))
        else:
            logger.info(f"Book '{title}' not found. Generating embedding...")

            if "description" not in row or pd.isna(row["description"]):
                logger.warning(
                    f"Skipping '{title}': No description provided for cold start."
                )
                continue
            try:
                vec = cold_processor.process_row(row)
                temp_id = f"NewBook_{i}"
                new_embeddings_list.append(vec)

                new_meta_rows.append(
                    {
                        "bookId": temp_id,
                        "title": title,
                        "author_clean": author,
                        "avg_rating": row.get("avg_rating", 4.0),
                    }
                )

                user_history.append((temp_id, rating))
            except Exception as e:
                logger.error(f"Error processing '{title}': {e}")

    if not user_history:
        logger.error("No valid books. Exiting.")
        pd.DataFrame({"error": ["No books"]}).to_csv(args.output_path)
        return

    final_embeddings = embeddings
    final_meta = books_meta

    if new_embeddings_list:
        logger.info(
            f"Adding {len(new_embeddings_list)} new books to temporary database."
        )

        new_emb_arr = np.vstack(new_embeddings_list)
        final_embeddings = np.vstack([embeddings, new_emb_arr])

        new_meta_df = pd.DataFrame(new_meta_rows)
        final_meta = pd.concat([books_meta, new_meta_df], ignore_index=True).fillna("")

    if args.popularity_weight is None:
        rec_weight = config.get("recommend", {}).get("popularity_weight", 0.9)
    else:
        rec_weight = args.popularity_weight

    logger.info(f"Generating top-{args.k} recommendations...")

    recommender = LightweightRecommender(
        embeddings=final_embeddings, books_meta=final_meta, popularity_weight=rec_weight
    )

    recs = recommender.recommend(
        user_history=user_history,
        k=args.k,
        exclude_book_ids=[bid for bid, _ in user_history],
    )

    recs_df = pd.DataFrame(recs)
    logger.info(f"Saving to {args.output_path}")

    if not recs_df.empty:
        cols = [
            "title",
            "author",
            "predicted_rating",
            "bookId",
            "similarity_to_user_profile",
        ]
        valid_cols = [c for c in cols if c in recs_df.columns]
        recs_df[valid_cols].to_csv(args.output_path, index=False)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--config", type=str, default="configs/default.yaml")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--popularity_weight", type=float, default=None)

    args = parser.parse_args()
    main(args)
