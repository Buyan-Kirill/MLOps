import os
import torch
import json
import logging
import pandas as pd
import numpy as np
import yaml
from ts.torch_handler.base_handler import BaseHandler
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from encoder import BookEncoderModel, BookEncoderConfig
    from recommender import LightweightRecommender
    from utils import robust_parse_genres, find_book_id_by_title_author
    from cold_start import ColdStartProcessor
except ImportError:
    from src.encoder import BookEncoderModel, BookEncoderConfig
    from src.recommender import LightweightRecommender
    from src.utils import robust_parse_genres, find_book_id_by_title_author
    from src.cold_start import ColdStartProcessor


logger = logging.getLogger(__name__)


class RecommendationHandler(BaseHandler):
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.model_dir = properties.get("model_dir")
        self.device = torch.device("cpu")

        logger.info(f"Initializing handler. Model dir: {self.model_dir}")

        config_file = os.path.join(self.model_dir, "default.yaml")
        if os.path.exists(config_file):
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning("Config file not found in archive, using defaults")
            self.config = {"embeddings": {"model_name": "all-MiniLM-L6-v2"}}

        model_config_path = os.path.join(self.model_dir, "config.json")
        model_conf = BookEncoderConfig.from_json_file(model_config_path)
        self.model = BookEncoderModel(model_conf)

        serialized_file = self.manifest["model"]["serializedFile"]
        state_dict = torch.load(
            os.path.join(self.model_dir, serialized_file), map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        base_name = self.config["embeddings"]["model_name"]
        self.base_model = SentenceTransformer(base_name, device="cpu")

        meta_path = os.path.join(self.model_dir, "books_meta_multimodal.csv")
        npy_files = [f for f in os.listdir(self.model_dir) if f.endswith(".npy")]
        emb_path = os.path.join(self.model_dir, npy_files[0])

        self.books_meta = pd.read_csv(meta_path, keep_default_na=False, na_values=[])
        self.books_meta["_title_norm"] = (
            self.books_meta["title"].astype(str).str.lower().str.strip()
        )
        self.books_meta["_author_norm"] = (
            self.books_meta["author_clean"].astype(str).str.lower().str.strip()
        )

        self.embeddings = np.load(emb_path)

        self.mlb = MultiLabelBinarizer()
        genres_list = (
            self.books_meta["genres_list"]
            .fillna("")
            .astype(str)
            .apply(robust_parse_genres)
        )
        self.mlb.fit(genres_list)

        self.cold_processor = ColdStartProcessor(
            self.model, self.base_model, self.mlb, self.device
        )

        self.pop_weight = self.config.get("recommend", {}).get("popularity_weight", 0.3)

        self.initialized = True
        logger.info("Service fully initialized.")

    def preprocess(self, data):
        processed_reqs = []
        for row in data:
            if isinstance(row, dict) and "body" in row:
                row = row.get("body")
            if isinstance(row, (bytes, bytearray)):
                row = json.loads(row.decode("utf-8"))

            # {"history": [...], "k": 5}
            processed_reqs.append(row)
        return processed_reqs

    def inference(self, inputs):
        batch_results = []

        for req in inputs:
            user_input_list = req.get(
                "history", []
            )  # [{title, author, rating, ...} ... ]
            k = req.get("k", 5)

            user_history_ids = []
            new_embeddings_list = []
            new_meta_rows = []

            for i, item in enumerate(user_input_list):
                title = item.get("title", "")
                author = item.get("author", "")
                rating = float(item.get("rating", 3.0))

                bid, msg = find_book_id_by_title_author(title, author, self.books_meta)
                if bid:
                    user_history_ids.append((bid, rating))
                else:
                    if "description" in item:
                        logger.info(f"Generating embedding for '{title}'")
                        try:
                            vec = self.cold_processor.process_row(item)
                            temp_id = f"NewBook_Req_{i}"

                            new_embeddings_list.append(vec)
                            new_meta_rows.append(
                                {
                                    "bookId": temp_id,
                                    "title": title,
                                    "author_clean": author,
                                    "avg_rating": item.get("avg_rating", 4.0),
                                }
                            )

                            user_history_ids.append((temp_id, rating))
                        except Exception as e:
                            logger.error(f"Error generating embedding: {e}")
                    else:
                        logger.warning(
                            f"Skipping '{title}': Not found and no description provided."
                        )

            if not user_history_ids:
                batch_results.append([])
                continue

            current_embeddings = self.embeddings
            current_meta = self.books_meta

            if new_embeddings_list:
                new_emb_arr = np.vstack(new_embeddings_list)
                current_embeddings = np.vstack([self.embeddings, new_emb_arr])

                new_meta_df = pd.DataFrame(new_meta_rows)
                current_meta = pd.concat(
                    [self.books_meta, new_meta_df], ignore_index=True
                ).fillna("")

            recommender = LightweightRecommender(
                embeddings=current_embeddings,
                books_meta=current_meta,
                popularity_weight=self.pop_weight,
            )

            recs = recommender.recommend(
                user_history=user_history_ids,
                k=k,
                exclude_book_ids=[bid for bid, _ in user_history_ids],
            )
            batch_results.append(recs)

        return batch_results

    def postprocess(self, inference_output):
        return inference_output
