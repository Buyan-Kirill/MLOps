#!/usr/bin/env python3
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sentence_transformers.util import cos_sim


class LightweightRecommender:
    def __init__(
        self,
        embeddings: np.ndarray,
        books_meta: pd.DataFrame,
        popularity_weight: float = 0.3,
    ) -> None:
        if len(embeddings) != len(books_meta):
            raise ValueError(
                f"Несоответствие длины: "
                f"embeddings={len(embeddings)}, "
                f"books_meta={len(books_meta)}"
            )

        self.embeddings = embeddings.astype(np.float32)
        self.books_meta = books_meta.copy()
        self.popularity_weight = float(popularity_weight)
        assert (
            0.0 <= self.popularity_weight <= 1.0
        ), "popularity_weight должен быть в [0,1]"

        self.bookid_to_idx = {
            str(bid): i for i, bid in enumerate(self.books_meta["bookId"])
        }
        self.idx_to_bookid = {
            i: str(bid) for i, bid in enumerate(self.books_meta["bookId"])
        }

        self.avg_rating_map = (
            self.books_meta.set_index("bookId")["avg_rating"].fillna(3.0).to_dict()
        )

    def predict_rating(
        self,
        user_history: List[Tuple[str, int]],
        candidate_book_id: str,
    ) -> float:
        candidate_book_id = str(candidate_book_id)

        if candidate_book_id not in self.bookid_to_idx:
            return 3.0

        cand_idx = self.bookid_to_idx[candidate_book_id]
        cand_emb = self.embeddings[cand_idx]

        liked = [(bid, r) for bid, r in user_history if r >= 4]
        disliked = [(bid, r) for bid, r in user_history if r <= 2]

        score = 0.0

        for bid, rating in liked:
            bid = str(bid)
            if bid not in self.bookid_to_idx:
                continue
            emb = self.embeddings[self.bookid_to_idx[bid]]
            sim = cos_sim(
                torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
                torch.tensor(cand_emb, dtype=torch.float32).unsqueeze(0),
            ).item()
            score += sim * (rating / 5.0)

        for bid, rating in disliked:
            bid = str(bid)
            if bid not in self.bookid_to_idx:
                continue
            emb = self.embeddings[self.bookid_to_idx[bid]]
            sim = cos_sim(
                torch.tensor(emb, dtype=torch.float32).unsqueeze(0),
                torch.tensor(cand_emb, dtype=torch.float32).unsqueeze(0),
            ).item()
            score -= sim * ((6 - rating) / 5.0)

        personalized = np.clip(3.0 + score, 1.0, 5.0)

        avg_rating = self.avg_rating_map.get(candidate_book_id, 3.0)

        final_rating = (
            1.0 - self.popularity_weight
        ) * personalized + self.popularity_weight * avg_rating
        return float(np.clip(final_rating, 1.0, 5.0))

    def recommend(
        self,
        user_history: List[Tuple[str, int]],
        k: int = 5,
        exclude_book_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        if not user_history:
            candidates = self.books_meta.copy()
            if exclude_book_ids:
                exclude_set = {str(bid) for bid in exclude_book_ids}
                candidates = candidates[~candidates["bookId"].isin(exclude_set)]
            top_k = candidates.nlargest(k, "avg_rating")
            return [
                {
                    "bookId": row["bookId"],
                    "title": row.get("title", ""),
                    "author": row.get("author_clean", ""),
                    "predicted_rating": row["avg_rating"],
                    "similarity_to_user_profile": 0.0,
                }
                for _, row in top_k.iterrows()
            ]

        profile_emb = np.zeros(self.embeddings.shape[1], dtype=np.float32)
        total_weight = 0.0

        for bid, rating in user_history:
            bid = str(bid)
            if bid not in self.bookid_to_idx:
                continue
            idx = self.bookid_to_idx[bid]
            emb = self.embeddings[idx]
            weight = rating - 3.0
            profile_emb += emb * weight
            total_weight += abs(weight)

        if total_weight == 0:
            profile_emb = np.mean(self.embeddings, axis=0)
        else:
            profile_emb /= total_weight

        profile_tensor = torch.tensor(profile_emb, dtype=torch.float32).unsqueeze(0)
        all_embs_tensor = torch.tensor(self.embeddings, dtype=torch.float32)
        similarities = cos_sim(profile_tensor, all_embs_tensor).squeeze(0).numpy()

        exclude_set = {str(bid) for bid, _ in user_history}
        if exclude_book_ids:
            exclude_set.update(str(bid) for bid in exclude_book_ids)

        valid_indices = [
            i
            for i in range(len(self.books_meta))
            if self.idx_to_bookid[i] not in exclude_set
        ]

        if not valid_indices:
            return []

        valid_sims = similarities[valid_indices]
        top_k_idx = np.argsort(-valid_sims)[:k]
        top_indices = [valid_indices[i] for i in top_k_idx]

        result = []
        for idx in top_indices:
            book_id = self.idx_to_bookid[idx]
            row = self.books_meta.iloc[idx]
            pred_rating = self.predict_rating(user_history, book_id)

            result.append(
                {
                    "bookId": book_id,
                    "title": row.get("title", ""),
                    "author": row.get("author_clean", ""),
                    "predicted_rating": round(pred_rating, 2),
                    "similarity_to_user_profile": round(float(similarities[idx]), 3),
                }
            )

        return result
