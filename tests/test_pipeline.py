#!/usr/bin/env python3
"""
Тесты для ML-пайплайна рекомендательной системы.
Запуск: pytest tests/ -v
"""

import os
import numpy as np
import pandas as pd
import torch
import yaml
from src.utils import parse_genre_str
from scripts.train import ContrastiveBookDataset, weighted_contrastive_loss
from src.encoder import BookEncoderModel, BookEncoderConfig
from src.recommender import LightweightRecommender


def load_config():
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "default.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_processed_data_dir():
    config = load_config()
    return config["paths"]["processed_data_dir"]


def test_books_exist_and_required_columns():
    INPUT_DIR = get_processed_data_dir()
    path = os.path.join(INPUT_DIR, "books_clean.parquet")
    assert os.path.exists(path), f"Файл не найден: {path}"

    books = pd.read_parquet(path)

    required_cols = {
        "bookId",
        "title",
        "author",
        "description",
        "genres",
        "avg_rating",
    }
    missing = required_cols - set(books.columns)
    assert not missing, f"Отсутствуют колонки: {missing}"

    assert books["bookId"].dtype == "object", "bookId должен быть строкой"
    assert books["title"].dtype == "object", "title должен быть строкой"
    assert books["author"].dtype == "object", "title должен быть строкой"
    assert books["description"].dtype == "object", "title должен быть строкой"
    assert pd.api.types.is_numeric_dtype(
        books["avg_rating"]
    ), "avg_rating должен быть числом"


def test_ratings_exist_and_required_columns():
    INPUT_DIR = get_processed_data_dir()
    path = os.path.join(INPUT_DIR, "ratings_clean.parquet")
    assert os.path.exists(path), f"Файл не найден: {path}"

    ratings = pd.read_parquet(path)

    required_cols = {"user_id", "title", "rating", "bookId"}
    missing = required_cols - set(ratings.columns)
    assert not missing, f"Отсутствуют колонки: {missing}"

    assert ratings["bookId"].dtype == "object", "bookId должен быть строкой"
    assert ratings["title"].dtype == "object", "title должен быть строкой"
    assert pd.api.types.is_integer_dtype(ratings["rating"]), "rating должен быть int"
    assert ratings["rating"].between(1, 5).all(), "rating вне [1,5]"


def test_books_meta_multimodal_valid():
    INPUT_DIR = get_processed_data_dir()
    path = os.path.join(INPUT_DIR, "books_meta_multimodal.csv")
    assert os.path.exists(path), f"Файл не найден: {path}"

    meta = pd.read_csv(path, keep_default_na=False, na_values=[])

    assert not meta["bookId"].isna().any(), "bookId содержит NaN"
    assert not (meta["bookId"] == "").any(), "bookId содержит пустые строки"
    assert not meta["title"].isna().any(), "title содержит NaN"

    assert meta["series_clean"].dtype == "object", "series_clean не object"
    assert meta["author_clean"].dtype == "object", "author_clean не object"
    assert all(
        isinstance(x, str) for x in meta["series_clean"]
    ), "series_clean содержит не-строки"
    assert all(
        isinstance(x, str) for x in meta["author_clean"]
    ), "author_clean содержит не-строки"

    for g in meta["genres_list"]:
        assert isinstance(g, str), f"genres_list не строка: {type(g)} — {g}"
        genres = parse_genre_str(g)
        assert isinstance(genres, set), f"parse_genre_str вернул не set: {type(genres)}"


def test_book_embeddings_multimodal_valid():
    INPUT_DIR = get_processed_data_dir()
    path_emb = os.path.join(INPUT_DIR, "book_embeddings_multimodal.npy")
    path_meta = os.path.join(INPUT_DIR, "books_meta_multimodal.csv")

    assert os.path.exists(path_emb), f"Файл не найден: {path_emb}"
    assert os.path.exists(path_meta), f"Файл не найден: {path_meta}"

    embeddings = np.load(path_emb)
    meta = pd.read_csv(path_meta, keep_default_na=False, na_values=[])

    assert len(embeddings) == len(
        meta
    ), f"Несоответствие длины: emb={len(embeddings)}, meta={len(meta)}"
    assert embeddings.dtype in (
        np.float32,
        np.float64,
    ), f"Тип эмбеддингов: {embeddings.dtype} (ожидался float32 или float64)"
    assert embeddings.ndim == 2, f"Эмбеддинги должны быть 2D, а не {embeddings.ndim}D"
    assert embeddings.shape[1] > 0, "Нулевая размерность эмбеддингов"

    assert not np.isnan(embeddings).any(), "Эмбеддинги содержат NaN"
    assert not np.isinf(embeddings).any(), "Эмбеддинги содержат Inf"


def test_contrastive_dataset_getitem():
    INPUT_DIR = get_processed_data_dir()
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
        num_negatives=2,
        seed=42,
    )

    item = dataset[0]
    assert len(item) == 6, "Ожидалось 6 элементов: anchor, pos_x4, negatives"
    anchor, pos_s, pos_a, pos_g, pos_d, negatives = item

    for i, t in enumerate(item):
        assert isinstance(t, torch.Tensor), f"Элемент {i} не тензор: {type(t)}"
        assert t.dtype == torch.float32, f"Элемент {i} не float32: {t.dtype}"

    emb_dim = book_embeddings.shape[1]
    assert anchor.shape == (emb_dim,), f"anchor shape: {anchor.shape}"
    assert pos_s.shape == (emb_dim,), f"pos_s shape: {pos_s.shape}"
    assert negatives.shape == (
        2,
        emb_dim,
    ), f"negatives shape: {negatives.shape}"

    assert not torch.allclose(
        anchor, torch.zeros_like(anchor)
    ), "anchor — нулевой вектор"
    assert not torch.allclose(
        negatives, torch.zeros_like(negatives)
    ), "negatives — нулевые"


def test_weighted_contrastive_loss_computes():
    batch_size, emb_dim, num_neg = 4, 128, 3
    anchor = torch.randn(batch_size, emb_dim)
    pos_s = torch.randn(batch_size, emb_dim)
    pos_a = torch.randn(batch_size, emb_dim)
    pos_g = torch.randn(batch_size, emb_dim)
    pos_d = torch.randn(batch_size, emb_dim)
    negatives = torch.randn(batch_size, num_neg, emb_dim)

    loss = weighted_contrastive_loss(
        anchor,
        pos_s,
        pos_a,
        pos_g,
        pos_d,
        negatives,
        margin=0.2,
    )

    assert isinstance(loss, torch.Tensor), "loss не тензор"
    assert loss.numel() == 1, "loss должен быть скаляром"
    assert not torch.isnan(loss), "loss = NaN"
    assert not torch.isinf(loss), "loss = Inf"
    assert loss >= 0, f"loss отрицательный: {loss}"


def test_model_encodes_embeddings():
    INPUT_DIR = get_processed_data_dir()
    book_embeddings = np.load(os.path.join(INPUT_DIR, "book_embeddings_multimodal.npy"))

    config = BookEncoderConfig(
        input_dim=book_embeddings.shape[1],
        hidden_dim=256,
        output_dim=64,
    )
    model = BookEncoderModel(config)

    model.eval()
    with torch.no_grad():
        batch = torch.tensor(book_embeddings[:10], dtype=torch.float32)
        out = model(batch)

    assert out.shape == (10, 64), f"Неверная форма: {out.shape}"
    assert not torch.isnan(out).any(), "Выход содержит NaN"
    assert not torch.isinf(out).any(), "Выход содержит Inf"


def test_recommendation():
    INPUT_DIR = get_processed_data_dir()
    books_meta = pd.read_csv(
        os.path.join(INPUT_DIR, "books_meta_multimodal.csv"),
        keep_default_na=False,
        na_values=[],
    )
    embeddings = np.load(os.path.join(INPUT_DIR, "book_embeddings_multimodal.npy"))

    user_book_ids = books_meta.head(10)["bookId"].tolist()
    user_ratings = [5] * 10
    user_history = list(zip(user_book_ids, user_ratings))

    recommender = LightweightRecommender(
        embeddings=embeddings,
        books_meta=books_meta,
        popularity_weight=0.3,
    )

    recommendations = recommender.recommend(
        user_history=user_history,
        k=5,
        exclude_book_ids=user_book_ids,
    )

    assert isinstance(recommendations, list), "Рекомендации не список"
    assert (
        len(recommendations) == 5
    ), f"Ожидалось 5 рекомендаций, получено {len(recommendations)}"

    for i, rec in enumerate(recommendations):
        assert isinstance(rec, dict), f"Рекомендация #{i} не словарь"
        assert "bookId" in rec, f"Нет bookId в #{i}"
        assert "title" in rec, f"Нет title в #{i}"
        assert "predicted_rating" in rec, f"Нет predicted_rating в #{i}"

        assert (
            1.0 <= rec["predicted_rating"] <= 5.0
        ), f"Рейтинг вне [1,5]: {rec['predicted_rating']}"

        assert (
            rec["bookId"] not in user_book_ids
        ), f"Рекомендована книга из истории: {rec['bookId']}"

    cand_id = recommendations[0]["bookId"]
    pred = recommender.predict_rating(user_history, cand_id)
    assert 1.0 <= pred <= 5.0, f"predict_rating вне [1,5]: {pred}"
