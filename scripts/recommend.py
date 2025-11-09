#!/usr/bin/env python3
import os
import argparse
import sys
import pandas as pd
import numpy as np
import yaml
from difflib import get_close_matches
from typing import Tuple, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.recommender import LightweightRecommender


def load_model_and_data(config_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    PROCESSED_DIR = config["paths"]["processed_data_dir"]
    OUTPUT_DIR = config["paths"]["outputs_dir"]
    OUTPUT_DIM = config["training"]["output_dim"]

    embeddings_path = os.path.join(
        OUTPUT_DIR,
        f"book_encoder_contrastive_{OUTPUT_DIM}",
        f"book_embeddings_contrastive_{OUTPUT_DIM}.npy",
    )
    meta_path = os.path.join(PROCESSED_DIR, "books_meta_multimodal.csv")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Эмбеддинги не найдены: {embeddings_path}. Сначала выполните: make train"
        )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Метаданные не найдены: {meta_path}. Сначала выполните: make embed"
        )

    embeddings = np.load(embeddings_path)
    books_meta = pd.read_csv(meta_path, keep_default_na=False, na_values=[])

    books_meta["_title_norm"] = books_meta["title"].str.lower().str.strip()
    books_meta["_author_norm"] = books_meta["author_clean"].str.lower().str.strip()

    return embeddings, books_meta


def parse_list_arg(arg: str) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(";")]


def find_book_id_by_title_author(
    title: str, author: str, books_meta: pd.DataFrame, threshold: float = 0.85
) -> Tuple[Optional[str], str]:
    title_norm = title.lower().strip()
    author_norm = author.lower().strip()

    exact = books_meta[
        (books_meta["_title_norm"] == title_norm)
        & (books_meta["_author_norm"] == author_norm)
    ]
    if not exact.empty:
        return (
            str(exact.iloc[0]["bookId"]),
            f"Точное совпадение: '{title}' — {author}",
        )

    by_author = books_meta[
        books_meta["_author_norm"].str.contains(author_norm, na=False, regex=False)
    ]
    if len(by_author) > 0:
        titles = by_author["_title_norm"].tolist()
        matches = get_close_matches(title_norm, titles, n=1, cutoff=0.6)
        if matches:
            candidate = by_author[by_author["_title_norm"] == matches[0]].iloc[0]
            return (
                str(candidate["bookId"]),
                f"Найдено похожее: '{candidate['title']}' "
                f"— {candidate['author_clean']}",
            )

    all_titles = books_meta["_title_norm"].tolist()
    matches = get_close_matches(title_norm, all_titles, n=1, cutoff=0.7)
    if matches:
        candidate = books_meta[books_meta["_title_norm"] == matches[0]].iloc[0]
        return (
            str(candidate["bookId"]),
            f"Найдено по названию: '{candidate['title']}' "
            f"— {candidate['author_clean']}",
        )

    return None, f"Не найдено: '{title}' — {author}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Рекомендация книг по названию и автору",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Путь к конфигу",
    )
    parser.add_argument(
        "--titles",
        type=str,
        required=True,
        help="Названия книг (через запятую)",
    )
    parser.add_argument(
        "--authors",
        type=str,
        required=True,
        help="Авторы книг (через запятую)",
    )
    parser.add_argument(
        "--ratings",
        type=str,
        required=True,
        help="Рейтинги (1-5, через запятую)",
    )
    parser.add_argument("--k", type=int, default=5, help="Сколько книг рекомендовать")
    parser.add_argument(
        "--popularity-weight",
        type=float,
        default=None,
        help="Вес популярности [0,1]",
    )
    args = parser.parse_args()

    titles = parse_list_arg(args.titles)
    authors = parse_list_arg(args.authors)
    ratings = [int(r) for r in parse_list_arg(args.ratings)]

    if not (len(titles) == len(authors) == len(ratings)):
        print("Ошибка: количество названий, авторов и рейтингов должно совпадать.")
        print(
            f"   Названий: {len(titles)}, "
            f"Авторов: {len(authors)}, "
            f"Рейтингов: {len(ratings)}"
        )
        sys.exit(1)

    try:
        embeddings, books_meta = load_model_and_data(args.config)

        if args.popularity_weight is not None:
            popularity_weight = args.popularity_weight
        else:
            with open(args.config) as f:
                config = yaml.safe_load(f)
            popularity_weight = config.get("recommend", {}).get(
                "popularity_weight", 0.3
            )

        print("Поиск книг по названию и автору...")
        user_history = []
        for title, author, rating in zip(titles, authors, ratings):
            book_id, msg = find_book_id_by_title_author(title, author, books_meta)
            print(f"  {msg}")
            if book_id is None:
                print(f"Пропуск: не найдена книга '{title}' — {author}")
                continue
            user_history.append((book_id, rating))

        if not user_history:
            print("Ни одна книга не найдена. Попробуйте уточнить названия или авторов.")
            sys.exit(1)

        recommender = LightweightRecommender(
            embeddings=embeddings,
            books_meta=books_meta,
            popularity_weight=popularity_weight,
        )

        recommendations = recommender.recommend(
            user_history=user_history,
            k=args.k,
            exclude_book_ids=[bid for bid, _ in user_history],
        )

        print("\n" + "=" * 70)
        print(f"Найдено {len(user_history)} книг. Рекомендации:")
        print("=" * 70)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} — {rec['author']}")
            print(
                f"Прогноз: {rec['predicted_rating']}, "
                f"сходство: {rec['similarity_to_user_profile']}"
            )
            print(f"ID: {rec['bookId']}\n")

        if not recommendations:
            print("Нет рекомендаций (попробуйте изменить рейтинги или увеличить k)")

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
