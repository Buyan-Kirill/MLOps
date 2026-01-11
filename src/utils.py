#!/usr/bin/env python3
import random
import numpy as np
import torch
import re
import logging
import pandas as pd
import ast
from typing import Any, Set, List, Union
from difflib import get_close_matches


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__name__)


def clean_html(text: Any) -> str:
    if not isinstance(text, str):
        logger.warning(f"clean_html: не строка ({type(text)}) — возвращаем ''")
        return ""
    return re.sub(r"<.*?>", "", text).strip()


def normalize_title(title: Any) -> str:
    if not isinstance(title, str):
        logger.warning(f"normalize_title: не строка ({type(title)}) — возвращаем ''")
        return ""
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip().lower()


def safe_str(x: Any) -> str:
    if pd.isna(x) or x is None or str(x).lower() in {"nan", "none", ""}:
        return ""
    return str(x).strip()


def parse_genre_str(g: str) -> Set[str]:
    if pd.isna(g) or g is None or g == "":
        return set()
    if isinstance(g, str):
        return set(g.split(","))
    return set(str(g).split(","))


def parse_genres(genres_str: Union[str, Any]) -> List[str]:
    if not isinstance(genres_str, str) or not genres_str.strip():
        return []
    try:
        genres_list = ast.literal_eval(genres_str)
        if isinstance(genres_list, list):
            return [g.strip().lower() for g in genres_list if isinstance(g, str)]
    except (ValueError, SyntaxError):
        pass
    return []


def robust_parse_genres(x):
    if not isinstance(x, str) or not x.strip():
        return []
    x = x.strip()
    return [g.strip() for g in x.split(",") if g.strip()]


def find_book_id_by_title_author(title: str, author: str, books_meta: pd.DataFrame):
    t_norm = str(title).lower().strip()
    a_norm = str(author).lower().strip()

    exact = books_meta[
        (books_meta["_title_norm"] == t_norm) & (books_meta["_author_norm"] == a_norm)
    ]
    if not exact.empty:
        return str(exact.iloc[0]["bookId"]), "Exact match"

    by_author = books_meta[
        books_meta["_author_norm"].str.contains(a_norm, na=False, regex=False)
    ]
    if len(by_author) > 0:
        titles = by_author["_title_norm"].tolist()
        matches = get_close_matches(t_norm, titles, n=1, cutoff=0.6)
        if matches:
            cand = by_author[by_author["_title_norm"] == matches[0]].iloc[0]
            return str(cand["bookId"]), "Fuzzy match by author"

    return None, "Not found"
