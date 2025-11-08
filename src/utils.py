import random
import numpy as np
import torch
import re
import logging
import pandas as pd
import ast
from typing import Any, Set, List, Union


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
        logger.warning(
            f"clean_html: не строка ({type(text)}) — возвращаем ''"
        )
        return ""
    return re.sub(r"<.*?>", "", text).strip()


def normalize_title(title: Any) -> str:
    if not isinstance(title, str):
        logger.warning(
            f"normalize_title: не строка ({type(title)}) — возвращаем ''"
        )
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
            return [
                g.strip().lower()
                for g in genres_list
                if isinstance(g, str)
            ]
    except (ValueError, SyntaxError):
        pass
    return []