import torch
import numpy as np

try:
    from utils import clean_html, safe_str, robust_parse_genres
except ImportError:
    from src.utils import clean_html, safe_str, robust_parse_genres


class ColdStartProcessor:
    def __init__(self, model, base_model, mlb, device="cpu"):
        self.device = device
        self.base_model = base_model
        self.mlb = mlb
        self.encoder_model = model

    def process_row(self, row_dict):
        desc_clean = clean_html(str(row_dict.get("description", "")))
        author_clean = safe_str(str(row_dict.get("author", "")))
        series_clean = safe_str(str(row_dict.get("series", "")))

        raw_genres = str(row_dict.get("genres", ""))
        if isinstance(row_dict.get("genres"), list):
            genres_parsed = row_dict.get("genres")
        else:
            genres_parsed = robust_parse_genres(raw_genres)

        with torch.no_grad():
            series_emb = self.base_model.encode([series_clean])
            author_emb = self.base_model.encode([author_clean])
            desc_emb = self.base_model.encode([desc_clean])

        genre_vec = self.mlb.transform([genres_parsed]).astype(np.float32)

        combined = np.hstack([series_emb, author_emb, desc_emb, genre_vec])
        input_tensor = torch.tensor(combined, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            final_emb = self.encoder_model(input_tensor).cpu().numpy()

        return final_emb[0]
