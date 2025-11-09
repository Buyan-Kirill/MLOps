#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import numpy as np
import torch
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.encoder import BookEncoderModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/encode.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    INPUT_DIR = config["paths"]["processed_data_dir"]
    OUTPUT_DIR = config["paths"]["outputs_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Загрузка данных...")
    book_embeddings = np.load(os.path.join(INPUT_DIR, "book_embeddings_multimodal.npy"))

    encoder_path = os.path.join(
        OUTPUT_DIR,
        f"book_encoder_contrastive_{config['training']['output_dim']}",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")

    logger.info(f"Загружаем обученную модель: {encoder_path}")
    model = BookEncoderModel.from_pretrained(encoder_path).to(device)
    model.eval()

    all_embs = []
    batch_size = config["training"]["batch_size"]
    logger.info("Генерация контрастивных эмбеддингов...")
    with torch.no_grad():
        for i in tqdm(
            range(0, len(book_embeddings), batch_size),
            desc="Кодирование",
        ):
            batch = torch.tensor(
                book_embeddings[i : i + batch_size],
                dtype=torch.float32,
                device=device,
            )
            emb = model(batch).cpu().numpy()
            all_embs.append(emb)
    all_embs = np.vstack(all_embs)

    output_path = os.path.join(
        encoder_path,
        f"book_embeddings_contrastive_{config['training']['output_dim']}.npy",
    )
    np.save(output_path, all_embs)
    logger.info(f"Эмбеддинги сохранены: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
