#!/bin/bash
set -e

if [ ! -f .dvc/config.local ] && [ -n "$AWS_ACCESS_KEY_ID" ]; then
    echo "Configuring DVC remote credentials from Env Vars..."
    dvc remote modify --local yandex-s3 access_key_id "$AWS_ACCESS_KEY_ID"
    dvc remote modify --local yandex-s3 secret_access_key "$AWS_SECRET_ACCESS_KEY"
fi

echo "Pulling model and data..."
dvc pull processed_data/books_meta_multimodal.csv
dvc pull outputs/book_encoder_contrastive_256/book_embeddings_contrastive_256.npy
dvc pull outputs/book_encoder_contrastive_256/model.safetensors
dvc pull outputs/book_encoder_contrastive_256/config.json

echo "Running command..."
exec "$@"
