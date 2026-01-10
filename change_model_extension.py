import torch
import os
import sys
import yaml

sys.path.append(os.getcwd())
try:
    from src.encoder import BookEncoderModel
except ImportError:
    sys.path.append(".")
    from src.encoder import BookEncoderModel


def export(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dim = config["training"]["output_dim"]
    model_dir = os.path.join(
        config["paths"]["outputs_dir"], f"book_encoder_contrastive_{out_dim}"
    )

    print(f"Loading from {model_dir}...")
    model = BookEncoderModel.from_pretrained(model_dir)
    model.eval()

    # Сохраняем веса
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    print(f"Saved model.pt to {model_dir}")


if __name__ == "__main__":
    export("configs/default.yaml")
