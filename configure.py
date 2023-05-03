import argparse
import os

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", type=str, default="models.mlp.MLP.LitMultiLayerPerceptron", help="Please refer to the model list in the README.md file.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path of dataset.")
    parser.add_argument("--log_path", type=str, default="checkpoints/mlp", help="Path of lightning logs.")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/mlp/lightning_logs/version_2/checkpoints/epoch=11-step=192000.ckpt", help="Path of ckpt.")
    parser.add_argument("--img_size", type=tuple, default=(32, 32), help="The image is resized to the suggested size.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=tuple, default=6, help="Number of DataLoader worker.")
    opt = parser.parse_args()

    os.makedirs(opt.log_path, exist_ok=True)
    return opt