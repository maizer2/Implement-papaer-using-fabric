import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", type=str, default="models.mlp.MLP.LitMultiLayerPerceptron", help="Please refer to the model list in the README.md file.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path of dataset")
    parser.add_argument("--img_size", type=tuple, default=(32, 32), help="The image is resized to the suggested size.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_workers", type=tuple, default=6, help="Number of DataLoader worker.")
    opt = parser.parse_args()
    return opt