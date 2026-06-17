import torch, argparse
from torch.utils.data import DataLoader
from load_data import CropData
from Task1_1_utils import build_resnet18, evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./A3_Dataset")
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    test_loader = DataLoader(CropData(args.data_dir, f"{args.data_dir}/test.csv"), 32)
    model = build_resnet18(pretrained=False, device=device)
    model.load_state_dict(torch.load(args.model_path))
    
    acc, f1, auc = evaluate_model(model, test_loader, device)
    print(f"\nTASK 1.1 FINAL RESULTS ({args.model_path}):")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")