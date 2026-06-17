import torch, argparse
from torch.utils.data import DataLoader
from load_data import CropData
from Task2_2_utils import build_deit_dyt, evaluate_model, hyperparameters as hp

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./A3_Dataset")
parser.add_argument("--batch_size", type=int, default=hp['batch_size'])
parser.add_argument("--model_path", type=str, required=True, help="Path to best .pth file")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    test_loader = DataLoader(CropData(args.data_dir, f"{args.data_dir}/test.csv"), args.batch_size)
    model = build_deit_dyt(pretrained=False, device=device)
    model.load_state_dict(torch.load(args.model_path))
    acc, f1, auc = evaluate_model(model, test_loader, device)
    
    print(f"TASK 2.2 FINAL TEST RESULTS")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    