import argparse
import torch
from torch.utils.data import DataLoader
from load_data import CropData
from Task2_1_utils import hyperparameters as hp, build_deit, eval_model

parser = argparse.ArgumentParser(description="Task 2.1: Fine-tuning DeiT-3 (Testing)")
parser.add_argument("--data_dir", type=str, default="./A3_Dataset")
parser.add_argument("--batch_size", type=int, default=hp['batch_size'])
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .pth file")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f" Execution of Task 2.1 Testing starts ")
    
    ds = args.data_dir
    test_loader = DataLoader(CropData(ds, f"{ds}/test.csv"), args.batch_size, shuffle=False, pin_memory=True)
    print(f"\nLoading weights from {args.model_path}...")
    model = build_deit(pretrained=False, device=device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: {args.model_path} not found. Please run training first.")
        exit()

    # Evaluate all metrics
    t_acc, t_f1, t_auc = eval_model(model, test_loader, device)
    
    print("\n")
    print(" TASK 2.1 FINAL TEST METRICS ")
    print("\n")
    print(f" Accuracy : {t_acc:.4f}")
    print(f" F1 Score : {t_f1:.4f}")
    print(f" AUC Score: {t_auc:.4f}")
    