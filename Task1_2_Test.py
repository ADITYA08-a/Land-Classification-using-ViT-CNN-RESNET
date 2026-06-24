import argparse
import os
import torch
from torch.utils.data import DataLoader
from load_data import CropData
from utils import hyperparameters as hp
from Task1_2_utils import build_se_resnet, eval_model

parser = argparse.ArgumentParser(description="Task 1.2: Test SE-ResNet18")
parser.add_argument("--data_dir", type=str, default="./A3_Dataset")
parser.add_argument("--batch_size", type=int, default=hp.get('batch_size', 32))
parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth weights file")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("  Execution of Task 1.2 Testing starts ")
ds = args.data_dir
test_loader = DataLoader(CropData(ds, f"{ds}/test.csv"), args.batch_size, shuffle=False, pin_memory=True)

def run_test():
    print(f"\nLoading weights from {args.model_path}...")
    # Initializing the model  
    m = build_se_resnet(pretrained=bool(hp.get('PT', 1)))
    
    # Check if the file actually exists before trying to load it
    if not os.path.exists(args.model_path):
        print(f"Error: {args.model_path} not found. Please verify the path or run training first.")
        return

    # Load the weights into the model
    m.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    m = m.to(device)
    t_acc, t_f1, t_auc = eval_model(m, test_loader, device)
    print("\n")
    print(f"TASK 1.2 FINAL RESULTS")
    print(f"File: {os.path.basename(args.model_path)}")
    print(f"Accuracy:  {t_acc:.4f}")
    print(f"F1 Score:  {t_f1:.4f}")
    print(f"AUC Score: {t_auc:.4f}")
    

if __name__ == "__main__":
    run_test()