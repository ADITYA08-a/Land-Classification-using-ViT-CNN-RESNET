import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import CropData
from Task2_1_utils import hyperparameters as hp, build_deit, focal_loss, eval_model

print(f"PyTorch Version: {torch.__version__}")

parser = argparse.ArgumentParser(description="Task 2.1: Fine-tuning DeiT-3 (Training)")
parser.add_argument("--data_dir", type=str, default="./A3_Dataset")
parser.add_argument("--batch_size", type=int, default=hp['batch_size'])
parser.add_argument("--epochs", type=int, default=hp['no_of_epochs'])
parser.add_argument("--lr", type=float, default=hp['LR'])
parser.add_argument("--gamma", type=float, default=hp['gamma'], help="Focusing parameter for Focal Loss") 
parser.add_argument("--alpha", type=float, default=hp['alpha'], help="Balancing factor for Focal Loss")
parser.add_argument("--loss_type", type=str, choices=["ce", "focal"], default="ce")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(" Execution of Task 2.1 Training starts  ")
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current device: {device}")

def run_training(name, criterion_func):
    print(f"\nTraining {name} GOING ON !!!")
    model = build_deit(pretrained=True, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_auc, best_wts = 0.0, None
    train_losses = []
    
    for ep in range(args.epochs):
        model.train()
        epoch_loss = 0
        for img, lbl in loaders['train']:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            
            loss = criterion_func(model(img), lbl)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loaders['train'])
        train_losses.append(avg_loss)
        
        # Validation evaluation capturing all metrics
        acc, f1, auc = eval_model(model, loaders['val'], device)
        print(f"Epoch {ep+1} || Loss: {avg_loss:.4f} || Accuracy: {acc:.4f} || F1 Score: {f1:.4f} || Val AUC: {auc:.4f}")
        
        # Early STOPPING check BASED ON AUC
        if auc > best_auc:
            best_auc = auc
            best_wts = copy.deepcopy(model.state_dict())
            print(f" --> Saved new best weights (AUC: {auc:.4f})")
            
    filename = f"task2_1_best_deit_{args.loss_type}.pth"
    torch.save(best_wts, filename)
    print(f"Training Complete. Best weights saved to {filename}")
    
    return train_losses

if __name__ == "__main__":
    ds = args.data_dir
    loaders = {
        'train': DataLoader(CropData(ds, f"{ds}/train.csv"), args.batch_size, shuffle=True, pin_memory=True),
        'val': DataLoader(CropData(ds, f"{ds}/validation.csv"), args.batch_size, shuffle=False, pin_memory=True)
    }
    
    ce_losses, foc_losses = [], []
    
    if args.loss_type in ["ce", "both"]:
        ce_losses = run_training("Cross-Entropy Loss", nn.CrossEntropyLoss())
        
    if args.loss_type in ["focal", "both"]:
        focal_criterion = lambda i, t: focal_loss(i, t, args.gamma, args.alpha)
        foc_losses = run_training(f"Focal Loss (Alpha={args.alpha}, Gamma={args.gamma})", focal_criterion)

    if ce_losses or foc_losses:
        plt.figure(figsize=(10, 6))
        if ce_losses: 
            plt.plot(range(1, args.epochs + 1), ce_losses, 'b-o', label='Cross-Entropy')
        if foc_losses: 
            plt.plot(range(1, args.epochs + 1), foc_losses, 'r-s', 
                     label=f'Focal Loss ($\\alpha={args.alpha}, \\gamma={args.gamma}$)')
        
        plt.title('Training Loss: Task 2.1 Ablation Study')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("task2_1_loss_curve.png")
        print("\nPlot saved as task2_1_loss_curve.png")