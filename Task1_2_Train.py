import argparse
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from load_data import CropData
from utils import hyperparameters as hp
from Task1_2_utils import build_se_resnet, focal_loss, eval_model

print(f"PyTorch Version: {torch.__version__}")
parser = argparse.ArgumentParser(description="Task 1.2: Train SE-ResNet18")
parser.add_argument("--data_dir", type=str, default="./A3_Dataset")
parser.add_argument("--batch_size", type=int, default=hp.get('batch_size', 32))
parser.add_argument("--epochs", type=int, default=hp.get('epochs', 10))
parser.add_argument("--lr", type=float, default=hp.get('lr', 1e-4))
parser.add_argument("--gamma", type=float, default=hp.get('gamma', 2.1105))
parser.add_argument("--pretrained", type=int, choices=[0, 1], default=int(hp.get('PT', 1)))
parser.add_argument("--alpha", type=float, default=hp.get('alpha', 0.25)) # Ensures a default fallback if 'alpha' is missing in hp
parser.add_argument("--loss_type", type=str, choices=["ce", "focal", "both"], default="both")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Execution of Task 1.2 Training starts ")
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current device: {device}")

ds = args.data_dir
loaders = {
    'train': DataLoader(CropData(ds, f"{ds}/train.csv"), args.batch_size, shuffle=True, pin_memory=True),
    'val': DataLoader(CropData(ds, f"{ds}/validation.csv"), args.batch_size, shuffle=False, pin_memory=True)
}

def run_train(name, criterion, is_focal=False):
    print(f"\n {name} ")
    m = build_se_resnet(pretrained=bool(args.pretrained)).to(device)
    opt = optim.Adam(m.parameters(), lr=args.lr)
    best_auc, best_wts, hist = 0.0, None, []

    for ep in range(args.epochs):
        m.train()
        loss_sum = 0
        for img, lbl in loaders['train']:
            img, lbl = img.to(device), lbl.to(device)
            opt.zero_grad()  
            if is_focal:
                loss = criterion(m(img), lbl, gamma=args.gamma, alpha=args.alpha)
            else:
                loss = criterion(m(img), lbl)                
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        hist.append(loss_sum / len(loaders['train']))
        acc, f1, auc = eval_model(m, loaders['val'], device)
        print(f"At Ep {ep+1} Loss: {hist[-1]:.4f} || Accuracy: {acc:.4f} || AUC Score: {auc:.4f} || F1 Score: {f1:.4f}")
        if auc > best_auc: 
            best_auc, best_wts = auc, copy.deepcopy(m.state_dict())

    model_filename = f"Task1_2_best_model_{name.replace(' ', '_').lower()}.pth"
    torch.save(best_wts, model_filename)
    print(f"{name} Best weights saved to {model_filename}")
    return hist

if __name__ == "__main__":
    ce_losses, foc_losses = [], []
    if args.loss_type in ["ce", "both"]:
        ce_losses = run_train("Cross-Entropy Loss", nn.CrossEntropyLoss(), is_focal=False)
    if args.loss_type in ["focal", "both"]:
        foc_losses = run_train("Focal Loss", focal_loss, is_focal=True)
    if ce_losses or foc_losses:
        plt.figure(figsize=(9, 5))
        if ce_losses: plt.plot(range(1, args.epochs + 1), ce_losses, 'b-o', label='Cross-Entropy')
        if foc_losses: plt.plot(range(1, args.epochs + 1), foc_losses, 'r-s', label=f'Focal Loss (alpha={args.alpha}, gamma={args.gamma})')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.savefig('training_loss_plot.png')
        plt.show()