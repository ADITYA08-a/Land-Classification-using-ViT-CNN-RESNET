import torch, argparse, copy, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from load_data import CropData
from Task2_2_utils import hyperparameters as hp, build_deit_dyt, focal_loss, evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./A3_Dataset")
parser.add_argument("--epochs", type=int, default=hp['no_of_epochs'])
parser.add_argument("--batch_size", type=int, default=hp['batch_size'])
parser.add_argument("--lr", type=float, default=hp['LR'])
parser.add_argument("--gamma", type=float, default=hp['gamma'])
parser.add_argument("--alpha", type=float, default=hp['alpha'])
parser.add_argument("--loss_type", type=str, choices=["ce", "focal"], default="ce")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Execution of Task 2.2 Training starts ")
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Current device: {device}")
if __name__ == "__main__":
    loaders = {
        'train': DataLoader(CropData(args.data_dir, f"{args.data_dir}/train.csv"), args.batch_size, shuffle=True),
        'val': DataLoader(CropData(args.data_dir, f"{args.data_dir}/validation.csv"), args.batch_size)
    }

    model = build_deit_dyt(pretrained=True, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.loss_type == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = lambda i, t: focal_loss(i, t, args.gamma, args.alpha)

    best_auc, history = 0, []
    print(f"Training Task 2.2 (DeiT + DyT) || Loss: {args.loss_type} || Gamma: {args.gamma if args.loss_type=='focal' else 'N/A'} ||  Alpha: {args.alpha if args.loss_type=='focal' else 'N/A'}")

    for ep in range(args.epochs):
        model.train()
        loss_sum = 0
        for img, lbl in loaders['train']:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            
            outputs = model(img)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        
        avg_loss = loss_sum / len(loaders['train'])
        history.append(avg_loss)
        
        acc, f1, auc = evaluate_model(model, loaders['val'], device)
        print(f"Epoch {ep+1} || Loss: {avg_loss:.4f} || Acc: {acc:.4f} || Val AUC: {auc:.4f} || Val F1: {f1:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"Task2_2_best_{args.loss_type}.pth")
            print(f"Saved Best Model of score (AUC: {auc:.4f})")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), history, marker='o')
    plt.title(f"Task 2.2 DyT Training ({args.loss_type})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"Task2_2_{args.loss_type}_plot.png")
    plt.show()