import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

hyperparameters = {
    'batch_size': 32,
    'no_of_epochs': 10,
    'LR': 1e-4,
    'gamma': 2.0,
    'alpha': 0.25
}

def build_deit(num_classes=10, pretrained=True, device='cuda'):
    model = timm.create_model('deit3_small_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model.to(device)

def focal_loss(inp, tgt, gamma, alpha):
    ce_loss = F.cross_entropy(inp, tgt, reduction='none')
    p_t = torch.exp(-ce_loss) 
    loss = alpha * ((1 - p_t) ** gamma) * ce_loss
    return loss.mean()

def eval_model(m, loader, device):
    m.eval()
    p, y, probs = [], [], []
    with torch.no_grad():
        for img, lbl in loader:
            out = m(img.to(device))
            p.extend(torch.argmax(out, 1).cpu().numpy())
            probs.extend(torch.softmax(out, dim=1).cpu().numpy())
            y.extend(lbl.numpy())
            
    return (accuracy_score(y, p), 
            f1_score(y, p, average='macro'), 
            roc_auc_score(y, probs, multi_class='ovr'))