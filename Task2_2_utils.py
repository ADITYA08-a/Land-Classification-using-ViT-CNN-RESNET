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

class DynamicTanh(nn.Module):
    def __init__(self, dim, alpha_init=0.25):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), alpha_init))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.gamma + self.beta

def replace_norm_with_dyt(model):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            dim = module.normalized_shape[0]
            setattr(model, name, DynamicTanh(dim))
        else:
            replace_norm_with_dyt(module)

def build_deit_dyt(num_classes=10, pretrained=True, device='cuda'):
    model = timm.create_model('deit3_small_patch16_224', pretrained=pretrained)
    replace_norm_with_dyt(model) # Surgery happens here
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model.to(device)

def focal_loss(inp, tgt, gamma=hyperparameters['gamma'], alpha=hyperparameters['alpha']):
    ce = F.cross_entropy(inp, tgt, reduction='none')
    p_t = torch.exp(-ce)
    return (alpha * ((1 - p_t) ** gamma) * ce).mean()

def evaluate_model(model, loader, device):
    model.eval()
    p, y, probs = [], [], []
    with torch.no_grad():
        for img, lbl in loader:
            out = model(img.to(device))
            p.extend(torch.argmax(out, 1).cpu().numpy())
            probs.extend(F.softmax(out, dim=1).cpu().numpy())
            y.extend(lbl.numpy())
    return accuracy_score(y, p), f1_score(y, p, average='macro'), roc_auc_score(y, probs, multi_class='ovr')