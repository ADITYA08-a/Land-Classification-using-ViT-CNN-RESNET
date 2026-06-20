import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.sqz = nn.AdaptiveAvgPool2d(1)
        val = channels//reduction_ratio
        self.exc = nn.Sequential(
            nn.Linear(channels, val, bias=False), 
            nn.ReLU(True), 
            nn.Linear(val, channels, bias=False), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.exc(self.sqz(x).view(b, c)).view(b, c, 1, 1).expand_as(x)

def build_se_resnet(pretrained=True, num_classes=10):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    
    # Append SE blocks to the end of ResNet's native layers
    m.layer1 = nn.Sequential(m.layer1, SqueezeExcite(64))
    m.layer2 = nn.Sequential(m.layer2, SqueezeExcite(128))
    m.layer3 = nn.Sequential(m.layer3, SqueezeExcite(256))
    m.layer4 = nn.Sequential(m.layer4, SqueezeExcite(512))
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def focal_loss(inp, tgt, gamma=2.0, alpha=0.25):
    ce = F.cross_entropy(inp, tgt, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * ((1 - pt) ** gamma) * ce).mean()

def eval_model(m, loader, device):
    m.eval()
    p, y, probs = [], [], []
    with torch.no_grad():
        for img, lbl in loader:
            out = m(img.to(device))
            p.extend(torch.argmax(out, 1).cpu().numpy())
            probs.extend(F.softmax(out, dim=1).cpu().numpy())
            y.extend(lbl.numpy())
            
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average='macro')
    auc = roc_auc_score(y, probs, multi_class='ovr')
    return acc, f1, auc