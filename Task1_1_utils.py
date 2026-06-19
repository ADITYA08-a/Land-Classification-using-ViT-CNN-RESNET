# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

hyperparameters = {
    'batch_size': 32,
    'no_of_epochs': 10,
    'LR': 1e-4,
    'gamma': 2.0,
    'alpha': 0.25,
    'PT': True # Pre-trained
}

def build_resnet18(num_classes=10, pretrained=True, device='cuda'):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model.to(device)

def focal_loss(inp, tgt, gamma=hyperparameters['gamma'], alpha=hyperparameters['alpha']):
    ce_loss = F.cross_entropy(inp, tgt, reduction='none')
    p_t = torch.exp(-ce_loss)
    loss = alpha * ((1 - p_t) ** gamma) * ce_loss
    return loss.mean()

def evaluate_model(model, loader, device):
    model.eval()
    predictions, labels_list, probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            prob = F.softmax(outputs, dim=1)
            preds = torch.argmax(prob, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            probs.extend(prob.cpu().numpy())
            
    acc = accuracy_score(labels_list, predictions)
    f1 = f1_score(labels_list, predictions, average='macro')
    auc = roc_auc_score(labels_list, probs, multi_class='ovr')
    return acc, f1, auc