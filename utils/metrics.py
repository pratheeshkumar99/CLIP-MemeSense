import torch
import torchmetrics

class MemeMetrics:
    def __init__(self,num_classes,device):
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
        self.auroc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes).to(device)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
        
    def compute(self, logits, labels):
        preds_proxy = torch.sigmoid(logits)
        _, preds = logits.max(1)
        
        return {
            'accuracy': self.accuracy(preds, labels),
            'auroc': self.auroc(preds_proxy, labels),
            'f1': self.f1(preds, labels)
        }
    
    def reset(self):
        self.accuracy.reset()
        self.auroc.reset()
        self.f1.reset()

class MetricTracker:
    def __init__(self):
        self.metrics = {'train': {}, 'val': {}, 'test': {}}
    
    def update(self, split, metrics):
        for k, v in metrics.items():
            if k not in self.metrics[split]:
                self.metrics[split][k] = []
            self.metrics[split][k].append(v)
    
    def get_epoch_metrics(self, split):
        return {k: sum(v)/len(v) for k, v in self.metrics[split].items()}
    
    def reset(self):
        for split in self.metrics:
            self.metrics[split] = {}