import torch
from torch import nn

from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

class STOIMetric(nn.Module):
    def __init__(self, sample_rate=16000, extended=False):
        super().__init__()
        self.metric = ShortTimeObjectiveIntelligibility(
            fs=sample_rate,
            extended=extended,
        )

    def forward(self, preds, target):
        preds = preds.squeeze(1)
        target = target.squeeze(1)
        return self.metric(preds, target)

class NISQAMetric(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sample_rate)

    def forward(self, preds):
        preds = preds.squeeze(1)
        return self.metric(preds)

class MetricCollection(nn.Module):
    def __init__(self, sample_rate=16000, use_nisqa=False):
        super().__init__()
        self.stoi = STOIMetric(sample_rate=sample_rate)
        self.nisqa = NISQAMetric(sample_rate=sample_rate) if use_nisqa else None

    @torch.no_grad()
    def forward(self, preds, target):
        metrics = {
            "stoi": self.stoi(preds, target),
        }
        if self.nisqa is not None:
            metrics["nisqa"] = self.nisqa(preds)
        return metrics
