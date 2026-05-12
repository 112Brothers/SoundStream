import torch.nn.functional as F


def adversarial_d_loss(real_logits, fake_logits):
    """Hinge adversarial loss for discriminator"""
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / len(real_logits)

def adversarial_g_loss(fake_logits):
    """Hinge adversarial loss for generator"""
    loss = 0.0
    for logit in fake_logits:
        loss = loss + F.relu(1.0 - logit).mean()
    return loss / len(fake_logits)

def feature_matching_loss(real_feats, fake_feats, num_last_layers=2):
    """L1 feature matching over the last N layers of each discriminator
    """
    loss = 0.0
    n = 0
    for real_list, fake_list in zip(real_feats, fake_feats):
        if num_last_layers is not None:
            real_list = real_list[-num_last_layers:]
            fake_list = fake_list[-num_last_layers:]
        for rf, ff in zip(real_list, fake_list):
            min_len = min(rf.shape[-1], ff.shape[-1])
            loss = loss + F.l1_loss(ff[..., :min_len], rf[..., :min_len].detach())
            n += 1
    return loss / max(n, 1)
