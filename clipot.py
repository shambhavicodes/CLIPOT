import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging  # Import logging
import copy
from collections import OrderedDict

# Initialize the logger
logging.basicConfig(level=logging.INFO, filename='clipot_implementation.log', filemode='w')
logger = logging.getLogger()

class ClipOTLoss(nn.Module):
    def __init__(self, prototypes, temperature=0.01, epsilon=0.7):
        super(ClipOTLoss, self).__init__()
        self.temperature = temperature
        self.prototypes = prototypes
        self.epsilon = epsilon

    def forward(self, features, logits):
        # Normalize prototypes
        self.prototypes = self.prototypes.to(features.device)

        with torch.no_grad():
            w = self.prototypes.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.data = w

        # Compute similarities and soft codes for all features
        similarities = torch.mm(features, self.prototypes.t()) / self.temperature
        soft_code = distributed_sinkhorn(similarities, epsilon=self.epsilon)

        # Compute cross-entropy loss between soft_code and logits
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(soft_code * log_probs, dim=1))

        return loss


def distributed_sinkhorn(similarities, epsilon=0.7, num_iters=3):
    Q = torch.exp(similarities / epsilon).t()
    Q = Q / Q.sum(dim=0, keepdim=True)  # Normalize each column

    K, B = Q.shape  # Number of clusters and batch size

    r = torch.ones(K).to(similarities.device) / K
    c = torch.ones(B).to(similarities.device) / B

    for _ in range(num_iters):
        u = Q.sum(dim=1)
        Q = Q * (r / (u + 1e-8)).unsqueeze(1)  # Row normalization

        v = Q.sum(dim=0)
        Q = Q * (c / (v + 1e-8)).unsqueeze(0)  # Column normalization

    Q = (Q / Q.sum(dim=0, keepdim=True)).t()

    if torch.isnan(Q).any():
        logger.error("NaNs detected in Q matrix after Sinkhorn normalization")
    return Q


class ClipOTMethod:
    def __init__(self, model, clipot_loss, text_embeddings, use_avg_embeddings, lr):
        self.model = model
        self.clipot_loss = clipot_loss
        self.text_embeddings = text_embeddings
        self.use_avg_embeddings = use_avg_embeddings
        self.device = next(self.model.parameters()).device

        self.model.backbone = self.set_ln_grads(self.model.backbone)
        params, _ = self.collect_ln_params(self.model.backbone)
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2, betas=(0.9, 0.999))

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

    @staticmethod
    def set_ln_grads(model):
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        logger.info(f"Prototypes before reset: {self.clipot_loss.prototypes}")
        if self.use_avg_embeddings:
            self.clipot_loss.prototypes = self.text_embeddings.mean(dim=1).to(self.device)
        else:
            self.clipot_loss.prototypes = self.text_embeddings.to(self.device)
        logger.info(f"Prototypes after reset: {self.clipot_loss.prototypes}")

    @staticmethod
    def collect_ln_params(model):
        params = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
        return params, None

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state
