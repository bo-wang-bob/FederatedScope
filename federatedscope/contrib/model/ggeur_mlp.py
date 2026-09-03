"""
GGEUR_Clip MLP Model

Simple MLP classifier for CLIP feature embeddings.
"""

import torch
import torch.nn as nn

from federatedscope.register import register_model


class GGEURMLP(nn.Module):
    """
    MLP classifier for GGEUR_Clip.
    Works with pre-extracted CLIP embeddings.
    """

    def __init__(self, input_dim=512, num_classes=65, hidden_dim=0, dropout=0.0):
        super(GGEURMLP, self).__init__()

        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Simple linear classifier
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def build_ggeur_mlp(model_config, local_data=None):
    """Build GGEUR_Clip MLP model based on config."""
    configured_input_dim = model_config.input_dim \
        if hasattr(model_config, 'input_dim') else 0
    input_dim = configured_input_dim if configured_input_dim > 0 else 512
    num_classes = model_config.num_classes if hasattr(model_config, 'num_classes') else 65
    hidden_dim = model_config.hidden_dim if hasattr(model_config, 'hidden_dim') else 0
    dropout = model_config.dropout if hasattr(model_config, 'dropout') else 0.0

    model = GGEURMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    return model


def call_ggeur_mlp(model_config, local_data=None):
    """Factory function for GGEUR_Clip MLP model."""
    if model_config.type.lower() == 'ggeur_mlp':
        return build_ggeur_mlp(model_config, local_data)
    return None


register_model('ggeur_mlp', call_ggeur_mlp)
