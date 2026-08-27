"""
Lightweight RNN/LSTM classifiers for GGEUR text embeddings.

The pretrained text encoder is used only as a frozen feature extractor.  These
models are the trainable heads used by the GGEUR federated pipeline.
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from federatedscope.register import register_model


class GGEURTextRNNClassifier(nn.Module):
    """Sequence classifier over fixed-size embedding features."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 rnn_type: str = 'lstm'):
        super().__init__()

        rnn_type = str(rnn_type).lower()
        if rnn_type not in {'rnn', 'lstm'}:
            raise ValueError(
                f"Unsupported rnn_type={rnn_type}, expected 'rnn' or 'lstm'")

        num_layers = max(int(num_layers), 1)
        dropout = float(dropout)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.RNN if rnn_type == 'rnn' else nn.LSTM
        self.rnn = rnn_cls(input_size=int(input_dim),
                           hidden_size=int(hidden_dim),
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=rnn_dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f'Expected input dim 2 or 3, got {x.dim()}')

        out, _ = self.rnn(x)
        features = out[:, -1, :]
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits


def _build(model_config, input_shape, rnn_type):
    input_dim = int(getattr(model_config, 'in_channels', 0))
    if input_dim <= 0 and input_shape is not None:
        input_dim = int(input_shape[-1])

    hidden_dim = int(getattr(model_config, 'hidden', 256))
    num_layers = int(getattr(model_config, 'layer', 1))
    dropout = float(getattr(model_config, 'dropout', 0.0))
    num_classes = int(
        getattr(model_config, 'num_classes',
                getattr(model_config, 'out_channels', 2)))

    return GGEURTextRNNClassifier(input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_classes=num_classes,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  rnn_type=rnn_type)


def call_ggeur_text_rnn(model_config, input_shape):
    model_type = str(model_config.type).lower()
    if model_type == 'ggeur_rnn':
        return _build(model_config, input_shape, rnn_type='rnn')
    if model_type == 'ggeur_lstm':
        return _build(model_config, input_shape, rnn_type='lstm')
    return None


register_model('ggeur_rnn', call_ggeur_text_rnn)
register_model('ggeur_lstm', call_ggeur_text_rnn)
