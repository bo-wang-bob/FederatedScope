"""Portable inference extensions; old single-head checkpoints stay unchanged."""


def snapshot_domain_heads(heads):
    if not heads:
        return None
    return {'version': 1, 'kind': 'domain-linear-l2', 'domains': {
        str(domain): {key: value.detach().cpu().float().clone()
                      for key, value in state.items()}
        for domain, state in heads.items()}}


def classifier_for_domain(model, domain, extension, input_dim, num_classes):
    """Use the same per-domain L2-normalized linear head as GGEURServer.

Domain selection is supplied by dataset membership, never by the class label.
An absent extension preserves the original global-head evaluation exactly.
"""
    if extension is None:
        return model
    import torch
    import torch.nn.functional as functional
    if (not isinstance(extension, dict) or extension.get('version') != 1
            or extension.get('kind') != 'domain-linear-l2'
            or not isinstance(extension.get('domains'), dict)):
        raise ValueError('Unsupported checkpoint inference-head protocol')
    for state in extension['domains'].values():
        if (not isinstance(state, dict) or set(state) != {'weight', 'bias'}
                or not isinstance(state['weight'], torch.Tensor)
                or not isinstance(state['bias'], torch.Tensor)
                or state['weight'].shape != (num_classes, input_dim)
                or state['bias'].shape != (num_classes,)
                or not torch.isfinite(state['weight']).all()
                or not torch.isfinite(state['bias']).all()):
            raise ValueError('Invalid checkpoint domain classifier')
    state = extension['domains'].get(str(domain))
    if state is None:
        return model

    class DomainClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('weight', state['weight'].detach().cpu().float())
            self.register_buffer('bias', state['bias'].detach().cpu().float())

        def forward(self, features):
            norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
            return functional.linear(features / norms, self.weight, self.bias)

    return DomainClassifier().eval()
