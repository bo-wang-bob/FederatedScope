# GGEUR Five-Model Final Experiments

This directory groups the final cross-domain heterogeneous experiments by
dataset and model:

- `officehome_vit`, `officehome_cnn`, `officehome_mixer`
- `domainnet_vit`, `domainnet_cnn`, `domainnet_mixer`
- `mdsent_rnn`, `mdsent_lstm`

The RNN/LSTM groups use the Multi-Domain Sentiment dataset and frozen
`nlptown/bert-base-multilingual-uncased-sentiment` BERT features:

- Dataset: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
- Pretrained model: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

Each group contains method configs named `fedavg.yaml`, `fedprox.yaml`,
`fedproto.yaml`, `fedopt.yaml`, and `ggeur.yaml`. Vision groups also include
`moon.yaml`.

Run one case:

```bash
python scripts/run_ggeur_final_experiment.py --group officehome_vit --method ggeur
```

List cases:

```bash
python scripts/run_ggeur_final_experiment.py --list
```

Pass extra FederatedScope overrides after `--`:

```bash
python scripts/run_ggeur_final_experiment.py --group mdsent_rnn --method fedavg -- device 1
```

All configs still support direct launch:

```bash
python federatedscope/main.py --cfg scripts/example_configs/ggeur_final_5models/mdsent_lstm/ggeur.yaml
```

The original RNN/LSTM GGEUR entry points from
`feature/ggeur-backdoor-research` are kept as compatibility configs:

```bash
python federatedscope/main.py --cfg scripts/example_configs/ggeur_baseline_rnn/ggeur_rnn_mdsent_rating4.yaml
python federatedscope/main.py --cfg scripts/example_configs/ggeur_baseline_lstm/ggeur_lstm_mdsent_rating4.yaml
```
