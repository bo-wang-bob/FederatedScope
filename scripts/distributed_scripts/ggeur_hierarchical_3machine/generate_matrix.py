#!/usr/bin/env python3
"""Generate three-machine configs from the final five-model experiment set."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = (REPO_ROOT / "scripts" / "example_configs" /
               "ggeur_final_5models")
DEFAULT_OUTPUT = (REPO_ROOT / "scripts" / "distributed_scripts" /
                  "ggeur_hierarchical_3machine" / "runs")
SUBSERVER_ID_BASE = 100000
MDSENT_REFERENCE_BRANCH = "origin/feature/ggeur-backdoor-research"
MDSENT_REFERENCE_COMMITS = ["79ae3d7", "6654a33"]


def path_str(path):
    return str(path).replace("\\", "/")


def load_yaml(path):
    with open(path, "r", encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def write_yaml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(content, stream, sort_keys=False, allow_unicode=True)


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def source_method_name(method):
    """Map the public method alias to the retained source config name."""
    return ("ggeur" if str(method).lower() in {"ours", "platform"}
            else str(method).lower())


def iter_sources(groups, methods, case_specs=None):
    if case_specs:
        for group, method in case_specs:
            source_method = source_method_name(method)
            source = SOURCE_ROOT / group / f"{source_method}.yaml"
            if not source.is_file():
                raise FileNotFoundError(f"Missing source config: {source}")
            yield group, str(method).lower(), source
        return
    if groups:
        group_dirs = [SOURCE_ROOT / group for group in groups]
    else:
        group_dirs = sorted(
            path for path in SOURCE_ROOT.iterdir() if path.is_dir())

    for group_dir in group_dirs:
        if not group_dir.is_dir():
            continue
        if methods:
            for method in methods:
                source = group_dir / f"{source_method_name(method)}.yaml"
                if source.is_file():
                    yield group_dir.name, str(method).lower(), source
        else:
            for source in sorted(group_dir.glob("*.yaml")):
                yield group_dir.name, source.stem, source


def dataset_family(group):
    if group.startswith("digit3_"):
        return "digit3"
    if group.startswith("officehome_"):
        return "officehome"
    if group.startswith("domainnet_"):
        return "domainnet"
    if group.startswith("mdsent_"):
        return "mdsent"
    raise ValueError(f"Unsupported experiment group: {group}")


def client_profile(args, client_id):
    if args.dual_simulated_topology:
        # Keep high-volume model fan-out local.  MDSent's 4 MiB LSTM state is
        # consumed by 120 clients on the 128 GiB third host.  DomainNet stays
        # on the 8G host where the validated image feature caches already live.
        family = getattr(args, "current_family", "")
        if family == "mdsent":
            return {
                "name": "third",
                "repo": args.third_client_repo,
                "feature_cache_root": args.third_client_feature_cache_root,
                "officehome_root": args.third_client_officehome_root,
                "officehome_manifest_root": args.third_client_officehome_manifest_root,
                "domainnet_root": args.third_client_domainnet_root,
                "domainnet_manifest_path": args.third_client_domainnet_manifest_path,
                "digit3_root": args.third_client_digit3_root,
                "mdsent_root": args.third_client_mdsent_root,
                "clip_model_path": args.third_client_clip_model_path,
                "mixer_checkpoint_path": args.third_client_mixer_checkpoint_path,
                "bert_model_path": args.third_client_bert_model_path,
                "bind_host": args.third_client_bind_host,
                "advertise_host": args.third_client_advertise_host,
                "port": args.third_client_port_base + client_id - 1,
            }
        # DomainNet's augmented CNN features are large enough that launching
        # all 60 client processes on the 8G Windows host can exhaust memory.
        # Honour --windows-client-count in the dual-host topology so a formal
        # 60-client run can place clients 1..N on 8G and the remainder on the
        # third host.  Other experiment families retain their validated
        # placement unless explicitly handled above.
        if (family == "domainnet" and
                client_id > args.windows_client_count):
            return {
                "name": "third",
                "repo": args.third_client_repo,
                "feature_cache_root": args.third_client_feature_cache_root,
                "officehome_root": args.third_client_officehome_root,
                "officehome_manifest_root": args.third_client_officehome_manifest_root,
                "domainnet_root": args.third_client_domainnet_root,
                "domainnet_manifest_path": args.third_client_domainnet_manifest_path,
                "digit3_root": args.third_client_digit3_root,
                "mdsent_root": args.third_client_mdsent_root,
                "clip_model_path": args.third_client_clip_model_path,
                "mixer_checkpoint_path": args.third_client_mixer_checkpoint_path,
                "bert_model_path": args.third_client_bert_model_path,
                "bind_host": args.third_client_bind_host,
                "advertise_host": args.third_client_advertise_host,
                "port": args.third_client_port_base + client_id - 1,
            }
        return {
            "name": "8g",
            "repo": args.client_repo,
            "feature_cache_root": args.client_feature_cache_root,
            "officehome_root": args.client_officehome_root,
            "officehome_manifest_root": args.client_officehome_manifest_root,
            "domainnet_root": args.client_domainnet_root,
            "domainnet_manifest_path": args.client_domainnet_manifest_path,
            "digit3_root": args.client_digit3_root,
            "mdsent_root": args.client_mdsent_root,
            "clip_model_path": args.client_clip_model_path,
            "mixer_checkpoint_path": args.client_mixer_checkpoint_path,
            "bert_model_path": args.client_bert_model_path,
            "bind_host": args.client_bind_host,
            "advertise_host": args.client_advertise_host,
            "port": args.client_port_base + client_id - 1,
        }
    if client_id <= args.windows_client_count:
        return {
            "name": "8g",
            "repo": args.client_repo,
            "feature_cache_root": args.client_feature_cache_root,
            "officehome_root": args.client_officehome_root,
            "officehome_manifest_root": args.client_officehome_manifest_root,
            "domainnet_root": args.client_domainnet_root,
            "domainnet_manifest_path": args.client_domainnet_manifest_path,
            "digit3_root": args.client_digit3_root,
            "mdsent_root": args.client_mdsent_root,
            "clip_model_path": args.client_clip_model_path,
            "mixer_checkpoint_path": args.client_mixer_checkpoint_path,
            "bert_model_path": args.client_bert_model_path,
            "bind_host": args.client_bind_host,
            "advertise_host": args.client_advertise_host,
            "port": args.client_port_base + client_id - 1,
        }
    linux_index = client_id - args.windows_client_count - 1
    return {
        "name": "4090",
        "repo": args.root_client_repo,
        "feature_cache_root": args.root_client_feature_cache_root,
        "officehome_root": args.root_client_officehome_root,
        "officehome_manifest_root": args.root_client_officehome_manifest_root,
        "domainnet_root": args.root_client_domainnet_root,
        "domainnet_manifest_path": args.root_client_domainnet_manifest_path,
        "digit3_root": args.root_client_digit3_root,
        "mdsent_root": args.root_client_mdsent_root,
        "clip_model_path": args.root_client_clip_model_path,
        "mixer_checkpoint_path": args.root_client_mixer_checkpoint_path,
        "bert_model_path": args.root_client_bert_model_path,
        "bind_host": args.root_client_bind_host,
        "advertise_host": args.root_client_advertise_host,
        "port": args.root_client_port_base + linux_index,
    }


def rewrite_client_paths(cfg, args, group, case_name, profile, client_id):
    family = dataset_family(group)
    cfg["data"]["root"] = profile[f"{family}_root"]
    ggeur = cfg.setdefault("ggeur", {})
    if ggeur.get("feature_extractor") == "clip":
        ggeur["clip_model_path"] = profile["clip_model_path"]
    elif ggeur.get("feature_extractor") == "timm":
        ggeur["timm_checkpoint_path"] = profile["mixer_checkpoint_path"]
    elif ggeur.get("feature_extractor") == "bert":
        ggeur["bert_model_path"] = profile["bert_model_path"]
    ggeur["feature_cache_dir"] = (
        f"{profile['feature_cache_root'].rstrip('/')}/{group}")
    ggeur["use_feature_cache"] = True
    ggeur["require_complete_feature_cache"] = True
    if family == "officehome":
        manifest_root = profile["officehome_manifest_root"].rstrip("/")
        ggeur["officehome_manifest_path"] = (
            f"{manifest_root}/client_{client_id:06d}/client_manifest.json")
        ggeur["officehome_manifest_base"] = ""
        ggeur["officehome_manifest_use_config_root"] = True
    elif family == "domainnet":
        ggeur["domainnet_manifest_path"] = profile[
            "domainnet_manifest_path"]
        ggeur["domainnet_domains"] = list(args.domainnet_domains_list)
    elif family == "digit3":
        digit_root = profile["digit3_root"].rstrip("/")
        ggeur["digit3_manifest_path"] = (
            f"{digit_root}/manifests/client_{client_id:06d}/"
            "client_manifest.json")
        ggeur["digit3_manifest_base"] = ""
        ggeur["digit3_manifest_use_config_root"] = True
        ggeur["digit3_global_manifest_path"] = (
            f"{digit_root}/dataset_manifest.json")
        ggeur["digit3_domains"] = ["emnist_digits", "usps", "svhn"]
        # Distributed clients are cache-only and must never receive the Mixer
        # checkpoint used by the other timm experiment group.
        ggeur["timm_checkpoint_path"] = ""
    # Keep the base path compact.  The versioned cache namespace includes
    # dataset/extractor metadata and can otherwise exceed Windows MAX_PATH.
    cache_run_id = (getattr(args, "augmented_feature_cache_run_id", "")
                    or args.run_id)
    ggeur["augmented_feature_cache_dir"] = (
        f"{profile['repo']}/exp/aug/{cache_run_id}/{case_name}")


def configure_common(cfg, args, client_num):
    cfg.setdefault("trainer", {})["type"] = "ggeur"
    federate = cfg.setdefault("federate", {})
    federate["method"] = "ggeur"
    federate["mode"] = "distributed"
    federate["client_num"] = client_num
    configured_sample_num = int(federate.get("sample_client_num", 0) or 0)
    if args.sample_client_num is not None:
        federate["sample_client_num"] = int(args.sample_client_num)
    elif configured_sample_num > client_num:
        federate["sample_client_num"] = client_num
    # Formal three-machine runs use one common 100-round acceptance contract.
    # Some source experiment configs (notably MDSent) use a longer standalone
    # schedule, so do not inherit their round count into the distributed matrix.
    federate["total_round_num"] = args.total_round_num
    federate["make_global_eval"] = False
    federate["online_aggr"] = False
    distribute = cfg.setdefault("distribute", {})
    distribute.update({
        "use": True,
        "join_timeout_seconds": args.join_timeout_seconds,
        "grpc_max_send_message_length": args.grpc_max_message_length,
        "grpc_max_receive_message_length": args.grpc_max_message_length,
        "grpc_enable_http_proxy": False,
        "grpc_compression": args.grpc_compression,
    })
    ggeur = cfg.setdefault("ggeur", {})
    ggeur["distributed_stage_timeout"] = args.stage_timeout_seconds
    ggeur["statistics_upload_stagger_seconds"] = \
        args.statistics_upload_stagger_seconds


def apply_experiment_overrides(cfg, args, method):
    """Apply explicit matrix-wide tuning without mutating source configs."""
    text_hidden_dim = getattr(args, "text_hidden_dim", None)
    model_type = str(cfg.get("model", {}).get("type", "")).lower()
    if (text_hidden_dim is not None and
            model_type in {"ggeur_rnn", "ggeur_lstm"}):
        # Keep the same recurrent architecture for every comparison method,
        # but permit a smaller hidden state for communication-bound,
        # all-client distributed validation.
        cfg.setdefault("model", {})["hidden"] = int(text_hidden_dim)
    if args.train_learning_rate is not None:
        optimizer = cfg.setdefault("train", {}).setdefault("optimizer", {})
        optimizer["lr"] = float(args.train_learning_rate)
    train_local_update_steps = getattr(
        args, "train_local_update_steps", None)
    if train_local_update_steps is not None:
        cfg.setdefault("train", {})["local_update_steps"] = int(
            train_local_update_steps)
    eval_frequency = getattr(args, "eval_frequency", None)
    if eval_frequency is not None:
        cfg.setdefault("eval", {})["freq"] = int(eval_frequency)

    # FedOpt's server optimizer acts on the pseudo-gradient w_t - w_bar and
    # therefore has its own learning-rate scale.  Never copy the client LR to
    # it implicitly: doing so turned the validated Adam server LR (1e-2) into
    # 1e-5 in the tuned matrix and left the global model near initialization.
    fedopt_server_lr = getattr(args, "fedopt_server_learning_rate", None)
    if (fedopt_server_lr is not None and
            cfg.get("fedopt", {}).get("use", False)):
        server_optimizer = cfg["fedopt"].setdefault("optimizer", {})
        server_optimizer["lr"] = float(fedopt_server_lr)

    ggeur = cfg.setdefault("ggeur", {})
    reuse_augmented_cache = getattr(
        args, "reuse_augmented_feature_cache", None)
    save_augmented_cache = getattr(
        args, "save_augmented_feature_cache", None)
    if reuse_augmented_cache is not None:
        ggeur["reuse_augmented_feature_cache"] = bool(
            reuse_augmented_cache)
    if save_augmented_cache is not None:
        ggeur["save_augmented_feature_cache"] = bool(
            save_augmented_cache)
    headonly_eval_mode = getattr(args, "ggeur_headonly_eval_mode", None)
    if headonly_eval_mode is not None:
        ggeur["headonly_eval_mode"] = str(headonly_eval_mode)
    terminal_client_eval_only = getattr(
        args, "ggeur_terminal_client_eval_only", None)
    if terminal_client_eval_only is not None:
        ggeur["terminal_client_eval_only"] = bool(
            terminal_client_eval_only)

    # Baselines must keep augmentation disabled.  Generation overrides apply
    # only to the GGEUR case even though every distributed config uses the
    # common GGEUR worker implementation.
    if str(method).lower() != "ggeur":
        ggeur["num_generated_per_sample"] = 0
        ggeur["num_generated_per_prototype"] = 0
        ggeur["target_size_per_class"] = 0
        return
    classifier_overrides = (
        ("mlp_hidden_dim", getattr(args, "ggeur_mlp_hidden_dim", None),
         int),
        ("mlp_dropout", getattr(args, "ggeur_mlp_dropout", None), float),
        ("max_cross_client_prototypes_per_class", getattr(
            args, "ggeur_max_cross_client_prototypes_per_class", None), int),
        ("diagonal_covariance", getattr(
            args, "ggeur_diagonal_covariance", None), bool),
        ("prototype_classifier_init", getattr(
            args, "ggeur_prototype_classifier_init", None), bool),
        ("lda_classifier_init", getattr(
            args, "ggeur_lda_classifier_init", None), bool),
        ("domain_prototype_ensemble_per_class", getattr(
            args, "ggeur_domain_prototype_ensemble_per_class", None), int),
        ("domain_personalized_head_epochs", getattr(
            args, "ggeur_domain_personalized_head_epochs", None), int),
        ("domain_personalized_head_lr", getattr(
            args, "ggeur_domain_personalized_head_lr", None), float),
        ("domain_personalized_head_use_full_train_cache", getattr(
            args, "ggeur_domain_personalized_head_use_full_train_cache",
            None), bool),
    )
    for key, value, converter in classifier_overrides:
        if value is not None:
            ggeur[key] = converter(value)
    overrides = (
        ("num_generated_per_sample", args.ggeur_num_generated_per_sample),
        ("num_generated_per_prototype",
         args.ggeur_num_generated_per_prototype),
        ("target_size_per_class", args.ggeur_target_size_per_class),
    )
    for key, value in overrides:
        if value is not None:
            ggeur[key] = int(value)


def split_clients(client_num, subserver_num):
    assignments = []
    base = client_num // subserver_num
    remainder = client_num % subserver_num
    next_client = 1
    for idx in range(1, subserver_num + 1):
        count = base + (1 if idx <= remainder else 0)
        client_ids = list(range(next_client, next_client + count))
        assignments.append({
            "subserver_id": idx,
            "client_ids": client_ids,
        })
        next_client += count
    return assignments


def choose_subserver_num(args, client_num):
    if args.subserver_num > 0:
        return min(args.subserver_num, client_num)
    return max(1, min(args.max_auto_subservers,
                      int(math.ceil(client_num /
                                    float(args.clients_per_subserver)))))


def generate_case(args, run_root, group, method, source_path):
    source = load_yaml(source_path)
    source_method = source_path.stem.lower()
    client_num = (int(args.client_num) if args.client_num is not None else
                  int(source["federate"]["client_num"]))
    subserver_num = choose_subserver_num(args, client_num)
    assignments = split_clients(client_num, subserver_num)
    family = dataset_family(group)
    # Keep the retained source filename for compatibility while exposing the
    # public method name requested by the operator in run directories.
    display_method = ("platform" if method.lower() in
                      {"ggeur", "ours", "platform"} else method.lower())
    case_name = f"{group}_{display_method}"
    case_dir = run_root / case_name
    cfg_dir = case_dir / "configs"

    root_cfg = copy.deepcopy(source)
    configure_common(root_cfg, args, client_num)
    apply_experiment_overrides(root_cfg, args, source_method)
    args.current_family = family
    if args.dual_simulated_topology:
        root_cfg["data"]["root"] = getattr(args, f"client_{family}_root")
    else:
        root_cfg["data"]["root"] = getattr(
            args, f"root_client_{family}_root")
    root_ggeur = root_cfg.setdefault("ggeur", {})
    if family == "domainnet":
        root_ggeur["statistics_upload_stagger_seconds"] = \
            args.domainnet_statistics_upload_stagger_seconds
    root_ggeur["use_feature_cache"] = True
    root_ggeur["require_complete_feature_cache"] = True
    if family == "officehome":
        root_ggeur["officehome_manifest_path"] = ""
        root_ggeur["officehome_manifest_base"] = \
            args.root_client_officehome_manifest_root
        root_ggeur["officehome_manifest_use_config_root"] = True
    elif family == "domainnet":
        root_ggeur["domainnet_manifest_path"] = (
            args.client_domainnet_manifest_path
            if args.dual_simulated_topology
            else args.root_client_domainnet_manifest_path)
        root_ggeur["domainnet_domains"] = list(args.domainnet_domains_list)
    elif family == "digit3":
        digit_root = root_cfg["data"]["root"].rstrip("/")
        root_ggeur["digit3_manifest_path"] = ""
        root_ggeur["digit3_manifest_base"] = f"{digit_root}/manifests"
        root_ggeur["digit3_manifest_use_config_root"] = True
        root_ggeur["digit3_global_manifest_path"] = (
            f"{digit_root}/dataset_manifest.json")
        root_ggeur["digit3_domains"] = ["emnist_digits", "usps", "svhn"]
        # The formal acceptance run records both equal-domain server accuracy
        # and every terminal client's local-model accuracy at round 99.
        if args.ggeur_headonly_eval_mode is None:
            root_ggeur["headonly_eval_mode"] = "both"
        if args.eval_frequency is None:
            root_cfg.setdefault("eval", {})["freq"] = 99
    if root_ggeur.get("feature_extractor") == "clip":
        root_ggeur["clip_model_path"] = args.root_client_clip_model_path
    elif root_ggeur.get("feature_extractor") == "timm":
        root_ggeur["timm_checkpoint_path"] = (
            "" if family == "digit3" else
            args.root_client_mixer_checkpoint_path)
    elif root_ggeur.get("feature_extractor") == "bert":
        root_ggeur["bert_model_path"] = args.root_client_bert_model_path
    root_cfg["use_gpu"] = args.root_use_gpu
    root_cfg["device"] = args.root_device
    root_cfg["distribute"].update({
        "role": "server",
        "server_host": args.root_bind_host,
        "server_port": args.root_port,
    })
    root_cfg["ggeur"].update({
        "hierarchical_training": True,
        "hierarchical_subserver_num": subserver_num,
        "hierarchical_subserver_id_base": SUBSERVER_ID_BASE,
        "feature_cache_dir": (
            f"{args.root_client_feature_cache_root.rstrip('/')}/{group}"),
    })
    root_cfg["outdir"] = f"{args.root_repo}/exp/hierarchical/{args.run_id}/{case_name}"
    root_cfg["expname"] = f"{case_name}_root"
    root_config_path = cfg_dir / "root_server.yaml"
    write_yaml(root_config_path, root_cfg)

    subserver_configs = []
    subserver_hosts = {"8g": [], "third": []}
    client_to_subserver = {}
    for assignment in assignments:
        subserver_id = assignment["subserver_id"]
        subserver_port = args.subserver_port_base + subserver_id - 1
        if args.dual_simulated_topology:
            subserver_profile = "third" if family == "mdsent" else (
                "8g" if subserver_id % 2 == 1 else "third")
            subserver_advertise_host = (
                args.client_advertise_host if subserver_profile == "8g"
                else args.third_client_advertise_host)
        else:
            subserver_profile = "third"
            subserver_advertise_host = args.subserver_advertise_host
        for client_id in assignment["client_ids"]:
            client_to_subserver[client_id] = (
                subserver_id, subserver_advertise_host, subserver_port)
        sub_cfg = {
            "run_id": args.run_id,
            "case": case_name,
            "subserver_id": subserver_id,
            "sender_id": SUBSERVER_ID_BASE + subserver_id,
            "listen_host": args.subserver_bind_host,
            "listen_port": subserver_port,
            "advertise_host": subserver_advertise_host,
            "root_host": args.root_host,
            "root_port": args.root_port,
            "client_ids": assignment["client_ids"],
            "round_timeout_sec": args.subserver_round_timeout_seconds,
            "grpc_max_send_message_length": args.grpc_max_message_length,
            "grpc_max_receive_message_length": args.grpc_max_message_length,
            "grpc_compression": args.grpc_compression,
        }
        sub_dir = (f"subservers_{subserver_profile}"
                   if args.dual_simulated_topology else "")
        sub_path = cfg_dir / sub_dir / f"subserver_{subserver_id}.json"
        write_json(sub_path, sub_cfg)
        subserver_configs.append(path_str(sub_path))
        subserver_hosts[subserver_profile].append({
            "subserver_id": subserver_id,
            "host": subserver_advertise_host,
            "port": subserver_port,
            "config": path_str(sub_path),
        })

    client_configs = []
    client_hosts = {"8g": [], "third": []} if \
        args.dual_simulated_topology else {"8g": [], "4090": []}
    for client_id in range(1, client_num + 1):
        profile = client_profile(args, client_id)
        client_cfg = copy.deepcopy(source)
        configure_common(client_cfg, args, client_num)
        apply_experiment_overrides(client_cfg, args, source_method)
        if family == "domainnet":
            client_cfg.setdefault("ggeur", {})[
                "statistics_upload_stagger_seconds"] = \
                args.domainnet_statistics_upload_stagger_seconds
        rewrite_client_paths(client_cfg, args, group, case_name, profile,
                             client_id)
        client_cfg["use_gpu"] = args.client_use_gpu
        client_cfg["device"] = args.client_device
        subserver_id, subserver_host, subserver_port = \
            client_to_subserver[client_id]
        client_port = profile["port"]
        client_cfg["distribute"].update({
            "role": "client",
            "client_id": client_id,
            "server_host": subserver_host,
            "server_port": subserver_port,
            "client_host": profile["bind_host"],
            "client_port": client_port,
            "client_advertise_host": profile["advertise_host"],
            "client_advertise_port": client_port,
            "data_idx": client_id,
        })
        client_cfg["ggeur"]["hierarchical_training"] = False
        client_cfg["outdir"] = (
            f"{profile['repo']}/exp/hierarchical/{args.run_id}/"
            f"{case_name}/client_{client_id:06d}")
        client_cfg["expname"] = f"{case_name}_client_{client_id:06d}"
        client_path = (cfg_dir / f"clients_{profile['name']}" /
                       f"client_{client_id:06d}.yaml")
        write_yaml(client_path, client_cfg)
        client_configs.append(path_str(client_path))
        client_hosts[profile["name"]].append({
            "client_id": client_id,
            "host": profile["advertise_host"],
            "port": client_port,
            "config": path_str(client_path),
        })

    manifest = {
        "run_id": args.run_id,
        "case": case_name,
        "group": group,
        "method": display_method,
        "source_config": path_str(source_path.relative_to(REPO_ROOT)),
        "algorithm_reference": {
            "branch": MDSENT_REFERENCE_BRANCH,
            "commits": MDSENT_REFERENCE_COMMITS,
            "training_scope": (
                "frozen_bert_cls_plus_trainable_rnn_lstm_head"
                if dataset_family(group) == "mdsent" else
                "source_config_defined"),
        },
        "client_num": client_num,
        "subserver_num": subserver_num,
        "root_host": args.root_host,
        "root_port": args.root_port,
        "subserver_host": args.subserver_advertise_host,
        "subserver_ports": [args.subserver_port_base + idx
                            for idx in range(subserver_num)],
        "client_hosts": client_hosts,
        "windows_client_count": min(args.windows_client_count, client_num),
        "root_client_count": max(0, client_num - args.windows_client_count),
        "assignments": assignments,
        "root_config": path_str(root_config_path),
        "subserver_configs": subserver_configs,
        "subserver_hosts": subserver_hosts,
        "client_configs": client_configs,
        "physical_topology": (
            "dual_host_three_logical_layers"
            if args.dual_simulated_topology else "three_host"),
        "case_dir": path_str(case_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "parameter_overrides": {
            "train_learning_rate": args.train_learning_rate,
            "train_local_update_steps": args.train_local_update_steps,
            "eval_frequency": args.eval_frequency,
            "fedopt_server_learning_rate": (
                args.fedopt_server_learning_rate
                if source_method == "fedopt" else None),
            "reuse_augmented_feature_cache": (
                args.reuse_augmented_feature_cache),
            "save_augmented_feature_cache": (
                args.save_augmented_feature_cache),
            "ggeur_headonly_eval_mode": args.ggeur_headonly_eval_mode,
            "ggeur_terminal_client_eval_only": (
                args.ggeur_terminal_client_eval_only),
            "ggeur_num_generated_per_sample": (
                args.ggeur_num_generated_per_sample
                if source_method == "ggeur" else None),
            "ggeur_num_generated_per_prototype": (
                args.ggeur_num_generated_per_prototype
                if source_method == "ggeur" else None),
            "ggeur_target_size_per_class": (
                args.ggeur_target_size_per_class
                if source_method == "ggeur" else None),
            "ggeur_mlp_hidden_dim": (
                args.ggeur_mlp_hidden_dim
                if source_method == "ggeur" else None),
            "ggeur_mlp_dropout": (
                args.ggeur_mlp_dropout
                if source_method == "ggeur" else None),
            "ggeur_max_cross_client_prototypes_per_class": (
                args.ggeur_max_cross_client_prototypes_per_class
                if source_method == "ggeur" else None),
            "ggeur_diagonal_covariance": (
                args.ggeur_diagonal_covariance
                if source_method == "ggeur" else None),
            "ggeur_prototype_classifier_init": (
                args.ggeur_prototype_classifier_init
                if source_method == "ggeur" else None),
            "ggeur_lda_classifier_init": (
                args.ggeur_lda_classifier_init
                if source_method == "ggeur" else None),
            "ggeur_domain_prototype_ensemble_per_class": (
                args.ggeur_domain_prototype_ensemble_per_class
                if source_method == "ggeur" else None),
            "ggeur_domain_personalized_head_epochs": (
                args.ggeur_domain_personalized_head_epochs
                if source_method == "ggeur" else None),
            "ggeur_domain_personalized_head_lr": (
                args.ggeur_domain_personalized_head_lr
                if source_method == "ggeur" else None),
            "ggeur_domain_personalized_head_use_full_train_cache": (
                args.ggeur_domain_personalized_head_use_full_train_cache
                if source_method == "ggeur" else None),
        },
    }
    write_json(case_dir / "manifest.json", manifest)
    return manifest


def parse_csv(value):
    # Preserve the caller's order so the generated manifest can also serve as
    # the authoritative execution order for scoped formal runs.
    values = []
    for item in str(value or "").split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    return values


def parse_case_specs(value):
    specs = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid --case-specs item {item!r}; expected group:method")
        group, method = (part.strip() for part in item.split(":", 1))
        spec = (group, method)
        if not all(spec):
            raise ValueError(f"Invalid --case-specs item {item!r}")
        if spec not in specs:
            specs.append(spec)
    return specs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--append-manifest", action="store_true",
        help=("Append or replace generated cases in an existing run manifest "
              "so dataset families with different cache roots can share one "
              "sequential queue"))
    parser.add_argument("--groups", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument(
        "--client-num", type=int, default=None,
        help="Override the total client count for functional scale tests")
    parser.add_argument(
        "--sample-client-num", type=int, default=None,
        help=("Override the clients selected per round; zero selects every "
              "configured client"))
    parser.add_argument(
        "--case-specs", default="",
        help="Ordered exact cases as group:method,group:method")
    parser.add_argument(
        "--dual-simulated-topology",
        action=argparse.BooleanOptionalAction, default=False,
        help="Use 8G+third while preserving root/subserver/client layers")
    parser.add_argument("--root-host", default="10.112.81.135")
    parser.add_argument("--root-bind-host", default="0.0.0.0")
    parser.add_argument("--root-port", type=int, default=60050)
    parser.add_argument(
        "--total-round-num", type=int, default=100,
        help="Training rounds used by every generated formal distributed case")
    parser.add_argument(
        "--train-learning-rate", type=float, default=None,
        help="Override only the client optimizer LR in every generated case")
    parser.add_argument(
        "--train-local-update-steps", type=int, default=None,
        help="Override the local classifier epochs in every generated case")
    parser.add_argument(
        "--eval-frequency", type=int, default=None,
        help=("Evaluate every N aggregation rounds; omitted preserves each "
              "source config"))
    parser.add_argument(
        "--fedopt-server-learning-rate", type=float, default=None,
        help=("Override only the FedOpt server optimizer LR; if omitted, "
              "retain the source experiment's independently tuned value"))
    parser.add_argument(
        "--text-hidden-dim", type=int, default=None,
        help=("Override the hidden state width of RNN and LSTM models for "
              "every comparison method"))
    parser.add_argument(
        "--reuse-augmented-feature-cache",
        action=argparse.BooleanOptionalAction, default=None,
        help="Allow GGEUR to load previously generated feature caches")
    parser.add_argument(
        "--save-augmented-feature-cache",
        action=argparse.BooleanOptionalAction, default=None,
        help="Allow GGEUR to persist generated feature caches")
    parser.add_argument(
        "--augmented-feature-cache-run-id", default="",
        help=("Read/write the augmented cache namespace from another run; "
              "empty keeps the current run id and preserves prior behavior"))
    parser.add_argument(
        "--ggeur-headonly-eval-mode",
        choices=["server", "client", "both"], default=None,
        help=("Select root test-set evaluation, per-client test-set "
              "evaluation, or both; omitted preserves the source config"))
    parser.add_argument(
        "--ggeur-terminal-client-eval-only",
        action=argparse.BooleanOptionalAction, default=None,
        help=("With headonly_eval_mode=both, evaluate the root every "
              "configured round and all terminal clients only on the final "
              "round"))
    parser.add_argument(
        "--ggeur-num-generated-per-sample", type=int, default=None,
        help="Override sample-based generation only for GGEUR cases")
    parser.add_argument(
        "--ggeur-num-generated-per-prototype", type=int, default=None,
        help="Override prototype-based generation only for GGEUR cases")
    parser.add_argument(
        "--ggeur-target-size-per-class", type=int, default=None,
        help="Override final per-client class size only for GGEUR cases")
    parser.add_argument(
        "--ggeur-mlp-hidden-dim", type=int, default=None,
        help="Override classifier hidden width only for GGEUR cases")
    parser.add_argument(
        "--ggeur-mlp-dropout", type=float, default=None,
        help="Override classifier dropout only for GGEUR cases")
    parser.add_argument(
        "--ggeur-max-cross-client-prototypes-per-class", type=int,
        default=None,
        help=("Limit cross-client prototypes retained per class only for "
              "GGEUR cases; zero keeps every prototype"))
    parser.add_argument(
        "--ggeur-diagonal-covariance",
        action=argparse.BooleanOptionalAction, default=None,
        help=("Transmit per-dimension variances instead of full covariance "
              "matrices only for GGEUR cases"))
    parser.add_argument(
        "--ggeur-prototype-classifier-init",
        action=argparse.BooleanOptionalAction, default=None,
        help=("Initialize a linear classifier from round-0 global class "
              "prototypes only for GGEUR cases"))
    parser.add_argument(
        "--ggeur-lda-classifier-init",
        action=argparse.BooleanOptionalAction, default=None,
        help=("Initialize a linear classifier from pooled diagonal LDA "
              "statistics only for GGEUR cases"))
    parser.add_argument(
        "--ggeur-domain-prototype-ensemble-per-class", type=int,
        default=None,
        help=("Use up to this many round-0 class prototypes in a "
              "maximum-cosine inference head only for GGEUR cases"))
    parser.add_argument(
        "--ggeur-domain-personalized-head-epochs", type=int, default=None)
    parser.add_argument(
        "--ggeur-domain-personalized-head-lr", type=float, default=None)
    parser.add_argument(
        "--ggeur-domain-personalized-head-use-full-train-cache",
        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--root-use-gpu", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--root-device", type=int, default=1)
    parser.add_argument("--subserver-advertise-host",
                        default="10.129.248.111")
    parser.add_argument("--subserver-bind-host", default="0.0.0.0")
    parser.add_argument("--subserver-port-base", type=int, default=61000)
    parser.add_argument("--client-advertise-host",
                        default="10.129.222.189")
    parser.add_argument("--client-bind-host", default="0.0.0.0")
    parser.add_argument("--client-port-base", type=int, default=20000)
    parser.add_argument(
        "--windows-client-count", type=int, default=60,
        help="Number of lowest client IDs launched on the 8G Windows host")
    parser.add_argument("--root-client-advertise-host",
                        default="10.112.81.135")
    parser.add_argument("--root-client-bind-host", default="0.0.0.0")
    parser.add_argument("--root-client-port-base", type=int, default=20000)
    parser.add_argument("--third-client-advertise-host",
                        default="10.129.248.111")
    parser.add_argument("--third-client-bind-host", default="0.0.0.0")
    parser.add_argument("--third-client-port-base", type=int, default=20000)
    parser.add_argument("--client-use-gpu",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--client-device", type=int, default=0)
    parser.add_argument("--subserver-num", type=int, default=0)
    parser.add_argument("--clients-per-subserver", type=int, default=30)
    parser.add_argument("--max-auto-subservers", type=int, default=8)
    parser.add_argument("--join-timeout-seconds", type=int, default=1800)
    parser.add_argument("--stage-timeout-seconds", type=int, default=14400)
    parser.add_argument(
        "--statistics-upload-stagger-seconds", type=float, default=3.0,
        help=("Deterministic interval between client round-0 statistics "
              "uploads; prevents large covariance payload fan-in bursts"))
    parser.add_argument(
        "--domainnet-statistics-upload-stagger-seconds", type=float,
        default=20.0,
        help=("DomainNet-specific round-0 upload interval. Its covariance "
              "payloads can exceed 200 MiB for 1024-dimensional CNN "
              "features, so serialize cross-host uploads instead of "
              "creating a 30-RPC fan-in burst."))
    parser.add_argument("--subserver-round-timeout-seconds", type=int,
                        default=12000)
    parser.add_argument("--grpc-max-message-length", type=int,
                        default=1024 * 1024 * 1024)
    parser.add_argument("--grpc-compression", default="nocompression",
                        choices=["nocompression", "gzip", "deflate"])
    parser.add_argument("--root-repo",
                        default="/root/autodl-tmp/FederatedScope")
    parser.add_argument("--subserver-repo",
                        default="/root/FederatedScope")
    parser.add_argument("--client-repo",
                        default="D:/Projects/FederatedScope")
    parser.add_argument(
        "--client-feature-cache-root",
        default="D:/Projects/FederatedScope/exp/distributed_feature_cache")
    parser.add_argument("--client-officehome-root",
                        default="D:/Projects/FederatedScope/OfficeHomeDataset_10072016")
    parser.add_argument(
        "--client-officehome-manifest-root",
        default=("D:/Projects/FederatedScope/exp/distributed_manifests/"
                 "officehome_60c_lds01_seed42"))
    parser.add_argument("--client-domainnet-root",
                        default="D:/Projects/FederatedScope/data/DomainNet")
    parser.add_argument(
        "--client-domainnet-manifest-path",
        default=("D:/Projects/FederatedScope/exp/distributed_manifests/"
                 "domainnet_4domains/domainnet_manifest.json"))
    parser.add_argument(
        "--client-digit3-root",
        default="D:/Projects/FederatedScope/data/digit_three_domain")
    parser.add_argument("--client-mdsent-root",
                        default="D:/Projects/FederatedScope/data/sentiment")
    parser.add_argument("--client-clip-model-path",
                        default="D:/Projects/FederatedScope/pretrained_models/ViT-B-16.pt")
    parser.add_argument("--client-mixer-checkpoint-path",
                        default="D:/Projects/FederatedScope/pretrained_models/mixer_b16_224_complete.pth")
    parser.add_argument("--client-bert-model-path",
                        default="D:/Projects/FederatedScope/pretrained_models/nlptown_bert_base_multilingual_uncased_senti")
    parser.add_argument("--root-client-repo",
                        default="/root/autodl-tmp/FederatedScope")
    parser.add_argument(
        "--root-client-feature-cache-root",
        default="/root/autodl-tmp/FederatedScope/exp/distributed_feature_cache")
    parser.add_argument(
        "--root-client-officehome-root",
        default="/root/autodl-tmp/datasets/OfficeHomeDataset_10072016")
    parser.add_argument(
        "--root-client-officehome-manifest-root",
        default=("/root/autodl-tmp/FederatedScope/exp/"
                 "distributed_manifests/officehome_60c_lds01_seed42"))
    parser.add_argument("--root-client-domainnet-root",
                        default="/root/autodl-tmp/datasets/DomainNet")
    parser.add_argument(
        "--root-client-domainnet-manifest-path",
        default=("/root/autodl-tmp/FederatedScope/exp/"
                 "distributed_manifests/domainnet_4domains/"
                 "domainnet_manifest.json"))
    parser.add_argument(
        "--root-client-digit3-root",
        default="/root/autodl-tmp/datasets/digit_three_domain")
    parser.add_argument("--domainnet-domains",
                        default="clipart,painting,real,sketch")
    parser.add_argument("--root-client-mdsent-root",
                        default="/root/autodl-tmp/datasets/sentiment")
    parser.add_argument(
        "--root-client-clip-model-path",
        default="/root/.cache/clip/ViT-B-16.pt")
    parser.add_argument(
        "--root-client-mixer-checkpoint-path",
        default="/root/autodl-tmp/models/mixer_b16_224_complete.pth")
    parser.add_argument(
        "--root-client-bert-model-path",
        default="/root/autodl-tmp/models/nlptown_bert_base_multilingual_uncased_senti")
    parser.add_argument("--third-client-repo",
                        default="C:/Users/pc/FederatedScope")
    parser.add_argument(
        "--third-client-feature-cache-root",
        default="C:/Users/pc/FederatedScope/exp/distributed_feature_cache")
    parser.add_argument("--third-client-officehome-root",
                        default="C:/Users/pc/FederatedScope/OfficeHomeDataset_10072016")
    parser.add_argument(
        "--third-client-officehome-manifest-root",
        default=("C:/Users/pc/FederatedScope/exp/distributed_manifests/"
                 "officehome_60c_lds01_seed42"))
    parser.add_argument("--third-client-domainnet-root",
                        default="C:/Users/pc/FederatedScope/data/DomainNet")
    parser.add_argument(
        "--third-client-domainnet-manifest-path",
        default=("C:/Users/pc/FederatedScope/exp/distributed_manifests/"
                 "domainnet_4domains/domainnet_manifest.json"))
    parser.add_argument(
        "--third-client-digit3-root",
        default="C:/Users/pc/FederatedScope/data/digit_three_domain")
    parser.add_argument("--third-client-mdsent-root",
                        default="C:/Users/pc/FederatedScope/data/sentiment")
    parser.add_argument("--third-client-clip-model-path",
                        default="C:/Users/pc/FederatedScope/pretrained_models/ViT-B-16.pt")
    parser.add_argument("--third-client-mixer-checkpoint-path",
                        default="C:/Users/pc/FederatedScope/pretrained_models/mixer_b16_224_complete.pth")
    parser.add_argument("--third-client-bert-model-path",
                        default=("C:/Users/pc/FederatedScope/pretrained_models/"
                                 "nlptown_bert_base_multilingual_uncased_senti"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.windows_client_count < 0:
        raise SystemExit("--windows-client-count must be non-negative")
    if (args.sample_client_num is not None and
            (args.sample_client_num < 0 or
             (args.client_num is not None and
              args.sample_client_num > args.client_num))):
        raise SystemExit(
            "--sample-client-num must be between zero and --client-num")
    if args.statistics_upload_stagger_seconds < 0:
        raise SystemExit(
            "--statistics-upload-stagger-seconds must be non-negative")
    if args.domainnet_statistics_upload_stagger_seconds < 0:
        raise SystemExit(
            "--domainnet-statistics-upload-stagger-seconds must be "
            "non-negative")
    if (args.train_learning_rate is not None and
            args.train_learning_rate <= 0):
        raise SystemExit("--train-learning-rate must be positive")
    if (args.train_local_update_steps is not None and
            args.train_local_update_steps <= 0):
        raise SystemExit("--train-local-update-steps must be positive")
    if args.eval_frequency is not None and args.eval_frequency <= 0:
        raise SystemExit("--eval-frequency must be positive")
    if (args.fedopt_server_learning_rate is not None and
            args.fedopt_server_learning_rate <= 0):
        raise SystemExit("--fedopt-server-learning-rate must be positive")
    if args.text_hidden_dim is not None and args.text_hidden_dim <= 0:
        raise SystemExit("--text-hidden-dim must be positive")
    for name in (
            "ggeur_num_generated_per_sample",
            "ggeur_num_generated_per_prototype",
            "ggeur_target_size_per_class",
            "ggeur_mlp_hidden_dim",
            "ggeur_max_cross_client_prototypes_per_class",
            "ggeur_domain_prototype_ensemble_per_class"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if (args.ggeur_domain_personalized_head_epochs is not None and
            args.ggeur_domain_personalized_head_epochs < 0):
        raise SystemExit(
            "--ggeur-domain-personalized-head-epochs must be non-negative")
    if (args.ggeur_domain_personalized_head_lr is not None and
            args.ggeur_domain_personalized_head_lr <= 0):
        raise SystemExit(
            "--ggeur-domain-personalized-head-lr must be positive")
    if (args.ggeur_mlp_dropout is not None and
            not 0.0 <= args.ggeur_mlp_dropout < 1.0):
        raise SystemExit("--ggeur-mlp-dropout must be in [0, 1)")
    args.domainnet_domains_list = [
        item.strip() for item in args.domainnet_domains.split(',')
        if item.strip()
    ]
    if not args.domainnet_domains_list:
        raise SystemExit("--domainnet-domains must not be empty")
    groups = parse_csv(args.groups)
    methods = parse_csv(args.methods)
    case_specs = parse_case_specs(args.case_specs)
    if args.dual_simulated_topology:
        args.root_host = args.client_advertise_host
        args.root_repo = args.client_repo
        args.root_use_gpu = False
        args.root_device = 0
        args.root_client_feature_cache_root = args.client_feature_cache_root
        args.root_client_clip_model_path = args.client_clip_model_path
        args.root_client_mixer_checkpoint_path = \
            args.client_mixer_checkpoint_path
        args.root_client_bert_model_path = args.client_bert_model_path
    base = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT
    run_root = base / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    cases = [generate_case(args, run_root, group, method, source)
             for group, method, source in iter_sources(
                 groups, methods, case_specs)]
    if not cases:
        raise SystemExit("No matching experiment cases")
    combined_cases = []
    matrix_path = run_root / "matrix_manifest.json"
    if args.append_manifest and matrix_path.is_file():
        existing = json.loads(matrix_path.read_text(encoding="utf-8-sig"))
        combined_cases.extend(existing.get("cases", []))
    replacement_names = {item["case"] for item in cases}
    combined_cases = [
        item for item in combined_cases
        if item.get("case") not in replacement_names
    ]
    combined_cases.extend(cases)
    matrix = {
        "run_id": args.run_id,
        "case_count": len(combined_cases),
        "output_root": path_str(run_root),
        "cases": combined_cases,
    }
    write_json(matrix_path, matrix)
    print(json.dumps({
        "run_id": args.run_id,
        "generated_case_count": len(cases),
        "case_count": len(combined_cases),
        "output_root": path_str(run_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
