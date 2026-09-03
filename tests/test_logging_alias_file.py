import logging

from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.configs.config import global_cfg


def test_update_logger_writes_operator_log_alias(tmp_path):
    cfg = global_cfg.clone()
    cfg.defrost()
    cfg.outdir = str(tmp_path / "exp")
    cfg.expname = "alias-test"
    cfg.log_file = str(tmp_path / "operator.log")
    cfg.verbose = 1

    update_logger(cfg, clear_before_add=True)
    logging.getLogger("federatedscope.alias_test").info(
        "operator alias marker")
    for handler in logging.getLogger("federatedscope").handlers:
        handler.flush()

    operator_log = tmp_path / "operator.log"
    assert operator_log.is_file()
    assert "operator alias marker" in operator_log.read_text(
        encoding="utf-8")
    assert (tmp_path / "exp" / "alias-test" / "exp_print.log").is_file()
