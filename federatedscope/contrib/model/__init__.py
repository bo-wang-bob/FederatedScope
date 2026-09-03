import importlib
import logging
import os
from os.path import dirname, basename, isfile, join
import glob

logger = logging.getLogger(__name__)

if os.environ.get('FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT') == '1':
    modules = [
        join(dirname(__file__), 'ggeur_mlp.py'),
        join(dirname(__file__), 'ggeur_text_rnn.py'),
    ]
else:
    modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

for _name in __all__:
    try:
        module = importlib.import_module(f"{__name__}.{_name}")
        globals()[_name] = module
    except Exception as error:
        logger.warning(
            'Skip importing federatedscope.contrib.model.%s due to %s', _name,
            error)
