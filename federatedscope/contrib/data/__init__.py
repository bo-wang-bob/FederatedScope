import importlib
import glob
import logging
import os
from os.path import dirname, basename, isfile, join

if os.environ.get('FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT') == '1':
    modules = [
        join(dirname(__file__), 'ggeur_data.py'),
        join(dirname(__file__), 'mdsent_data.py'),
    ]
else:
    modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

logger = logging.getLogger(__name__)

for _name in __all__:
    try:
        module = importlib.import_module(f"{__name__}.{_name}")
        globals()[_name] = module
    except Exception as error:
        logger.warning(
            'Skip importing federatedscope.contrib.data.%s due to %s', _name,
            error)
