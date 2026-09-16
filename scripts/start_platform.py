"""Start the portable platform from any working directory."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from federatedscope.standalone_api.platform_app import main

if __name__ == '__main__':
    main()
