import glob
import json
import sys

import torch


path = glob.glob(sys.argv[1])[0]
payload = torch.load(path, map_location="cpu")
print(path)
print(json.dumps(payload.get("metadata", {}), sort_keys=True, default=str))
