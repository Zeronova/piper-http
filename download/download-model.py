# Download a .onnx model and its .json config from HuggingFace.
# Saves files with the original filename from the download URL.
#
# Usage: python download-model.py <model_url> <json_url> [target_folder]

import subprocess
import sys
import os
from urllib.parse import urlparse


if len(sys.argv) < 3:
    print("Usage: python download-model.py <model_url> <json_url> [target_folder]")
    sys.exit(1)

link_model = sys.argv[1]
link_json = sys.argv[2]
target_folder = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))

os.makedirs(target_folder, exist_ok=True)

# Extract original filename from URL
model_basename = os.path.basename(urlparse(link_model).path)        # e.g. "de_DE-thorsten-high.onnx"
json_basename = os.path.basename(urlparse(link_json).path)          # e.g. "de_DE-thorsten-high.onnx.json"

filename_model = os.path.join(target_folder, model_basename)
filename_json = os.path.join(target_folder, json_basename)

folder = os.path.dirname(os.path.abspath(__file__))

subprocess.run(["python", f"{folder}/getfile.py", link_model, filename_model], check=True)
subprocess.run(["python", f"{folder}/getfile.py", link_json, filename_json], check=True)
