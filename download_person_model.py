"""
Download OpenVINO person detection model
"""
from pathlib import Path
import notebook_utils as utils

# Create models directory
models_dir = Path("models/person_detection")
models_dir.mkdir(parents=True, exist_ok=True)

# Model from OpenVINO Model Zoo
# person-detection-retail-0013 - optimized for retail/surveillance
model_name = "person-detection-retail-0013"
precision = "FP16"

base_model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"

model_xml = f"{model_name}.xml"
model_bin = f"{model_name}.bin"

print("Downloading OpenVINO person detection model...")
print(f"Model: {model_name}")
print(f"Precision: {precision}")
print()

# Download model files
utils.download_file(base_model_url + model_xml, filename=model_xml, directory=models_dir)
utils.download_file(base_model_url + model_bin, filename=model_bin, directory=models_dir)

print()
print(f"✓ Model downloaded to: {models_dir}")
print(f"  - {model_xml}")
print(f"  - {model_bin}")
