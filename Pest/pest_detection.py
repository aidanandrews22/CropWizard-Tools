import beam
from beam import endpoint, Image

if beam.env.is_remote():
    import os

requirements = ["os"]

volume_path = "./models"

image = Image(
    python_version="python3.12",
    python_packages=requirements,
    commands=["apt-get update"]
)

def load_model():
    print("tmp")

ourSecrets = [
    "tmp"
]

@endpoint(name="pest_detection",
               workers=1,
               cpu=1,
               memory="3Gi",
               max_pending_tasks=10,
               timeout=60,
               keep_warm_seconds=60 * 3,
               secrets=ourSecrets,
               on_start=load_model,
               image=image,
               volumes=[beam.Volume(name="my_models", mount_path=volume_path)])

def predict():
    print("tmp")
    def detect_pests():
        print("tmp")

