import os
import random
from locust import HttpUser, task, between

image_dir = "data/test"


def pick_image():
    classes = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
    folder = os.path.join(image_dir, random.choice(classes))
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return os.path.join(folder, random.choice(files))


class ModelUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        path = pick_image()
        with open(path, "rb") as img:
            self.client.post("/predict", files={"file": img})

    @task(1)
    def health(self):
        self.client.get("/health")
