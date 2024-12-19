from ultralytics import YOLO
import os
from PIL import Image

model = YOLO("yolo11x.pt")

image_folder = "bench/google_street_view_hd"
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

for image_path in image_files:
    with Image.open(image_path) as img:
        width, height = img.size
        max_dim = max(width, height)  # Choose the larger dimension for resizing, if needed

    results = model.predict(source=image_path, imgsz=max_dim, conf=0.3, classes=[9], save=True, save_txt=True)

