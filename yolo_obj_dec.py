from ultralytics import YOLO
import os
from PIL import Image

# Load the YOLO model
model = YOLO("yolo11x.pt")

# Path to your folder of images
image_folder = "bench/google_street_view_hd"

# List all image files in the folder
image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Run detection on all images
#results = model.predict(source=image_files, imgsz=1280, conf=0.3, classes=[9], save=True, save_txt=True)

# Process each image dynamically
for image_path in image_files:
    # Get image size dynamically
    with Image.open(image_path) as img:
        width, height = img.size
        max_dim = max(width, height)  # Choose the larger dimension for resizing, if needed

    # Run prediction on the image with its size dynamically set
    results = model.predict(source=image_path, imgsz=max_dim, conf=0.3, classes=[9], save=True, save_txt=True)

