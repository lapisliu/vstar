import os
import json
from PIL import Image
from ultralytics import YOLO
import openai
import base64

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

model = YOLO("yolo11x.pt")

benchmark_folder = "difficult"

image_folder = "bench/" + benchmark_folder
label_folder = image_folder
output_file = benchmark_folder+".json"

results = []

def get_gpt_response(question, options, cropped_image_pathes):
    base64_images = []
    for cropped_image_path in cropped_image_pathes:
        img_file = open(cropped_image_path, "rb")
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        base64_images.append(base64_image)

    prompt = f"Question: {question}\nPlease choose the most appropriate option from: {', '.join(options)}\n \
      				Response with exactly one of the given options. Be sure to consider the images below.\n"
    content = []
    content.append({
        "type": "text",
        "text": prompt
    })
    for i, base64_image in enumerate(base64_images):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions about traffic light images."},
                {"role": "user",
                 "content": content
                }
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error with GPT API call: {e}")
        return "unknown"

for image_name in os.listdir(image_folder):
    if not image_name.endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(image_folder, image_name)
    label_path = os.path.join(label_folder, image_name.replace('.jpg', '.json').replace('.png', '.json'))

    with open(label_path, 'r') as f:
        label_data = json.load(f)

    question = label_data['question']
    options = label_data['options']
    correct_answer = options[0]  # First option is correct

    with Image.open(image_path) as img:
        width, height = img.size
        max_dim = max(width, height) 

    yolo_results = model.predict(source=image_path, imgsz=max_dim, conf=0.3, classes=[9])
    detections = yolo_results[0].boxes  # Get detections for this image

    bounding_boxes = []

    img = Image.open(image_path)
    cropped_pathes = []

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
        bounding_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        cropped_path = f"temp_cropped_{i}.jpg"
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(cropped_path)
        cropped_pathes.append(cropped_path)
    
    response = get_gpt_response(question, options, cropped_pathes)

    for cropped_path in cropped_pathes:
        os.remove(cropped_path)

    is_correct = correct_answer in response

    results.append({
        "image": image_name,
        "bounding_boxes": bounding_boxes,
        "gpt_response": response,
        "correct_response": correct_answer,
        "correctness": is_correct
    })

with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")

