import argparse
from copy import deepcopy
import re
import os

import bleach
import cv2
import gradio as gr
from PIL import Image
import numpy as np
import torch


from visual_search import parse_args, VSM, visual_search
from vstar_bench_eval import normalize_bbox, expand2square, VQA_LLM

import cv2

import logging
from functools import wraps

logging.basicConfig(
    filename="gradio_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_function_call(func):
    """Decorator to log function calls and exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logging.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            logging.info(f"Function {func.__name__} returned {result}")
            return result
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White
def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
	"""Visualizes a single bounding box on the image"""
	x_min, y_min, w, h = bbox
	x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
	
	((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)    
	cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
	cv2.putText(
		img,
		text=class_name,
		org=(x_min, y_min - int(0.3 * text_height)),
		fontFace=cv2.FONT_HERSHEY_SIMPLEX,
		fontScale=0.5, 
		color=TEXT_COLOR, 
		lineType=cv2.LINE_AA,
	)
	return img

def parse_args_vqallm(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
	parser.add_argument("--vqa-model-base", type=str, default=None)
	parser.add_argument("--conv_type", default="v1", type=str,)
	parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
	parser.add_argument("--minimum_size_scale", default=4.0, type=float)
	parser.add_argument("--minimum_size", default=224, type=int)
	return parser.parse_args(args)

args = parse_args_vqallm({})
# init VQA LLM
vqa_llm = VQA_LLM(args)
# init VSM
vsm_args = parse_args({})
vsm_args.version = args.vsm_model_path
vsm = VSM(vsm_args)

missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
focus_msg = "Additional visual information to focus on: "

# Gradio
examples = [
	[
		"Based on the exact content of the flag on the roof, what can we know about its owner?",
		"./assets/example_images/flag.JPG",
	],
	[
		"What is the logo on that bag of bottles carried by the man?",
		"./assets/example_images/bag_of_bottle.jpeg",
	],
	[
		"At which conference did someone get that black mug?",
		"./assets/example_images/blackmug.JPG",
	],	
	[
		"Where to buy a mug like this based on its logo?",
		"./assets/example_images/desktop.webp",
	],
	[
		"Which company does that little doll belong to?",
		"./assets/example_images/doll.JPG",
	],
	[
		"What is the instrument held by an ape?",
		"./assets/example_images/instrument.webp",
	],
	[
		"What color is the liquid in the glass?",
		"./assets/example_images/animate_glass.jpg",
	],
	[
		"Tell me the number of that player who is shooting.",
		"./assets/example_images/nba.png",
	],
	[
		"From the information on the black framed board, how long do we have to wait in line for this attraction?",
		"./assets/example_images/queue_time.jpg",
	],
	[
		"What animal is drawn on that red signicade?",
		"./assets/example_images/signicade.JPG",
	],
	[
		"What kind of drink can we buy from that vending machine?",
		"./assets/example_images/vending_machine.jpg",
	]
]

title = "V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs"

description = """
<font size=4>
This is the demo of our SEAL framework with V* visual search mechanism. \n
**Note**: The current framework is built on top of **LLaVA-7b**. \n
**Note**: The current visual search model and search algorithm mainly focus on common objects and single instance cases.\n
</font>
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2312.14135' target='_blank'>
Preprint Paper
</a>
\n
<p style='text-align: center'>
<a href='https://github.com/penghao-wu/vstar' target='_blank'>   Github </a></p>
"""

@log_function_call
def inference(input_str, input_image):
    logging.debug(f"Received input_str: {input_str}, input_image: {input_image}")
    
    try:
        input_str = bleach.clean(input_str)
        if not re.match(r"^[A-Za-z ,.!?\'\"]+$", input_str) or len(input_str) < 1:
            logging.warning(f"Invalid input detected: {input_str}")
            return "[Error] Invalid input.", None, None, None
        
        question = input_str
        image = Image.open(input_image).convert('RGB')
        logging.info(f"Image loaded successfully: {input_image}")
        
        # Preprocess and model inference
        image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
        prediction = vqa_llm.free_form_inference(image, question, max_new_tokens=512)
        logging.debug(f"Initial model prediction: {prediction}")
        
        missing_objects = []
        if missing_objects_msg in prediction:
            missing_objects = prediction.split(missing_objects_msg)[-1].strip().rstrip('.').split(',')
            missing_objects = [obj.strip() for obj in missing_objects]
            logging.info(f"Missing objects identified: {missing_objects}")
        
        if not missing_objects:
            return prediction, None, None, None
        
        search_result, failed_objects = [], []
        for object_name in missing_objects:
            try:
                logging.debug(f"Performing visual search for: {object_name}")
                image = Image.open(input_image).convert('RGB')
                smallest_size = max(int(np.ceil(min(image.width, image.height) / args.minimum_size_scale)), args.minimum_size)
                final_step, path_length, search_successful, all_valid_boxes = visual_search(
                    vsm, image, object_name, confidence_low=0.3, target_bbox=None, smallest_size=smallest_size
                )
                if not search_successful:
                    failed_objects.append(object_name)
                if all_valid_boxes:
                    for bbox in all_valid_boxes:
                        bbox[0] += final_step['bbox'][0]
                        bbox[1] += final_step['bbox'][1]
                        search_result.append({'bbox': bbox.tolist(), 'name': object_name})
                logging.info(f"Search result for {object_name}: {search_result}")
            except Exception as e:
                logging.error(f"Error during visual search for {object_name}: {e}", exc_info=True)
        
        # Generate search result image
        search_result_image = np.array(image).copy()
        for obj in search_result:
            search_result_image = visualize_bbox(search_result_image, obj['bbox'], obj['name'])
        logging.info("Search result visualization complete.")
        
        # Final answer generation
        object_names = [obj['name'] for obj in search_result]
        response = vqa_llm.free_form_inference(image, question, max_new_tokens=512)
        logging.info(f"Final response: {response}")
        
        return f"Need to conduct visual search for: {', '.join(missing_objects)}", \
               f"Search results: {object_names}", search_result_image, response
    
    except Exception as e:
        logging.error("Error in inference pipeline.", exc_info=True)
        return "[Error] Something went wrong.", None, None, None

# Attach Gradio interface with the enhanced inference function
demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    outputs=[
        gr.Textbox(lines=1, placeholder=None, label="Direct Answer"),
        gr.Textbox(lines=1, placeholder=None, label="Visual Search Results"),
        gr.Image(type="pil", label="Visual Search Results"),
        gr.Textbox(lines=1, placeholder=None, label="Final Answer"),
    ],
    examples=examples,
    title=title,
    description=description,
    article=article,
    allow_flagging="auto",
)
demo.queue()
demo.launch()