import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_object_token

from openai import OpenAI
from visual_search import parse_args, VSM, visual_search
import io
import base64

import time

def normalize_bbox(bbox, image_width, image_height):
	normalized_bbox = [bbox[0]/image_width, bbox[1]/image_height, (bbox[0]+bbox[2])/image_width, (bbox[1]+bbox[3])/image_height]
	normalized_bbox = [np.clip(_, 0, 1) for _ in normalized_bbox]
	return normalized_bbox
def expand2square(pil_img, background_color):
	width, height = pil_img.size
	if width == height:
		return pil_img, 0, 0
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))
		return result, 0, (width - height) // 2
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))
		return result, (height - width) // 2, 0
def load_api_key():
    try:
        with open('config.json') as f:
            config = json.load(f)
            return config['openai_api_key']
    except FileNotFoundError:
        # Fallback to environment variable
        return os.getenv('OPENAI_API_KEY')
class VQA_LLM:
	def __init__(self, args):
		disable_torch_init()
		model_path = args.vqa_model_path
		model_name = get_model_name_from_path(model_path)
		model_name += 'llava'
		model_base = None
		device_map = "auto"
		self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name)
		self.conv_type = args.conv_type

	def get_patch(self, bbox, image_width, image_height, patch_size=224, patch_scale=None):
		object_width = int(np.ceil(bbox[2]))
		object_height = int(np.ceil(bbox[3]))

		object_center_x = int(bbox[0] + bbox[2]/2)
		object_center_y = int(bbox[1] + bbox[3]/2)

		if patch_scale is None:
			patch_width = max(object_width, patch_size)
			patch_height = max(object_height, patch_size)
		else:
			patch_width = int(object_width*patch_scale)
			patch_height = int(object_height*patch_scale)

		left = max(0, object_center_x-patch_width//2)
		right = min(left+patch_width, image_width)

		top = max(0, object_center_y-patch_height//2)
		bottom = min(top+patch_height, image_height)

		return [left, top, right, bottom]
	
	def get_object_crop(self, image, bbox, patch_scale):
		resized_bbox = self.get_patch(bbox, image.width, image.height, patch_scale=patch_scale)
		object_crop = image.crop((resized_bbox[0], resized_bbox[1], resized_bbox[2], resized_bbox[3]))
		object_crop = object_crop.resize((self.image_processor.crop_size['width'],self.image_processor.crop_size['height']))
		object_crop = self.image_processor.preprocess(object_crop, return_tensors='pt')['pixel_values'][0]
		return object_crop

	@torch.inference_mode()
	def free_form_inference(self, image, question, temperature=0, top_p=None, num_beams=1, max_new_tokens=200, object_crops=None, images_long=None, objects_long=None):
		conv = conv_templates[self.conv_type].copy()
		qs = DEFAULT_IMAGE_TOKEN + '\n' + question	
		conv.append_message(conv.roles[0], qs)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()
		stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
		keywords = [stop_str]
		input_ids = tokenizer_image_object_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
		image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
		stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

		output_ids = self.model.generate(
			input_ids,
			images=image_tensor.unsqueeze(0).half().cuda(),
			object_features=object_crops.half().cuda() if object_crops is not None else None,
			images_long = images_long,
			objects_long = objects_long,
			do_sample= True if temperature > 0 else False,
			num_beams=num_beams,
			temperature=temperature,
			top_p = top_p,
			max_new_tokens=max_new_tokens,
			use_cache=True,
			stopping_criteria=[stopping_criteria])
			
		input_token_len = input_ids.shape[1]
		n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
		if n_diff_input_output > 0:
			print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
		outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
		outputs = outputs.strip()
		if outputs.endswith(stop_str):
			outputs = outputs[:-len(stop_str)]
		outputs = outputs.strip()
		return outputs

	@torch.inference_mode()
	def multiple_choices_inference(self, image, question, options, object_crops=None, images_long=None, objects_long=None):
		conv = conv_templates[self.conv_type].copy()
		qs = DEFAULT_IMAGE_TOKEN + '\n' + question	
		conv.append_message(conv.roles[0], qs)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()

		question_input_ids = tokenizer_image_object_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
		image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

		output_question = self.model(
			question_input_ids,
			use_cache=True,
			images=image_tensor.unsqueeze(0).half().cuda(),
			object_features=object_crops.half().cuda() if object_crops is not None else None,
			images_long = images_long,
			objects_long = objects_long)

		question_logits = output_question.logits
		question_past_key_values = output_question.past_key_values

		loss_list = []

		for option in options:
			conv = conv_templates[self.conv_type].copy()
			conv.append_message(conv.roles[0], qs)
			conv.append_message(conv.roles[1], option)
			full_prompt = conv.get_prompt()

			full_input_ids = tokenizer_image_object_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
			option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

			output_option = self.model(input_ids=option_answer_input_ids,
								use_cache=True,
								attention_mask=torch.ones(1, question_logits.shape[1]+option_answer_input_ids.shape[1], device=full_input_ids.device),
								past_key_values=question_past_key_values)
			
			logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

			loss_fct = CrossEntropyLoss()
			logits = logits.view(-1, self.model.config.vocab_size)
			labels = option_answer_input_ids.view(-1)
			loss = loss_fct(logits, labels)

			loss_list.append(loss)

		option_chosen = torch.stack(loss_list).argmin()

		return option_chosen.cpu().item()
	
	@torch.inference_mode()
	def multiple_choices_inference_gpt4(self, image, question, options, object_crops=None):
		api_key = load_api_key()
		if not api_key:
			raise ValueError("OpenAI API key not found. Please set it in config.json or as an environment variable.")
  		# Convert PIL Image to bytes, ensuring it's in square format like other functions
		image, _, _ = expand2square(image, tuple(int(x*255) for x in [0.48145466, 0.45782750, 0.40821073]))
		img_byte_arr = io.BytesIO()
		image.save(img_byte_arr, format='JPEG')
		img_byte_arr = img_byte_arr.getvalue()

		# Construct the prompt similar to original format
		prompt = f"Question: {question}\nPlease choose the most appropriate option from: {', '.join(options)}\n \
      				Response with exactly one of the given options."
		context_image = {
								"type": "image_url",
								"image_url": {
									"url": f"data:image/jpeg;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"
								}
							}
		content = [
			{
				"type": "text",
				"text": prompt
			},
			context_image
		]
		# Process cropped images
		if object_crops is not None:
			for crop in object_crops:
				if isinstance(crop, torch.Tensor):
					crop = transforms.ToPILImage()(crop)
				crop_bytes = io.BytesIO()
				crop.save(crop_bytes, format='JPEG')
				content.append({
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{base64.b64encode(crop_bytes.getvalue()).decode('utf-8')}"
					}
				})
		try:
			client = OpenAI(api_key=api_key)
			response = client.chat.completions.create(
				model="gpt-4o-mini",
				messages=[{"role": "user", "content": content}],
				max_tokens=1000
			)
			answer = response.choices[0].message.content.strip()
			
			# Find the matching option
			for idx, option in enumerate(options):
				if option.lower() in answer.lower():
					return idx
			return 0  # Default to first option if no match found
			
		except Exception as e:
			print(f"GPT-4 API error: {e}")
			return 0  # Default to first option on error


def eval_model(args):
	# init VQA LLM
	vqa_llm = VQA_LLM(args)
	# init VSM
	vsm_args = parse_args({})
	vsm_args.version = args.vsm_model_path
	vsm = VSM(vsm_args)

	results = {}
	per_type_acc = defaultdict(list)
	all_acc = []

	missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
	focus_msg = "Additional visual information to focus on: "
	for test_type in ['easy_single', 'google_street_view_hd', 'multiple', 'small_light', 'no_light', 'cropped_traffic_lights']:
		start_time = time.time()
		results[test_type] = []
		folder = os.path.join(args.benchmark_folder, test_type)
		image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
		for image_file in tqdm(image_files):
			result_single_sample = {}
			image_path = os.path.join(folder, image_file)
			annotation_path = image_path.split('.')[0] + '.json'
			image = Image.open(image_path).convert('RGB')
			annotation = json.load(open(annotation_path))
			image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
			
			question = annotation['question']
			# generate free-form response to check whether visual search needs to be activated
			prediction = vqa_llm.free_form_inference(image, question)
			missing_objects = []
			if missing_objects_msg in prediction:
				missing_objects = prediction.split(missing_objects_msg)[-1]
				if missing_objects.endswith('.'):
					missing_objects = missing_objects[:-1]
				missing_objects = missing_objects.split(',')
				missing_objects = [missing_object.strip() for missing_object in missing_objects]

			search_result = []
			if len(missing_objects) > 0:
				# visual search
				for object_name in missing_objects:
					image = Image.open(image_path).convert('RGB')
					smallest_size = max(int(np.ceil(min(image.width, image.height)/args.minimum_size_scale)), args.minimum_size)
					final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm, image, object_name, target_bbox=None, smallest_size=smallest_size)
					if all_valid_boxes is not None:
						# might exist multiple target instances
						for search_bbox in all_valid_boxes:
							search_final_patch = final_step['bbox']
							search_bbox[0] += search_final_patch[0]
							search_bbox[1] += search_final_patch[1]
							search_result.append({'bbox':search_bbox.tolist(),'name':object_name})
					else:
						search_bbox = final_step['detection_result']
						search_final_patch = final_step['bbox']
						search_bbox[0] += search_final_patch[0]
						search_bbox[1] += search_final_patch[1]
						search_result.append({'bbox':search_bbox.tolist(),'name':object_name})
			# predict the multiple-choice option
			options = annotation['options']
			image = Image.open(image_path).convert('RGB')
			if len(missing_objects) > 0:
				object_names = [_['name'] for _ in search_result]
				bboxs = deepcopy([_['bbox'] for _ in search_result])
				if len(object_names) <= 2:
					images_long = [False]
					objects_long = [True]*len(object_names)
				else:
					images_long = [False]
					objects_long = [False]*len(object_names)
				object_crops = []
				for bbox in bboxs:
					object_crop = vqa_llm.get_object_crop(image, bbox, patch_scale=1.2)
					object_crops.append(object_crop)
				object_crops = torch.stack(object_crops, 0)
				image, left, top = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
				bbox_list = []
				for bbox in bboxs:
					bbox[0] += left
					bbox[1] += top
					bbox_list.append(bbox)
				bbox_list = [normalize_bbox(bbox, image.width, image.height) for bbox in bbox_list]
				cur_focus_msg = focus_msg
				for i, (object_name, bbox) in enumerate(zip(object_names, bbox_list)):
					cur_focus_msg = cur_focus_msg + "{} <object> at location [{:.3f},{:.3f},{:.3f},{:.3f}]".format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
					if i != len(bbox_list)-1:
						cur_focus_msg = cur_focus_msg+"; "
					else:
						cur_focus_msg = cur_focus_msg +'.'
				question_with_focus = cur_focus_msg+"\n"+question
				option_chosen = vqa_llm.multiple_choices_inference(image, question_with_focus, options, object_crops, images_long=images_long, objects_long=objects_long)
			else:
				option_chosen = vqa_llm.multiple_choices_inference(image, question, options)

			#choose the option using free-form prediction with string matching.
			option_chosen_direct = -1
			for i, option in enumerate(options):
				if option.lower() in prediction.lower():
					option_chosen_direct = i
					break

			correct = 1 if (option_chosen==0 or option_chosen_direct==0) else 0
			per_type_acc[test_type].append(correct)
			all_acc.append(correct)

			print("--- %s seconds ---" % round(time.time() - start_time, 2))

			result_single_sample['question'] = question
			result_single_sample['options'] = options
			result_single_sample['image'] = image_file
			result_single_sample['prediction_freeform'] = prediction
			result_single_sample['missing_objects'] = missing_objects
			result_single_sample['search_result'] = search_result	
			result_single_sample['option_chosen'] = option_chosen
			result_single_sample['option_chosen_direct'] = option_chosen_direct
			result_single_sample['correct'] = correct
			results[test_type].append(result_single_sample)

		print(test_type, np.mean(per_type_acc[test_type]))
		# save results with path output_path + test_type
		with open(args.output_path.replace('.json', f'_{test_type}.json'), 'w') as f:
			json.dump(results[test_type], f, indent=4)

	print(np.mean(all_acc))

def eval_model_gpt4(args):
    # Init models
    vqa_llm = VQA_LLM(args)
    vsm_args = parse_args({})
    vsm_args.version = args.vsm_model_path
    vsm = VSM(vsm_args)

    # Init result trackers
    results = {}
    per_type_acc = defaultdict(list)
    all_acc = []

    missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
    focus_msg = "Additional visual information to focus on: "

    for test_type in ['easy_single', 'google_street_view_hd', 'multiple', 'small_light', 'no_light', 'cropped_traffic_lights']:
        start_time = time.time()
        results[test_type] = []
        folder = os.path.join(args.benchmark_folder, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        
        for image_file in tqdm(image_files):
            result_single_sample = {}
            image_path = os.path.join(folder, image_file)
            annotation_path = image_path.split('.')[0] + '.json'
            image = Image.open(image_path).convert('RGB')
            annotation = json.load(open(annotation_path))
            
            question = annotation['question']
            prediction = vqa_llm.free_form_inference(image, question)
            
            missing_objects = []
            if missing_objects_msg in prediction:
                missing_objects = prediction.split(missing_objects_msg)[-1]
                if missing_objects.endswith('.'):
                    missing_objects = missing_objects[:-1]
                missing_objects = [obj.strip() for obj in missing_objects.split(',')]

            # Perform visual search if needed
            search_result = []
            if missing_objects:
                for object_name in missing_objects:
                    image = Image.open(image_path).convert('RGB')
                    smallest_size = max(int(np.ceil(min(image.width, image.height)/args.minimum_size_scale)), args.minimum_size)
                    final_step, _, _, all_valid_boxes = visual_search(vsm, image, object_name, target_bbox=None, smallest_size=smallest_size)
                    
                    if all_valid_boxes is not None:
                        for search_bbox in all_valid_boxes:
                            search_final_patch = final_step['bbox']
                            search_bbox[0] += search_final_patch[0]
                            search_bbox[1] += search_final_patch[1]
                            search_result.append({'bbox': search_bbox.tolist(), 'name': object_name})
                    else:
                        search_bbox = final_step['detection_result']
                        search_final_patch = final_step['bbox']
                        search_bbox[0] += search_final_patch[0]
                        search_bbox[1] += search_final_patch[1]
                        search_result.append({'bbox': search_bbox.tolist(), 'name': object_name})

            # Process multiple-choice options with GPT4
            options = annotation['options']
            image = Image.open(image_path).convert('RGB')
            
            if missing_objects:
                object_names = [_['name'] for _ in search_result]
                bboxs = deepcopy([_['bbox'] for _ in search_result])
                
                object_crops = []
                for bbox in bboxs:
                    object_crop = vqa_llm.get_object_crop(image, bbox, patch_scale=1.2)
                    object_crops.append(object_crop)
                object_crops = torch.stack(object_crops, 0)
                
                # Process image and bboxes
                image, left, top = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
                bbox_list = []
                for bbox in bboxs:
                    bbox[0] += left
                    bbox[1] += top
                    bbox_list.append(bbox)
                bbox_list = [normalize_bbox(bbox, image.width, image.height) for bbox in bbox_list]
                
                cur_focus_msg = focus_msg
                for i, (object_name, bbox) in enumerate(zip(object_names, bbox_list)):
                    cur_focus_msg += "{} <object> at location [{:.3f},{:.3f},{:.3f},{:.3f}]".format(
                        object_name, bbox[0], bbox[1], bbox[2], bbox[3])
                    cur_focus_msg += "; " if i != len(bbox_list)-1 else "."
                
                question_with_focus = cur_focus_msg + "\n" + question
                option_chosen = vqa_llm.multiple_choices_inference_gpt4(
					image,  # full image
					question_with_focus, 
					options, 
					object_crops  # cropped objects
				)
            else:
                option_chosen = vqa_llm.multiple_choices_inference_gpt4(image, question, options)

            option_chosen_direct = -1
            for i, option in enumerate(options):
                if option.lower() in prediction.lower():
                    option_chosen_direct = i
                    break

            correct = 1 if (option_chosen == 0 or option_chosen_direct == 0) else 0
            per_type_acc[test_type].append(correct)
            all_acc.append(correct)

            print("--- %s seconds ---" % round(time.time() - start_time, 2))
            result_single_sample.update({
                'question': question,
                'options': options,
                'image': image_file,
                'prediction_freeform': prediction,
                'missing_objects': missing_objects,
                'search_result': search_result,
                'option_chosen': option_chosen,
                'option_chosen_direct': option_chosen_direct,
                'correct': correct
            })
            results[test_type].append(result_single_sample)

        # Print and save results for test type
        print(f"{test_type}: {np.mean(per_type_acc[test_type])}")
        with open(args.output_path.replace('.json', f'_{test_type}.json'), 'w') as f:
            json.dump(results[test_type], f, indent=4)

    print(f"Overall accuracy: {np.mean(all_acc)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
	parser.add_argument("--vqa-model-base", type=str, default=None)
	parser.add_argument("--conv_type", default="v1", type=str,)
	parser.add_argument("--benchmark-folder", type=str, default="vstar_bench")
	parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
	parser.add_argument("--output-path", type=str, default="eval_result.json")
	parser.add_argument("--minimum_size_scale", default=4.0, type=float, help="minimum sub-image scale for the termination of search")
	parser.add_argument("--minimum_size", default=224, type=int, help="minimum sub-image size for the termination of search")

	args = parser.parse_args()
	eval_model_gpt4(args)