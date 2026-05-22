import requests
from PIL import Image
from io import BytesIO
import urllib.parse
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tools import fetch_wikipedia_image_llava

def loading_llava_hf_7B():
    # Loads and returns the LLaVA 1.5 7B model and its processor
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def request_llava(images_paths, prompt, model, processor):
    # Runs inference with LLaVA on a given image URL and prompt, returns the model's text output
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    
    formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=fetch_wikipedia_image_llava(images_paths), text=formatted_prompt, return_tensors='pt').to(0, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.0, num_beams=1)    
    
    prompt_length = inputs["input_ids"].shape[-1]
    res = processor.decode(output[0][prompt_length:], skip_special_tokens=True)
    
    return res.strip()