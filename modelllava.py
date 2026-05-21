import requests
from PIL import Image
from io import BytesIO
import urllib.parse
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tools import recuperer_images_wikipedia_llava 

def loading_llava_hf_7B():
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)
    return model,processor

def request_llava(images_paths,prompt,model,processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    
    prompt_formate = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=recuperer_images_wikipedia_llava(images_paths), text=prompt_formate, return_tensors='pt').to(0, torch.float16)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=0.0, num_beams=1)    
    
    prompt_length = inputs["input_ids"].shape[-1]
    res = processor.decode(output[0][prompt_length:], skip_special_tokens=True)
    
    return res.strip()