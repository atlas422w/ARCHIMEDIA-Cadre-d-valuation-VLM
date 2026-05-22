from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tools import fetch_wikipedia_image_qwen

def loading_qwen3_VL_8B():
    # Loads and returns the Qwen3-VL 8B model and its processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model, processor


def request_qwen(images_paths, prompt, model, processor):
    # Runs inference with Qwen3-VL on a given image URL and prompt, returns the model's text output
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": fetch_wikipedia_image_qwen(images_paths)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=64,
        do_sample=False,      
        temperature=0.0,     
        num_beams=1           
    )
    print("sending to qwen")
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()