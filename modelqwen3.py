from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tools import recuperer_images_wikipedia_qwen
def loading_qwen3_VL_8B():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model,processor


def request_qwen(images_paths,prompt,model,processor):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": recuperer_images_wikipedia_qwen(images_paths)},
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
    print("envoie a qwen")
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()