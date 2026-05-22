import json
import argparse
from modelqwen3 import loading_qwen3_VL_8B, request_qwen
from tools import apply_prompt_blocks, generate_html_dashboard, move_tmp_to_home
from modelllava import loading_llava_hf_7B, request_llava
from answers_user import scan_and_retrieve_all, analyze_responses_by_question, inject_stats_into_results
import os
import shutil
import re

os.environ["HF_HOME"] = "/tmp/huggingface_cache"
data_path = "json/data_base.jsonl"


def parse_args():
    # Defines and parses all CLI arguments
    parser = argparse.ArgumentParser(description="ARCHIMEDIA VLM Evaluation")

    parser.add_argument(
        "--model", type=int, choices=[1, 2], required=True,
        help="1 = Qwen3-VL-8B, 2 = LLaVA-1.5-7B"
    )
    parser.add_argument(
        "--prompt-type", type=str, choices=["free", "contrast"], required=True,
        help="Prompt strategy"
    )
    parser.add_argument(
        "--contrast-option", type=str,
        choices=["possibilities-nom", "possibilities-nom-description", "possibilities-nom-description-image"],
        help="Required if --prompt-type is contrast"
    )
    parser.add_argument(
        "--none", action="store_true",
        help="Allow the model to answer None"
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Name used for the output files (e.g. run_qwen_free → result/run_qwen_free.json + .html)"
    )

    return parser.parse_args()


def request(image, prompt, model, processor, num_model):
    # Dispatches the inference request to the appropriate model
    if num_model == 1:
        return request_qwen(image, prompt, model, processor)
    elif num_model == 2:
        return request_llava(image, prompt, model, processor)


def handle_inference(line, model, processor, num_model, config):
    # Processes a single JSONL line: builds the prompt, runs inference, and returns enriched data
    line_data = json.loads(line)
    image = line_data["image_url"]

    with open("json/prompt.json", "r", encoding="utf-8") as f:
        prompt = apply_prompt_blocks(line_data, config, json.load(f))

    line_data["prompt_utilise"] = prompt
    try:
        result = request(image, prompt, model, processor, num_model)
        line_data["resulte"] = result
    except Exception as e:
        line_data["resulte"] = "error"
    return line_data

def handle_user_results(out_path):
    # Loads user annotations, computes per-question stats, and injects them into the results file
    raw_data = scan_and_retrieve_all("annotation_output_O705")

    statistics = analyze_responses_by_question(raw_data)

    inject_stats_into_results(statistics, out_path)


def main():
    args = parse_args()

    if args.prompt_type == "contrast" and not args.contrast_option:
        raise ValueError("--contrast-option is required when --prompt-type is contrast")

    config = {
        "type_prompt": args.prompt_type,
        "option_contrast": args.contrast_option,
        "inclure_none": args.none
    }

    if args.model == 1:
        model, processor = loading_qwen3_VL_8B()
        num_model, model_name = 1, "loading_qwen3_VL_8B"
    elif args.model == 2:
        model, processor = loading_llava_hf_7B()
        num_model, model_name = 2, "llava-1.5-7b-hf"

    out_path = f"/tmp/{args.run_name}.json"
    html_tmp_path = f"/tmp/{args.run_name}.html"

    processed_lines = 0
    with open(data_path, "r", encoding="utf-8") as read_file, \
         open(out_path, "w", encoding="utf-8") as write_file:

        for line in read_file:
            if processed_lines > 1:
                break
            result = handle_inference(line, model, processor, num_model, config)
            write_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed_lines += 1
            print(f"Line {processed_lines} processed.")

    handle_user_results(out_path)

    enriched_items = []
    with open(out_path, "r", encoding="utf-8") as read_file:
        for line in read_file:
            if line.strip():
                enriched_items.append(json.loads(line))

    generate_html_dashboard(enriched_items, model_name, html_tmp_path)

    move_tmp_to_home(args.run_name)

if __name__ == "__main__":
    main()