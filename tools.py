import requests
from PIL import Image
from io import BytesIO
import json
import random
import urllib.parse
import os
import shutil

def fetch_wikipedia_image_qwen(images_paths):
    # Fetches an image from a URL and returns it as a PIL RGB image for Qwen
    headers = {"User-Agent": "TestBot/1.0 (coolbot@example.org)"}
    
    response = requests.get(images_paths, headers=headers, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    return img

def fetch_wikipedia_image_llava(images_paths):
    # Fetches an image from a URL (with URL decoding) and returns it as a PIL RGB image for LLaVA
    headers = {"User-Agent": "TestBot/1.0 (coolbot@example.org)"}
    
    clean_url = urllib.parse.unquote(images_paths)
    raw_image = Image.open(requests.get(clean_url, headers=headers, timeout=10, stream=True).raw).convert("RGB")

    return raw_image

def apply_prompt_blocks(line_data, config, templates):
    # Builds the final prompt string by assembling the appropriate blocks based on config
    blocks = []
    mention = line_data["caption"]
    
    if config["type_prompt"] == "free":
        blocks.append(templates["free"].format(mention=mention))
        
    elif config["type_prompt"] == "contrast":
        blocks.append(templates["contrast"][0].format(mention=mention))
        
        opt = config["option_contrast"]
        if opt in templates["contrast"][1]:

            all_candidates = [{"name": line_data["entity_name"], "desc": line_data.get("entity_desc", ""), "thumb": line_data.get("entity_thumb", "")}]
            for alt in line_data["alternatives"]:
                all_candidates.append({"name": alt["name"], "desc": alt.get("desc", ""), "thumb": alt.get("thumb", "")})
            
            random.shuffle(all_candidates)
            

            lines = []
            for i, c in enumerate(all_candidates):
                if opt == "possibilities-nom":
                    lines.append(f"{i+1}. {c['name']}")
                elif opt == "possibilities-nom-description":
                    lines.append(f"{i+1}. {c['name']}: {c['desc']}")
                elif opt == "possibilities-nom-description-image":
                    lines.append(f"{i+1}. {c['name']}: {c['desc']} (Image: {c['thumb']})")

            option_text = templates["contrast"][1][opt].replace("{candidates}", "\n".join(lines))
            blocks.append(option_text)
            

    if config["inclure_none"]:
        blocks.append(templates["None"])
        
    blocks.append(templates["response format"])
    
    return "\n".join(blocks)



def configure_prompt():
    # Interactively asks the user to configure the prompt type and options, returns a config dict
    config = {
        "type_prompt": None,
        "option_contrast": None,
        "inclure_none": False
    }

    choice = int(input("Choose between: 1-free, 2-contrast : "))
    if choice == 1:
        config["type_prompt"] = "free"
    else:
        config["type_prompt"] = "contrast"
        
        print("\n1. possibilities-nom\n2. possibilities-nom-description\n3. possibilities-nom-description-image")
        opt_choice = int(input("Choose your option (1, 2 or 3) : "))
        if opt_choice == 1:
            config["option_contrast"] = "possibilities-nom"
        elif opt_choice == 2:
            config["option_contrast"] = "possibilities-nom-description"
        else:
            config["option_contrast"] = "possibilities-nom-description-image"

    none_choice = input("\nCan the model answer None if it doesn't know? (True/False) : ").strip()
    if none_choice == "True":
        config["inclure_none"] = True
    print(config)
    return config


def move_tmp_to_home(run_name):
    # Moves the result files from /tmp to the local result folder using run_name as filename
    os.makedirs("result", exist_ok=True)
    json_output = f"result/{run_name}.json"
    html_output = f"result/{run_name}.html"
    if os.path.exists(json_output): os.remove(json_output)
    if os.path.exists(html_output): os.remove(html_output)

    tmp_json = f"/tmp/{run_name}.json"
    tmp_html = f"/tmp/{run_name}.html"

    if os.path.exists(tmp_json):
        shutil.copy2(tmp_json, json_output)
        os.remove(tmp_json)

    if os.path.exists(tmp_html):
        shutil.copy2(tmp_html, html_output)
        os.remove(tmp_html)
    
def extract_choice(model_result):
    # Extracts the choice from the model's response. Returns 'none' if None, '' if unrecognized.
    s = model_result.strip()
    if s.lower() == "none":
        return "none"
    if ("Answer:" in s or "Réponse :" in s) and "'" in s:
        try:
            choice = s.split("'")[1].strip()
            return choice  # can be "None" or a real name
        except IndexError:
            pass
    return ""


def generate_html_dashboard(items, model_name, html_out_path="/tmp/dashboard.html"):
    # Generates an HTML dashboard displaying model predictions vs ground truth and user annotation stats
    total = len(items)
    correct = 0
    incorrect = 0
    total_none = 0

    for item in items:
        model_result = str(item.get("resulte", "")).strip()
        correct_answer = str(item.get("entity_name", "")).strip()

        if model_result == "error":
            continue

        extracted_choice = extract_choice(model_result)

        if extracted_choice.lower() == "none":
            total_none += 1
        elif extracted_choice != "":
            if extracted_choice.lower() == correct_answer.lower():
                correct += 1
            else:
                incorrect += 1
        else:
            if correct_answer.lower() in model_result.lower():
                correct += 1
            else:
                incorrect += 1

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dashboard ARCHIMEDIA - Entity Linking Evaluation</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-100 font-sans text-slate-800 antialiased min-h-screen">
    <header class="bg-slate-900 text-white shadow-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
            <div>
                <h1 class="text-xl font-bold tracking-tight">ARCHIMEDIA — VLM Evaluation Framework</h1>
                <p class="text-xs text-slate-400">Active model: <span class="text-amber-400 font-semibold">{model_name}</span></p>
            </div>
            <div class="flex flex-wrap gap-4 text-center text-sm">
                <div class="bg-slate-800 px-4 py-1.5 rounded-md border border-slate-700"><span class="text-xs text-slate-400 block">Total</span><span class="font-bold">{total}</span></div>
                <div class="bg-emerald-950/40 px-4 py-1.5 rounded-md border border-emerald-500/30"><span class="text-xs text-emerald-400 block">✅ Correct</span><span class="font-bold text-emerald-400">{correct}</span></div>
                <div class="bg-amber-950/40 px-4 py-1.5 rounded-md border border-amber-500/30"><span class="text-xs text-amber-400 block">🟨 None Answers</span><span class="font-bold text-amber-400">{total_none}</span></div>
                <div class="bg-rose-950/40 px-4 py-1.5 rounded-md border border-rose-500/30"><span class="text-xs text-rose-400 block">❌ Incorrect</span><span class="font-bold text-rose-400">{incorrect}</span></div>
            </div>
        </div>
    </header>
    <main class="max-w-6xl mx-auto px-4 py-8 space-y-12">"""

    for idx, item in enumerate(items):
        caption = item.get("caption", "No mention")
        image_url = item.get("image_url", "")
        model_result = str(item.get("resulte", "")).strip()
        used_prompt = item.get("prompt_utilise", "Prompt not saved")
        correct_answer = item.get("entity_name", "").strip()

        val_yes = item.get("percentage_yes", 0.0)
        val_no = item.get("percentage_no", 0.0)
        val_unc = item.get("percentage_uncertain", 0.0)
        comments_list = item.get("uncertain_comments", [])

        int_yes = int(float(val_yes)) if val_yes is not None else 0
        int_no = int(float(val_no)) if val_no is not None else 0
        int_unc = int(float(val_unc)) if val_unc is not None else 0

        extracted_choice = extract_choice(model_result)

        if model_result == "error":
            status_badge = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-slate-200 text-slate-700">⚠️ CRASH INFERENCE</span>'
        elif extracted_choice.lower() == "none":
            status_badge = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-amber-500 text-white">🟨 PREDICTION: NONE</span>'
        elif extracted_choice.lower() == correct_answer.lower() or (
            extracted_choice == "" and correct_answer.lower() in model_result.lower()
        ):
            status_badge = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-emerald-600 text-white">🟩 CORRECT ANSWER</span>'
        else:
            status_badge = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-rose-600 text-white">🟥 WRONG ANSWER</span>'

        candidates = []
        if "entity_name" in item:
            candidates.append(
                {
                    "name": item["entity_name"],
                    "desc": item.get("entity_desc", "No description."),
                    "thumb": item.get("entity_thumb", ""),
                    "is_truth": True,
                }
            )
        if "alternatives" in item and isinstance(item["alternatives"], list):
            for cand in item["alternatives"]:
                candidates.append(
                    {
                        "name": cand.get("name", "Unknown"),
                        "desc": cand.get("desc", "No description."),
                        "thumb": cand.get("thumb", ""),
                        "is_truth": False,
                    }
                )

        html_content += f"""
        <article class="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden p-6 space-y-6">
            <div class="border-b border-slate-100 pb-3 flex flex-wrap justify-between items-center gap-2">
                <div class="flex items-center gap-3">
                    <span class="text-xs font-mono font-bold uppercase tracking-wider bg-slate-800 text-white px-3 py-1 rounded-full">Example #{idx + 1}</span>
                    {status_badge}
                </div>
                <span class="text-sm font-medium text-slate-500">Mention: <strong class="text-slate-900">"{caption}"</strong></span>
            </div>
            
            <div class="bg-slate-50 rounded-xl p-5 border border-slate-200/60">
                <h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">📍 CASE 1: Input</h4>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="md:col-span-1 flex flex-col items-center justify-center bg-white p-2 rounded-lg border border-slate-200">
                        <img src="{image_url}" class="max-h-40 object-contain rounded" onerror="this.src='https://placehold.co/150x150?text=No+Image';">
                    </div>
                    <div class="md:col-span-3"><div class="bg-slate-900 rounded-lg p-4 font-mono text-xs text-emerald-400 overflow-y-auto max-h-40 whitespace-pre-wrap">{used_prompt}</div></div>
                </div>
            </div>
            
            <div class="bg-slate-50 rounded-xl p-5 border border-slate-200/60">
                <h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">🗂️ CASE 2: Candidates</h4>
                <div class="space-y-3">"""

        for cand in candidates:
            style_border = "border border-slate-200 bg-white"
            choice_badge = ""
            if cand["is_truth"]:
                style_border = "border-2 border-emerald-500 bg-emerald-50/20"
                choice_badge += '<span class="bg-emerald-600 text-white text-[9px] font-extrabold px-2 py-0.5 rounded shadow">GROUND TRUTH</span> '
            if (
                extracted_choice.lower() == cand["name"].lower()
                and extracted_choice.lower() != "none"
            ):
                style_border = "border-2 border-indigo-600 bg-indigo-50/50 shadow-md ring-4 ring-indigo-100"
                choice_badge += '<span class="bg-indigo-600 text-white text-[9px] font-extrabold px-2 py-0.5 rounded shadow">SELECTED</span>'

            html_content += f"""
                    <div class="relative flex gap-4 p-4 rounded-xl {style_border}">
                        <div class="absolute top-2 right-2 flex gap-1">{choice_badge}</div>
                        <div class="w-16 h-16 bg-slate-100 rounded-lg overflow-hidden border flex items-center justify-center flex-shrink-0">
                            <img src="{cand["thumb"]}" class="object-cover w-full h-full" onerror="this.src='https://placehold.co/100x100?text=N/A';">
                        </div>
                        <div class="flex-1 pr-36">
                            <h5 class="text-sm font-bold text-slate-900">{cand["name"]}</h5>
                            <p class="text-xs text-slate-500 mt-1 line-clamp-2">{cand["desc"]}</p>
                        </div>
                    </div>"""

        if extracted_choice.lower() == "none":
            html_content += f"""
                    <div class="relative flex items-center gap-4 p-4 rounded-xl border-2 border-amber-500 bg-amber-50/40">
                        <span class="absolute top-2 right-2 bg-amber-600 text-white text-[9px] font-extrabold px-2 py-0.5 rounded shadow">SELECTED</span>
                        <div class="w-12 h-12 bg-amber-100 text-amber-700 rounded-lg flex items-center justify-center font-bold text-xs">🗙</div>
                        <div class="flex-1">
                            <h5 class="text-sm font-bold text-amber-900">None (No matching entity / The model does not know)</h5>
                        </div>
                    </div>"""

        html_content += f"""
                </div>
            </div>

            <div class="bg-slate-50 rounded-xl p-5 border border-slate-200/60">
                <h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">👥 CASE 3: User Responses (Humans)</h4>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-center mb-4">
                    <div class="bg-white p-3 rounded-lg border border-slate-200">
                        <span class="text-xs text-slate-500 block">Percentage YES</span>
                        <span class="text-lg font-bold text-emerald-600">{val_yes}%</span>
                        <div class="w-full bg-slate-100 h-2 rounded-full mt-2 overflow-hidden"><div class="bg-emerald-500 h-full" style="width: {int_yes}%"></div></div>
                    </div>
                    <div class="bg-white p-3 rounded-lg border border-slate-200">
                        <span class="text-xs text-slate-500 block">Percentage NO</span>
                        <span class="text-lg font-bold text-rose-600">{val_no}%</span>
                        <div class="w-full bg-slate-100 h-2 rounded-full mt-2 overflow-hidden"><div class="bg-rose-500 h-full" style="width: {int_no}%"></div></div>
                    </div>
                    <div class="bg-white p-3 rounded-lg border border-slate-200">
                        <span class="text-xs text-slate-500 block">Percentage UNCERTAIN</span>
                        <span class="text-lg font-bold text-amber-600">{val_unc}%</span>
                        <div class="w-full bg-slate-100 h-2 rounded-full mt-2 overflow-hidden"><div class="bg-amber-500 h-full" style="width: {int_unc}%"></div></div>
                    </div>
                </div>"""

        if comments_list:
            html_content += """
                <div class="bg-amber-50 border border-amber-200 rounded-lg p-4 mt-3">
                    <h5 class="text-xs font-bold text-amber-800 uppercase tracking-wider mb-2">💬 Uncertainty comments:</h5>
                    <ul class="list-disc list-inside space-y-1.5 text-xs text-amber-900">"""
            for com in comments_list:
                html_content += f"<li>{com}</li>"
            html_content += """
                    </ul>
                </div>"""

        justification_display = model_result
        if extracted_choice.lower() == "none":
            justification_display = model_result if len(model_result) > 6 else "[⚠️ NO EXPLANATION GENERATED: The model only returned the word 'None' without following the explanation format]"

        html_content += f"""
            </div>

            <div class="bg-indigo-950 text-indigo-100 rounded-xl p-5 border border-indigo-900">
                <h4 class="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-2">🤖 CASE 4: Model Justification</h4>
                <div class="text-sm bg-indigo-900/40 p-4 rounded-lg whitespace-pre-wrap text-white">{justification_display}</div>
            </div>
        </article>"""

    html_content += """</main></body></html>"""
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html_content)