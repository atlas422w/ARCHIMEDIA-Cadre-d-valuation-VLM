import json
from modelqwen3 import loading_qwen3_VL_8B, request_qwen
from tools import configurer_le_prompt,appliquer_briques_prompt,generer_html_dashboard,mv_tmp_home
from modelllava import loading_llava_hf_7B,request_llava
import os
import shutil
import re

os.environ["HF_HOME"] = "/tmp/huggingface_cache"
data_path="json/data_base.jsonl"
out_path="/tmp/resultat.json"


def select_model():
    choix_du_model= int(input("1)loading_qwen3_VL_8B \n2)llava-1.5-7b-hf \nChoissiser un modele: \n"))
    if choix_du_model == 1:
        model, processor = loading_qwen3_VL_8B()
        return choix_du_model, model, processor
    elif choix_du_model == 2:
        model, processor = loading_llava_hf_7B()
        return choix_du_model, model, processor

def request(image,prompt,model,processor,num_model):
    if num_model==1: 
        return request_qwen(image,prompt,model,processor)
    elif num_model==2:
        return request_llava(image,prompt,model,processor)


def gerer_inference(ligne, model, processor, num_model,config):
    data_ligne = json.loads(ligne)
    image = data_ligne["image_url"]


    with open("json/prompt.json", "r", encoding="utf-8") as f:

        prompt= appliquer_briques_prompt(data_ligne,config,json.load(f))

    data_ligne["prompt_utilise"] = prompt
    try:
        resulte = request(image, prompt, model, processor, num_model)
        data_ligne["resulte"] = resulte
    except Exception as e:
        data_ligne["resulte"] ="error"
    return data_ligne



def main():
    items=[]
    num_model,model,processor=select_model()
    config=configurer_le_prompt()
    lignes_traitees=0
    with open(data_path, "r", encoding="utf-8") as fichier_lecture, \
         open(out_path, "w", encoding="utf-8") as fichier_ecriture:

        for ligne in fichier_lecture:
            
            if lignes_traitees>2:
                break
            resulte = gerer_inference(ligne, model, processor, num_model, config)
            fichier_ecriture.write(json.dumps(resulte, ensure_ascii=False) + "\n")
            items.append(resulte)
            
            lignes_traitees += 1
            print(f"Ligne {lignes_traitees} traitée.")
    model_name = "Qwen3-VL-8B-Instruct" if num_model == 1 else "llava-1.5-7b-hf"
    generer_html_dashboard(items, model_name, "/tmp/dashboard.html")

    mv_tmp_home()
    

if __name__ == "__main__":
    main()