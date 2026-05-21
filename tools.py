import requests
from PIL import Image
from io import BytesIO
import json
import random
import urllib.parse
import os
import shutil
def recuperer_images_wikipedia_qwen(images_paths):

    headers = {"User-Agent": "TestBot/1.0 (coolbot@example.org)"}
    
    response = requests.get(images_paths, headers=headers, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    return img

def recuperer_images_wikipedia_llava(images_paths):
    headers = {"User-Agent": "TestBot/1.0 (coolbot@example.org)"}
    
    
    url_propre = urllib.parse.unquote(images_paths)
    raw_image = Image.open(requests.get(url_propre, headers=headers, timeout=10, stream=True).raw).convert("RGB")

    return raw_image

def appliquer_briques_prompt(data_ligne, config, templates):

    briques = []
    mention = data_ligne["caption"]
    
    if config["type_prompt"] == "free":
        briques.append(templates["free"].format(mention=mention))
        
    elif config["type_prompt"] == "contrast":
        briques.append(templates["contrast"][0].format(mention=mention))
        
        opt = config["option_contrast"]
        if opt in templates["contrast"][1]:

            tous_les_candidats = [{"name": data_ligne["entity_name"], "desc": data_ligne.get("entity_desc", ""), "thumb": data_ligne.get("entity_thumb", "")}]
            for alt in data_ligne["alternatives"]:
                tous_les_candidats.append({"name": alt["name"], "desc": alt.get("desc", ""), "thumb": alt.get("thumb", "")})
            
            random.shuffle(tous_les_candidats)
            

            lignes = []
            for i, c in enumerate(tous_les_candidats):
                if opt == "possibilities-nom":
                    lignes.append(f"{i+1}. {c['name']}")
                elif opt == "possibilities-nom-description":
                    lignes.append(f"{i+1}. {c['name']}: {c['desc']}")
                elif opt == "possibilities-nom-description-image":
                    lignes.append(f"{i+1}. {c['name']}: {c['desc']} (Image: {c['thumb']})")

            texte_option = templates["contrast"][1][opt].replace("{candidates}", "\n".join(lignes))
            briques.append(texte_option)
            

    if config["inclure_none"]:
        briques.append(templates["None"])
        
    briques.append(templates["response format"])
    
    return "\n".join(briques)



def configurer_le_prompt():
    config = {
        "type_prompt": None,
        "option_contrast": None,
        "inclure_none": False
    }

    choix = int(input("Choose between: 1-free, 2-contrast : "))
    if choix == 1:
        config["type_prompt"] = "free"
    else:
        config["type_prompt"] = "contrast"
        
        print("\n1. possibilities-nom\n2. possibilities-nom-description\n3. possibilities-nom-description-image")
        opt_choix = int(input("Choose your option (1, 2 or 3) : "))
        if opt_choix == 1:
            config["option_contrast"] = "possibilities-nom"
        elif opt_choix == 2:
            config["option_contrast"] = "possibilities-nom-description"
        else:
            config["option_contrast"] = "possibilities-nom-description-image"

    none_choix = input("\nCan the model answer None if it doesn't know? (True/False) : ").strip()
    if none_choix == "True":
        config["inclure_none"] = True
    print(config)
    return config


def mv_tmp_home():
    os.makedirs("resultat", exist_ok=True)
    json_projet = "resultat/resultat.json"
    html_projet = "resultat/dashboard.html"
    if os.path.exists(json_projet): os.remove(json_projet)
    if os.path.exists(html_projet): os.remove(html_projet)
    
    if os.path.exists("/tmp/resultat.json"):
        shutil.copy2("/tmp/resultat.json", json_projet)
        os.remove("/tmp/resultat.json")
        
    if os.path.exists("/tmp/dashboard.html"):
        shutil.copy2("/tmp/dashboard.html", html_projet)
        os.remove("/tmp/dashboard.html")
    
def generer_html_dashboard(items, model_name, html_out_path="/tmp/dashboard.html"):
    """Génère la page HTML en évaluant strictement les résultats du fichier JSON."""
    total = len(items)
    correctes = 0
    incorrectes = 0
    total_none = 0

    # 1. Calcul des statistiques réelles basées sur ton JSON
    for item in items:
        resulte_modele = str(item.get("resulte", "")).strip()
        bonne_reponse = str(item.get("entity_name", "")).strip()
        
        if resulte_modele == "error":
            continue
            
        if resulte_modele.lower() == "none":
            total_none += 1
        elif "Réponse :" in resulte_modele and "'" in resulte_modele:
            try:
                choix = resulte_modele.split("'")[1].strip()
                if choix.lower() == bonne_reponse.lower(): correctes += 1
                else: incorrectes += 1
            except IndexError: incorrectes += 1
        elif "Answer:" in resulte_modele and "'" in resulte_modele:
            try:
                choix = resulte_modele.split("'")[1].strip()
                if choix.lower() == bonne_reponse.lower(): correctes += 1
                else: incorrectes += 1
            except IndexError: incorrectes += 1
        else:
            if bonne_reponse.lower() in resulte_modele.lower(): correctes += 1
            else: incorrectes += 1

    # 2. Structure HTML
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Dashboard ARCHIMEDIA - Évaluation Entity Linking</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-100 font-sans text-slate-800 antialiased min-h-screen">
    <header class="bg-slate-900 text-white shadow-md sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex flex-col sm:flex-row justify-between items-center gap-4">
            <div>
                <h1 class="text-xl font-bold tracking-tight">ARCHIMEDIA — Cadre d'Évaluation VLM</h1>
                <p class="text-xs text-slate-400">Modèle actif : <span class="text-amber-400 font-semibold">{model_name}</span></p>
            </div>
            <div class="flex flex-wrap gap-4 text-center text-sm">
                <div class="bg-slate-800 px-4 py-1.5 rounded-md border border-slate-700"><span class="text-xs text-slate-400 block">Total</span><span class="font-bold">{total}</span></div>
                <div class="bg-emerald-950/40 px-4 py-1.5 rounded-md border border-emerald-500/30"><span class="text-xs text-emerald-400 block">✅ Correctes</span><span class="font-bold text-emerald-400">{correctes}</span></div>
                <div class="bg-amber-950/40 px-4 py-1.5 rounded-md border border-amber-500/30"><span class="text-xs text-amber-400 block">🟨 Réponses None</span><span class="font-bold text-amber-400">{total_none}</span></div>
                <div class="bg-rose-950/40 px-4 py-1.5 rounded-md border border-rose-500/30"><span class="text-xs text-rose-400 block">❌ Incorrectes</span><span class="font-bold text-rose-400">{incorrectes}</span></div>
            </div>
        </div>
    </header>
    <main class="max-w-6xl mx-auto px-4 py-8 space-y-12">"""

    # 3. Cartes par exemple
    for idx, item in enumerate(items):
        caption = item.get("caption", "Aucune mention")
        image_url = item.get("image_url", "")
        resulte_modele = str(item.get("resulte", "")).strip()
        prompt_utilise = item.get("prompt_utilise", "Prompt non sauvegardé")
        bonne_reponse = item.get("entity_name", "").strip()
        
        choix_extrait = ""
        if "'" in resulte_modele:
            try: choix_extrait = resulte_modele.split("'")[1].strip()
            except IndexError: pass
        elif resulte_modele.lower() == "none":
            choix_extrait = "none"

        if resulte_modele == "error":
            badge_statut = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-slate-200 text-slate-700">⚠️ CRASH INFERENCE</span>'
        elif choix_extrait.lower() == "none":
            badge_statut = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-amber-500 text-white">🟨 PRÉDICTION : NONE</span>'
        elif choix_extrait.lower() == bonne_reponse.lower() or (choix_extrait == "" and bonne_reponse.lower() in resulte_modele.lower()):
            badge_statut = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-emerald-600 text-white">🟩 BONNE RÉPONSE</span>'
        else:
            badge_statut = '<span class="px-3 py-1 text-xs font-bold rounded-full bg-rose-600 text-white">🟥 MAUVAISE RÉPONSE</span>'

        candidats = []
        if "entity_name" in item:
            candidats.append({"name": item["entity_name"], "desc": item.get("entity_desc", "Pas de description."), "thumb": item.get("entity_thumb", ""), "is_truth": True})
        if "alternatives" in item and isinstance(item["alternatives"], list):
            for cand in item["alternatives"]:
                candidats.append({"name": cand.get("name", "Inconnu"), "desc": cand.get("desc", "Pas de description."), "thumb": cand.get("thumb", ""), "is_truth": False})

        html_content += f"""
        <article class="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden p-6 space-y-6">
            <div class="border-b border-slate-100 pb-3 flex flex-wrap justify-between items-center gap-2">
                <div class="flex items-center gap-3">
                    <span class="text-xs font-mono font-bold uppercase tracking-wider bg-slate-800 text-white px-3 py-1 rounded-full">Exemple #{idx + 1}</span>
                    {badge_statut}
                </div>
                <span class="text-sm font-medium text-slate-500">Mention : <strong class="text-slate-900">"{caption}"</strong></span>
            </div>
            <div class="bg-slate-50 rounded-xl p-5 border border-slate-200/60">
                <h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">📍 CASE 1 : Input</h4>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="md:col-span-1 flex flex-col items-center justify-center bg-white p-2 rounded-lg border border-slate-200">
                        <img src="{image_url}" class="max-h-40 object-contain rounded" onerror="this.src='https://placehold.co/150x150?text=Pas+d+Image';">
                    </div>
                    <div class="md:col-span-3"><div class="bg-slate-900 rounded-lg p-4 font-mono text-xs text-emerald-400 overflow-y-auto max-h-40 whitespace-pre-wrap">{prompt_utilise}</div></div>
                </div>
            </div>
            <div class="bg-slate-50 rounded-xl p-5 border border-slate-200/60">
                <h4 class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">🗂️ CASE 2 : Candidats</h4>
                <div class="space-y-3">"""

        for cand in candidats:
            style_border = "border border-slate-200 bg-white"
            badge_choix = ""
            if cand["is_truth"]:
                style_border = "border-2 border-emerald-500 bg-emerald-50/20"
                badge_choix += '<span class="bg-emerald-600 text-white text-[9px] font-extrabold px-2 py-0.5 rounded shadow">VÉRITÉ TERRAIN</span> '
            if choix_extrait.lower() == cand["name"].lower() and choix_extrait.lower() != "none":
                style_border = "border-2 border-indigo-600 bg-indigo-50/50 shadow-md ring-4 ring-indigo-100"
                badge_choix += '<span class="bg-indigo-600 text-white text-[9px] font-extrabold px-2 py-0.5 rounded shadow">SÉLECTIONNÉ</span>'

            html_content += f"""
                    <div class="relative flex gap-4 p-4 rounded-xl {style_border}">
                        <div class="absolute top-2 right-2 flex gap-1">{badge_choix}</div>
                        <div class="w-16 h-16 bg-slate-100 rounded-lg overflow-hidden border flex items-center justify-center flex-shrink-0">
                            <img src="{cand["thumb"]}" class="object-cover w-full h-full" onerror="this.src='https://placehold.co/100x100?text=N/A';">
                        </div>
                        <div class="flex-1 pr-36">
                            <h5 class="text-sm font-bold text-slate-900">{cand["name"]}</h5>
                            <p class="text-xs text-slate-500 mt-1 line-clamp-2">{cand["desc"]}</p>
                        </div>
                    </div>"""

        if choix_extrait.lower() == "none":
            html_content += f"""
                    <div class="relative flex items-center gap-4 p-4 rounded-xl border-2 border-amber-500 bg-amber-50/40">
                        <span class="absolute top-2 right-2 bg-amber-600 text-white text-[9px] font-extrabold px-2 py-0.5 rounded shadow">SÉLECTIONNÉ</span>
                        <div class="w-12 h-12 bg-amber-100 text-amber-700 rounded-lg flex items-center justify-center font-bold text-xs">🗙</div>
                        <div class="flex-1">
                            <h5 class="text-sm font-bold text-amber-900">None (Aucune entité ne correspond / Le modèle ne sait pas)</h5>
                        </div>
                    </div>"""

        html_content += f"""
                </div>
            </div>
            <div class="bg-indigo-950 text-indigo-100 rounded-xl p-5 border border-indigo-900">
                <h4 class="text-xs font-bold text-indigo-400 uppercase tracking-wider mb-2">💬 CASE 3 : Justification</h4>
                <div class="text-sm bg-indigo-900/40 p-4 rounded-lg whitespace-pre-wrap text-white">{resulte_modele}</div>
            </div>
        </article>"""

    html_content += """</main></body></html>"""
    with open(html_out_path, "w", encoding="utf-8") as f: f.write(html_content)