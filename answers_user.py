import json 
import os

def extract_responses_from_file(file_path, username):
    # Extracts annotated responses for a single user from a user_state.json file
    file_responses = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    response_map = data.get("instance_id_to_label_to_value", {})

    for instance_id, annotations in response_map.items():
        if not annotations:
            continue

        associated_comment = None
        for item in annotations:
            if item[0].get("schema") == "comment":
                associated_comment = item[1]

        for item in annotations:
            metadata = item[0]
            response_value = item[1]

            if metadata.get("schema") == "entity_is_illustrated":
                response_entry = {
                    "user": username,
                    "id": instance_id,
                    "response": response_value,
                    "message": None,
                }

                if response_value == "u":
                    response_entry["message"] = associated_comment

                file_responses.append(response_entry)

    return file_responses


def scan_and_retrieve_all(root_folder="annotation_output_O705"):
    # Scans the annotation output folder and collects all user responses across all users
    all_results = []

    for username in os.listdir(root_folder):
        subfolder_path = root_folder + "/" + username

        if os.path.isdir(subfolder_path):
            if "user_state.json" in os.listdir(subfolder_path):
                json_path = subfolder_path + "/user_state.json"

                file_data = extract_responses_from_file(
                    json_path, username
                )
                all_results.extend(file_data)

    return all_results

def analyze_responses_by_question(all_results):
    # Aggregates responses per question and computes yes/no/uncertain percentages
    question_analysis = {}

    for row in all_results:
        question_id = row["id"]
        response = row["response"]
        comment = row["message"]

        if question_id not in question_analysis:
            question_analysis[question_id] = {
                "total": 0,
                "yes": 0,
                "no": 0,
                "uncertain": 0,
                "uncertain_comments": [],
            }

        question_analysis[question_id]["total"] += 1

        if response == "y":
            question_analysis[question_id]["yes"] += 1
        elif response == "n":
            question_analysis[question_id]["no"] += 1
        elif response == "u":
            question_analysis[question_id]["uncertain"] += 1
            if comment:
                question_analysis[question_id]["uncertain_comments"].append(
                    comment
                )

    final_results = []

    for question_id, stats in question_analysis.items():
        total = stats["total"]

        avg_yes = (stats["yes"] / total) * 100
        avg_no = (stats["no"] / total) * 100
        avg_uncertain = (stats["uncertain"] / total) * 100

        question_summary = {
            "id": question_id,
            "percentage_yes": avg_yes,
            "percentage_no": avg_no,
            "percentage_uncertain": avg_uncertain,
            "uncertain_comments": stats["uncertain_comments"],
        }

        final_results.append(question_summary)

    return final_results


def inject_stats_into_results(stats_list, results_file_path):
    # Merges the human annotation statistics into each line of the results JSONL file
    stats_by_id = {}
    for stat in stats_list:
        stats_by_id[stat["id"]] = stat

    enriched_rows = []

    with open(results_file_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_row = line.strip()
            if clean_row:
                question_data = json.loads(clean_row)
                question_id = question_data["id"]

                if question_id in stats_by_id:
                    human_stats = stats_by_id[question_id]

                    question_data["percentage_yes"] = human_stats[
                        "percentage_yes"
                    ]
                    question_data["percentage_no"] = human_stats[
                        "percentage_no"
                    ]
                    question_data["percentage_uncertain"] = human_stats[
                        "percentage_uncertain"
                    ]
                    question_data["uncertain_comments"] = human_stats[
                        "uncertain_comments"
                    ]

                enriched_rows.append(question_data)

    with open(results_file_path, "w", encoding="utf-8") as f:
        for obj in enriched_rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")