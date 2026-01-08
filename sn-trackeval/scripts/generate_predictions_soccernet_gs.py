import os
import json
import random


# Script to generate fake prediction files for the SoccerNet GS task by adding noise the ground truth data

def compute_attribute_choices(predictions):
    role_choices = set()
    jersey_choices = set()
    team_choices = set()

    for prediction in predictions:
        if prediction.get("supercategory") == "object":
            attrs = prediction.get("attributes", {})
            role_choices.add(attrs.get("role"))
            jersey_choices.add(attrs.get("jersey"))
            team_choices.add(attrs.get("team"))

    # Remove None values in case there are predictions without these attributes
    return list(filter(None, role_choices)), list(filter(None, jersey_choices)), list(filter(None, team_choices))


def add_noise_to_value(value, noise_percentage):
    noise = value * noise_percentage * random.choice([-1, 1])
    return value + noise


def add_noise_to_ground_truth(predictions, noise_probability=0.1):
    role_choices, jersey_choices, team_choices = compute_attribute_choices(predictions)

    for prediction in predictions:
        if prediction.get("supercategory") == "object":
            # Apply noise to track_id with a check to avoid negative values
            if random.random() < noise_probability:
                prediction["track_id"] = prediction["track_id"] + 1000

            # Apply noise to bbox_image and bbox_pitch
            for key in ["bbox_image", "bbox_pitch"]:
                if random.random() < noise_probability:
                    bbox = prediction[key]
                    if bbox is None:
                        continue
                    for field in bbox:
                        value = bbox[field]
                        # Ensure the modified value is applied
                        bbox[field] = add_noise_to_value(value, 0.2)  # Adjusted for a 20% noise

            # Randomly change attributes with a 10% chance
            attrs = prediction["attributes"]
            if random.random() < noise_probability:
                current_role = attrs["role"]
                attrs["role"] = random.choice([r for r in role_choices if r != current_role])
            if random.random() < noise_probability:
                current_jersey = attrs["jersey"]
                attrs["jersey"] = random.choice([j for j in jersey_choices if j != current_jersey])
            if random.random() < noise_probability:
                current_team = attrs["team"]
                attrs["team"] = random.choice([t for t in team_choices if t != current_team])


def process_files(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):
            json_file_path = os.path.join(subdir_path, "Labels-GameState.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    data = json.load(file)

                modified_data = {"predictions": data.get("annotations", [])}

                # # Add noise to the ground truth data
                # add_noise_to_ground_truth(modified_data["predictions"])

                new_filename = f"{subdir}.json"
                new_file_path = os.path.join(target_dir, new_filename)

                with open(new_file_path, 'w') as new_file:
                    json.dump(modified_data, new_file, indent=4)

                print(f"Processed and saved: {new_filename}")

    print("All files have been processed.")


path_to_dataset = "/media/ymo/SSD_2TB/SoccerNet-GameState-2025/SN-GSR-2025/"
path_to_predictions = "/media/ymo/SSD_2TB/SoccerNet-GameState-2025/pred_Soccernet_GS2025"
for split in ["valid", "test"]:
    source_dir = os.path.join(path_to_dataset, split)
    target_dir = os.path.join(path_to_predictions, f"SoccerNetGS-{split}/GT")

    process_files(source_dir, target_dir)
