import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Path to the radar template image
template_image_path = "sfr/template/Radar.png"
radar_img = cv2.imread(template_image_path)

# Calculate the center of the radar image
center_x_px = radar_img.shape[1] / 2
center_y_px = radar_img.shape[0] / 2


def meter_to_pixel(x_meter, y_meter, center_x_px, center_y_px):
    """
    Convert meter coordinates to pixel coordinates based on the radar image center.
    """
    x_px = int(x_meter + center_x_px)
    y_px = int(y_meter + center_y_px)
    return x_px, y_px


def format_image_name(image_id):
    """
    Convert an image_id to a zero-padded 7-digit file name.

    Args:
        image_id (str): The image ID string (e.g., '20210000001').

    Returns:
        str: The formatted image file name (e.g., '0000001.jpg').
    """
    # Extract the last 7 digits of the image_id and format it as a zero-padded string
    formatted_id = f"{int(image_id[-6:]):06d}"
    return f"{formatted_id}.jpg"


def visualize_predictions(image_folder, output_folder, json_file_path):
    """
    Visualize predictions by drawing bounding boxes on images and plotting player positions on a radar template.
    """
    # Output directory for visualizations
    base_result_dir = os.path.join(output_folder, 'Predict_Visualization')
    if not os.path.exists(base_result_dir):
        os.makedirs(base_result_dir)

    # Load JSON data
    if not os.path.isfile(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Group predictions by image_id and video_id
    image_id_to_predictions = {}
    image_id_to_video_id = {}

    for prediction in data.get('predictions', []):
        image_id = prediction.get('image_id')
        video_id = prediction.get('video_id')

        if image_id not in image_id_to_predictions:
            image_id_to_predictions[image_id] = []
            image_id_to_video_id[image_id] = video_id  # Store video_id for each image_id

        image_id_to_predictions[image_id].append(prediction)

    # Process each frame based on image_id
    for image_id, predictions in image_id_to_predictions.items():
        # Construct the image file name from image_id
        file_name = format_image_name(image_id)

        # Get the corresponding video_id
        video_id = image_id_to_video_id.get(image_id, "unknown")

        # Create a subfolder for the current video_id
        result_dir = os.path.join(base_result_dir, f'SNGS-{video_id}')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Load the image
        image_path = os.path.join(image_folder, file_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image at {image_path}")
            continue

        # Create a copy of the radar template for this frame
        radar_overlay = radar_img.copy()

        # Draw bounding boxes and player positions on the image and radar overlay
        for prediction in predictions:
            bbox_image = prediction.get('bbox_image')
            bbox_pitch = prediction.get('bbox_pitch')
            track_id = prediction.get('track_id')
            attributes = prediction.get("attributes", {})
            role = str(attributes.get("role", "")).lower()

            # Determine drawing color and label text based on role and team.
            if role in ["referee", "other", "ball"]:
                # For these roles, label text is the role and use yellow.
                label_text = role.capitalize()
                draw_color = (0, 255, 255)  # Yellow (BGR)
            else:
                # For player/goalkeeper detections, use jersey and team info.
                jersey = attributes.get("jersey", "")
                if jersey is None or jersey == "100":
                    jersey = "100"
                team = attributes.get("team", "")
                team_label = ""
                if team:
                    if str(team).lower() == "left":
                        team_label = "L"
                        draw_color = (0, 0, 255)  # Red (BGR)
                    elif str(team).lower() == "right":
                        team_label = "R"
                        draw_color = (255, 0, 0)  # Blue (BGR)
                    else:
                        draw_color = (0, 255, 255)  # Yellow as default
                else:
                    draw_color = (0, 255, 255)  # Yellow as default
                label_text = f"ID:{track_id}, J:{jersey}, {team_label}"

            # Draw bounding boxes on the image if bbox_image exists.
            if bbox_image:
                x = int(bbox_image['x'])
                y = int(bbox_image['y'])
                w = int(bbox_image['w'])
                h = int(bbox_image['h'])
                cv2.rectangle(image, (x, y), (x + w, y + h), draw_color, 2)
                text_position = (x, y - 10)
                cv2.putText(image, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

            # Plot player positions on the radar template using the same color.
            if bbox_pitch:
                x_middle_meter = bbox_pitch['x_bottom_middle']
                y_middle_meter = bbox_pitch['y_bottom_middle']
                x_middle_px, y_middle_px = meter_to_pixel(x_middle_meter, y_middle_meter, center_x_px, center_y_px)
                cv2.circle(radar_overlay, (x_middle_px, y_middle_px), 1, draw_color, -1)
                # Optionally, add label text on the radar overlay (small font)
                # cv2.putText(radar_overlay, label_text, (x_middle_px, y_middle_px - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

        # Resize radar overlay for visualization
        radar_resized = cv2.resize(radar_overlay, (image.shape[1] // 4, image.shape[0] // 4))
        template_pos = (image.shape[1] - radar_resized.shape[1] - 10, 10)
        # Overlay the radar template on the image
        image[template_pos[1]:template_pos[1] + radar_resized.shape[0],
        template_pos[0]:template_pos[0] + radar_resized.shape[1]] = radar_resized

        # Save the result image in the video_id subfolder
        output_image_path = os.path.join(result_dir, f"{os.path.splitext(file_name)[0]}.jpg")
        cv2.imwrite(output_image_path, image)
        print(f"Saved visualization to {output_image_path}")


if __name__ == "__main__":

    data_sets = ['test']
    root_directory = "data/SoccerNetGS"

    for data_set in data_sets:
        folder_path = os.path.join(root_directory, data_set)
        for video_name in sorted(os.listdir(folder_path)):
            video_path = os.path.join(folder_path, video_name)
            if not os.path.isdir(video_path):
                continue
            # JSON file is expected to be <video_name>.json inside the video folder
            json_file_path = os.path.join(video_path, f"{video_name}.json")
            if not os.path.isfile(json_file_path):
                print(f"JSON file not found: {json_file_path}")
                continue
            # Iterate over image subfolders (e.g., img1)
            for subfolder in sorted(os.listdir(video_path)):
                image_folder = os.path.join(video_path, subfolder)
                if not os.path.isdir(image_folder):
                    continue
                output_folder = os.path.join(video_path, "visualization")
                print(f"Processing {image_folder} with {json_file_path}")
                visualize_predictions(image_folder, output_folder, json_file_path)

