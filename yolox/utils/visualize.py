#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis", "plot_tracking", "template_plot_tracking"]

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def template_plot_tracking(template, template_size, tlwhs, obj_ids, sports, teams, scores=None, frame_id=0, fps=0.,
                           ids2=None):
    text_scale = 6
    text_thickness = 4
    color = (255, 0, 0)
    image = np.ascontiguousarray(np.copy(template))

    # Set colors for each team
    team_colors = [(184, 133, 27), (250, 248, 246)]
    # heatmap = np.zeros((int(template_size[1]), int(template_size[0])), dtype=np.float32)
    # colormap = cv2.COLORMAP_JET
    # heatmap_alpha = 0.5

    # # Calculate moving averages for x and y coordinates separately
    # smoothed_x = moving_average(tlwhs[:, 0], 3)
    # smoothed_y = moving_average(tlwhs[:, 1], 3)

    team_lines = {}
    for i, tlwh in enumerate(tlwhs):
        x, y = tlwh
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        if 0 <= int(x) <= template_size[0] and 0 <= int(y) <= template_size[1]:
            if int(teams[i]) == 0:
                color = team_colors[int(teams[i])]
            elif int(teams[i]) == 1:
                color = team_colors[int(teams[i])]

            # heatmap[int(y), int(x)] += 1

            image = cv2.circle(image, (int(x), int(y)), 8, color, -1)
            # image = cv2.putText(image, id_text, (int(x) + 6, int(y) + 6), cv2.FONT_HERSHEY_PLAIN, text_scale, color,
            #                     thickness=text_thickness, lineType=cv2.LINE_AA)

            # Initialize the team line if it doesn't exist
            if teams[i] not in team_lines:
                team_lines[teams[i]] = []

            # Add the current player's point to the team line
            team_lines[teams[i]].append((int(x), int(y)))

            # Connect players of the same team with lines
            for j, (cx, cy) in enumerate(tlwhs):
                if teams[i] == teams[j] and i != j:  # Same team and different players
                    # if (0 <= int(x) <= template_size[0] and 0 <= int(y) <= template_size[1] and
                    #         0 <= int(cx) <= template_size[0] and 0 <= int(cy) <= template_size[1]):
                    cv2.line(image, (int(x), int(y)), (int(cx), int(cy)), color, 3)

    # # Draw lines for each team
    # for team, line_points in team_lines.items():
    #     line_points = np.array(line_points)
    #     cv2.polylines(image, [line_points], isClosed=False, color=team_colors[team], thickness=3)

    # plt.imshow(image)
    # plt.show()

    # Normalize, apply colormap to heatmap and combine with original image
    # heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    # heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), colormap)
    #
    # im0_with_heatmap = cv2.addWeighted(image, 1 - heatmap_alpha, heatmap_colored,  heatmap_alpha, 0)

    return image


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def plot_tracking(image, tlwhs, obj_ids, teams, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    # Set colors for each team
    team_colors = [(123, 119, 240), (245, 72, 72)]
    # Transparency value

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        x_center = x1 + (((x1 + w) - x1) // 2)
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        if int(teams[i]) == 0:
            color = team_colors[int(teams[i])]
        elif int(teams[i]) == 1:
            color = team_colors[int(teams[i])]

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.circle(im, (int(x_center), int(y1 + h)), 8, color, -1)
        cv2.putText(im, id_text, (int(x_center) + 3, int(y1 + h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                    (255, 255, 255),
                    thickness=text_thickness)

        # cv2.ellipse(
        #     im,
        #     # center=(int(x_center), int(y1 + h)),
        #     center=(int(x1 + w / 2), int(y1 + h)),
        #     axes=(int(70), int(0.35 * 70)),
        #     angle=0.0,
        #     startAngle=-45,
        #     endAngle=235,
        #     color=color,
        #     thickness=3,
        #     lineType=cv2.LINE_4
        # )

    # plt.imshow(im)
    # plt.show()

    return im


def plot_tracking_sportsmot(image, tlwhs, obj_ids, teams, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    # Set colors for each team
    # team_colors = [(123, 119, 240), (245, 72, 72)]
    # Transparency value

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        x_center = x1 + (((x1 + w) - x1) // 2)
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))

        # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.circle(im, (int(x_center), int(y1 + h)), 8, color, -1)
        cv2.putText(im, id_text, (int(x_center) + 3, int(y1 + h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                    (255, 255, 255),
                    thickness=text_thickness)

        cv2.ellipse(
            im,
            # center=(int(x_center), int(y1 + h)),
            center=(int(x1 + w / 2), int(y1 + h)),
            axes=(int(70), int(0.35 * 70)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,
            lineType=cv2.LINE_4
        )

    # plt.imshow(im)
    # plt.show()

    return im


def plot_tracking_with_template(image, template_img, tlwhs, obj_ids, court_positions, sports, teams, scores=None,
                                frame_id=0,
                                fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    # Number of clusters (teams)
    # Set colors for each team
    # team_colors = [(123, 119, 240), (245, 72, 72)]
    team_colors = [(0, 0, 0), (245, 72, 72)]
    # heatmap = np.zeros((int(im_h), int(im_w)), dtype=np.float32)
    # colormap = cv2.COLORMAP_HSV
    # heatmap_alpha = 0.5

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    # Calculate the size and position for the template region
    template_h, template_w = template_img.shape[:2]
    template_region_size = (im_w // 4, im_h // 4)

    template_position = (im_w - im_w // 8 - template_region_size[1], 3)

    # Resize the template image to fit the template region
    template_resized = cv2.resize(template_img, template_region_size)

    # Ensure the template region size matches the resized template image
    template_region_size = template_resized.shape[:2]

    # Check if the assignment region exceeds the dimensions of im_with_template
    if (template_position[0] + template_region_size[0] <= im_w and
            template_position[1] + template_region_size[1] <= im_h):

        # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            x_center = x1 + (((x1 + w) - x1) // 2)
            obj_id = int(obj_ids[i])
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            color = get_color(abs(obj_id))  # Change to your desired color
            cv2.putText(im, id_text, (int(x_center) + 3, int(y1 + h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (255, 255, 255),
                        thickness=text_thickness)

            if int(teams[i]) == 0:
                color = team_colors[int(teams[i])]
            elif int(teams[i]) == 1:
                color = team_colors[int(teams[i])]

            cv2.circle(im, (int(x_center), int(y1 + h)), 5, color, -1)
            cv2.ellipse(
                im,
                # center=(int(x_center), int(y1 + h)),
                center=(int(x1 + w / 2), int(y1 + h)),
                axes=(int(70), int(0.35 * 70)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=3,
                lineType=cv2.LINE_4
            )
            # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)

        # for finding two closet players
        min_distance = float('inf')
        closest_players = None
        # Find the two closest players
        for i in range(len(court_positions)):
            for j in range(i + 1, len(court_positions)):
                dist = calculate_distance(court_positions[i], court_positions[j])
                if dist < min_distance:
                    min_distance = dist
                    closest_players = (i, j)

        # Check if closest_players is not None before subscripting
        if closest_players is not None:
            # print("closest players : ", closest_players)
            player_0 = tlwhs[closest_players[0]]
            player_0_center = player_0[0] + (((player_0[0] + player_0[2]) - player_0[0]) // 2)
            # print("closest player_0 : ", player_0)
            player_1 = tlwhs[closest_players[1]]
            player_1_center = player_1[0] + (((player_1[0] + player_1[2]) - player_1[0]) // 2)
            # print("closest player_1 : ", player_1)

            # Draw a line between the two closest players
            cv2.line(im, (int(player_0_center), int(player_0[1] + player_0[3])), (int(player_1_center),
                                                                                  int(player_1[1] + player_1[3])),
                     (80, 200, 120), 2)
        else:
            pass
            # Handle the case where no players are found
            # print("No players found.")

        # Copy the original image to avoid modifying it
        im_with_template = np.copy(im)

        # Paste the resized template image onto the top-right corner
        im_with_template[template_position[1]:template_position[1] + template_region_size[0],
        template_position[0]:template_position[0] + template_region_size[1]] = template_resized

        # # Normalize, apply colormap to heatmap and combine with original image
        # heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        # heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), colormap)
        #
        # im0_with_heatmap = cv2.addWeighted(im_with_template, 1 - heatmap_alpha, heatmap_colored,  heatmap_alpha, 0)

        return im_with_template
    else:
        # cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        #             (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            x_center = x1 + (((x1 + w) - x1) // 2)
            # intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = '{}'.format(int(obj_id))
            if ids2 is not None:
                id_text = id_text + ', {}'.format(int(ids2[i]))
            color = get_color(abs(obj_id))

            if int(teams[i]) == 0:
                color = team_colors[int(teams[i])]
            elif int(teams[i]) == 1:
                color = team_colors[int(teams[i])]

            # cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.circle(im, (int(x_center), int(y1 + h)), 8, color, -1)
            cv2.putText(im, id_text, (int(x_center) + 3, int(y1 + h)), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (255, 255, 255),
                        thickness=text_thickness)
            cv2.ellipse(
                im,
                # center=(int(x_center), int(y1 + h)),
                center=(int(x1 + w / 2), int(y1 + h)),
                axes=(int(70), int(0.35 * 70)),
                angle=0.0,
                startAngle=-45,
                endAngle=235,
                color=color,
                thickness=3,
                lineType=cv2.LINE_4
            )

        return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
