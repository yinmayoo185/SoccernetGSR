def calculate_giou(box1, box2):
    # Calculate GIoU between two bounding boxes in the format (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / (area_box1 + area_box2 - intersection)

    # Calculate the enclosing box
    x1_enclose = min(box1[0], box2[0])
    y1_enclose = min(box1[1], box2[1])
    x2_enclose = max(box1[2], box2[2])
    y2_enclose = max(box1[3], box2[3])
    wc = x2_enclose - x1_enclose
    hc = y2_enclose - y1_enclose
    area_enclose = wc * hc

    giou = iou - (area_enclose - intersection) / area_enclose
    giou = (giou + 1.) / 2.0  # resize from (-1,1) to (0,1)
    return giou