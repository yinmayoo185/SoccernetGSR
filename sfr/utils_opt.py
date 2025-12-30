'''utils functions, variables
'''

import random

import readline

import cv2
import numpy as np
import torch
from torch import nn

try:
    import kornia
except ModuleNotFoundError:
    pass


def confirm(question='OK to continue?'):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question + ' [y/n] ').lower()
    return answer == "y"


def print_notification(content_list, notifi_type='NOTIFICATION'):
    print(
        '---------------------- {0} ----------------------'.format(notifi_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('-------------------------- END --------------------------')


# def to_torch(np_array):
#     if constant_var.USE_CUDA:
#         tensor = torch.from_numpy(np_array).float().cuda()
#     else:
#         tensor = torch.from_numpy(np_array).float()
#     return torch.autograd.Variable(tensor, requires_grad=False)

def to_torch(np_array):
    # configs = parse_configs()
    tensor = torch.from_numpy(np_array).float()
    return torch.autograd.Variable(tensor, requires_grad=False)


def set_tensor_device(torch_var):
    if constant_var.USE_CUDA:
        return torch_var.cuda()
    else:
        return torch_var


def set_model_device(model):
    if constant_var.USE_CUDA:
        return model.cuda()
    else:
        return model


def to_numpy(cuda_var):
    return cuda_var.data.cpu().numpy()


def isnan(x):
    return x != x


def hasnan(x):
    return isnan(x).any()


def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    assert isinstance(torch_img, torch.Tensor), 'cannot process data type: {0}'.format(type(torch_img))
    if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')


def np_img_to_torch_img(np_img):
    '''convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width

    Arguments:
        np_img {[type]} -- [description]
    '''
    assert isinstance(np_img, np.ndarray), 'cannot process data type: {0}'.format(type(np_img))
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return to_torch(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return to_torch(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return to_torch(np_img)
    else:
        raise ValueError('cannot process this image')


def fix_randomness():
    random.seed(542)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(719)
    np.random.seed(121)


# the homographies map to coordinates in the range [-0.5, 0.5] (the ones in GT datasets)
BASE_RANGE = 0.5


def FULL_CANON4PTS_NP():
    return np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32)


def LOWER_CANON4PTS_NP():
    return np.array([[-0.5, 0.1], [-0.5, 0.5], [0.5, 0.5], [0.5, 0.1]], dtype=np.float32)


def get_perspective_transform(src, dst):
    '''
    kornia: https://github.com/arraiyopensource/kornia
    license: https://github.com/arraiyopensource/kornia/blob/master/LICENSE
    '''
    try:
        return kornia.get_perspective_transform(src, dst)
    except:
        r"""Calculates a perspective transform from four pairs of the corresponding
        points.
        The function calculates the matrix of a perspective transform so that:
        .. math ::
            \begin{bmatrix}
            t_{i}x_{i}^{'} \\
            t_{i}y_{i}^{'} \\
            t_{i} \\
            \end{bmatrix}
            =
            \textbf{map_matrix} \cdot
            \begin{bmatrix}
            x_{i} \\
            y_{i} \\
            1 \\
            \end{bmatrix}
        where
        .. math ::
            dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3
        Args:
            src (Tensor): coordinates of quadrangle vertices in the source image.
            dst (Tensor): coordinates of the corresponding quadrangle vertices in
                the destination image.
        Returns:
            Tensor: the perspective transformation.
        Shape:
            - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
            - Output: :math:`(B, 3, 3)`
        """
        if not torch.is_tensor(src):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(src)))
        if not torch.is_tensor(dst):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(dst)))
        # if not src.shape[-2:] == (4, 2):
        #    raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
        #                    .format(src.shape))
        if not src.shape == dst.shape:
            raise ValueError("Inputs must have the same shape. Got {}"
                             .format(dst.shape))
        if not (src.shape[0] == dst.shape[0]):
            raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                             .format(src.shape, dst.shape))

        def ax(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
                 -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
                 ], dim=1)

        def ay(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
                 -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)

        # we build matrix A by using only 4 point correspondence. The linear
        # system is solved with the least square method, so here
        # we could even pass more correspondence
        p = []
        p.append(ax(src[:, 0], dst[:, 0]))
        p.append(ay(src[:, 0], dst[:, 0]))

        p.append(ax(src[:, 1], dst[:, 1]))
        p.append(ay(src[:, 1], dst[:, 1]))

        p.append(ax(src[:, 2], dst[:, 2]))
        p.append(ay(src[:, 2], dst[:, 2]))

        p.append(ax(src[:, 3], dst[:, 3]))
        p.append(ay(src[:, 3], dst[:, 3]))

        # A is Bx8x8
        A = torch.stack(p, dim=1)

        # b is a Bx8x1
        b = torch.stack([
            dst[:, 0:1, 0], dst[:, 0:1, 1],
            dst[:, 1:2, 0], dst[:, 1:2, 1],
            dst[:, 2:3, 0], dst[:, 2:3, 1],
            dst[:, 3:4, 0], dst[:, 3:4, 1],
        ], dim=1)

        # solve the system Ax = b
        X, LU = torch.solve(b, A)

        # create variable to return
        batch_size = src.shape[0]
        M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
        M[..., :8] = torch.squeeze(X, dim=-1)
        return M.view(-1, 3, 3)  # Bx3x3


def nms(heatmap, nms_threshold=0.25):
    keypoints = []
    # heatmap = np.copy(heatmap)
    # heatmap = heatmap.clone()
    heatmap_copy = heatmap.clone()
    # kernel = np.ones((3, 3), np.uint8)
    # heatmap = cv2.dilate(heatmap, kernel, iterations=1)
    for y in range(heatmap_copy.shape[0]):
        for x in range(heatmap_copy.shape[1]):
            if heatmap_copy[y, x] > nms_threshold:
                score = heatmap[y, x]
                keypoints.append((y, x, score))
                heatmap_copy[y - 2:y + 2, x - 2:x + 2] = 0
    return keypoints


def get_filtered_keypoints(configs, pred, template_kpts):
    points, pts = [], []
    distance_threshold = 20
    device = pred.device
    filtered_keypoints = torch.tensor((), requires_grad=True, device=device)
    # filtered_keypoints = []
    pred = pred[0]
    # pred = pred[0].cpu().detach().numpy()
    # import IPython; IPython.embed()

    '''locations = []
    for i, p in enumerate(pred):
        if p.max() > 0.7:
            loc = (p == p.max()).nonzero()[0]
            loc = loc *
            locations.append(loc)
        else:
            locations.append(None)
    #import IPython; IPython.embed()
    '''
    for i in range(configs.num_keypoints):
        sample_heatmap = pred[i, ...]
        # if (sample_heatmap > 0.7).nonzero().size(0) != 0:
        # import IPython; IPython.embed()
        # post-processing steps
        # nms_keypoints = nms(sample_heatmap, 0.8)  # volleyball
        nms_keypoints = nms(sample_heatmap, 0.8)  # soccer
        # nms_keypoints = np.array(nms_keypoints)
        nms_keypoints = torch.tensor(nms_keypoints, requires_grad=True, device=device)

        if len(nms_keypoints) > 0:
            max_index = nms_keypoints[:, 2].argmax()
            x = nms_keypoints[max_index][1] * configs.org_size[0] / configs.hm_size[0]
            y = nms_keypoints[max_index][0] * configs.org_size[1] / configs.hm_size[1]
            # import IPython; IPython.embed()
            # x = np.rint(nms_keypoints[max_index][1] * configs.org_size[0] / configs.hm_size[0])#.astype(np.int32)
            # y = np.rint(nms_keypoints[max_index][0] * configs.org_size[1] / configs.hm_size[1])#.astype(np.int32)

            # keypoint = (x,y)
            keypoint = torch.tensor((x, y), requires_grad=True, device=device)
            # Compare the distance between the current keypoint and all other keypoints

            # distances = [np.linalg.norm(np.array(keypoint) - np.array(kp)) for kp in filtered_keypoints]
            distances = [torch.linalg.norm(keypoint - kp) for kp in filtered_keypoints]
            # If the distance between the current keypoint and the closest other keypoint is greater than the threshold,
            # add the current keypoint to the filtered list

            if len(filtered_keypoints) == 0 or min(distances) > distance_threshold:
                # filtered_keypoints.append(keypoint)
                filtered_keypoints = torch.cat((filtered_keypoints, keypoint))
            else:
                # filtered_keypoints.append(torch.tensor((0., 0.)).to(device=device))
                # filtered_keypoints.append((0., 0.))
                filtered_keypoints = torch.cat(
                    (filtered_keypoints, torch.tensor((0., 0.), requires_grad=True, device=device)))
        else:
            filtered_keypoints = torch.cat(
                (filtered_keypoints, torch.tensor((0., 0.), requires_grad=True, device=device)))
            # filtered_keypoints.append(torch.tensor((0., 0.)).to(device=device))
            # filtered_keypoints.append((0., 0.))

    # points = np.array(filtered_keypoints).reshape(-1, 2)
    points = filtered_keypoints.reshape(-1, 2)

    # pts_sel, template_sel = [], []
    pts_sel, template_sel = torch.tensor((), requires_grad=True, device=device), torch.tensor((), requires_grad=True,
                                                                                              device=device)

    for kp_idx, kp in enumerate(points):
        if int(kp[0]) != 0 and int(kp[1]) != 0 and (0 <= int(kp[0]) < configs.org_size[0]) and (
                0 <= int(kp[1]) < configs.org_size[1]):
            x = int(kp[0])
            y = int(kp[1])
            tem_x = int(template_kpts[kp_idx][0])
            tem_y = int(template_kpts[kp_idx][1])
            # pts_sel.append((x, y))
            # template_sel.append((tem_x, tem_y))
            pts_sel = torch.cat((pts_sel, torch.tensor((x, y)).to(device)))
            template_sel = torch.cat((template_sel, torch.tensor((x, y)).to(device)))
    # import IPython; IPython.embed()
    # return np.array(pts_sel), np.array(template_sel)
    return pts_sel.reshape(-1, 2).unsqueeze(0), template_sel.reshape(-1, 2).unsqueeze(0)


def to_template(configs, pts, temp_kpts, org_size):
    valid_x = pts[:, 0] < org_size[0]
    valid_y = pts[:, 1] < org_size[1]
    valid_kpts = torch.bitwise_and(valid_x, valid_y)
    return pts[valid_kpts].unsqueeze(0), np.expand_dims(temp_kpts[valid_kpts.cpu().detach().numpy()][:, :2], axis=0)


def to_orig(configs, input, org_size):
    input_copy = input.clone()
    input[:, 0] = input_copy[:, 1] * org_size[0] / 384
    input[:, 1] = input_copy[:, 0] * org_size[1] / 384

    # input_copy = input.clone()
    # input[:, 0] = input_copy[:, 1]
    # input[:, 1] = input_copy[:, 0]
    return input


def softargmax2d(input, beta=100):
    *_, h, w = input.shape
    device = input.device
    input = input.reshape(*_, h * w)
    input = nn.functional.softmax(beta * input, dim=-1)

    indices_c, indices_r = np.meshgrid(
        np.linspace(0, 1, w),
        np.linspace(0, 1, h),
        indexing='xy'
    )

    indices_r = torch.tensor(np.reshape(indices_r, (-1, h * w))).to(device)
    indices_c = torch.tensor(np.reshape(indices_c, (-1, h * w))).to(device)

    result_r = torch.sum((h - 1) * input * indices_r, dim=-1)
    result_c = torch.sum((w - 1) * input * indices_c, dim=-1)

    result = torch.stack([result_r, result_c], dim=-1)

    return result


def get_homography_matrix(src, dst):
    '''
    kornia: https://github.com/arraiyopensource/kornia
    license: https://github.com/arraiyopensource/kornia/blob/master/LICENSE
    '''
    try:
        return kornia.get_perspective_transform(src, dst)
    except:
        r"""Calculates a perspective transform from four pairs of the corresponding
        points.
        The function calculates the matrix of a perspective transform so that:
        .. math ::
            \begin{bmatrix}
            t_{i}x_{i}^{'} \\
            t_{i}y_{i}^{'} \\
            t_{i} \\
            \end{bmatrix}
            =
            \textbf{map_matrix} \cdot
            \begin{bmatrix}
            x_{i} \\
            y_{i} \\
            1 \\
            \end{bmatrix}
        where
        .. math ::
            dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3
        Args:
            src (Tensor): coordinates of quadrangle vertices in the source image.
            dst (Tensor): coordinates of the corresponding quadrangle vertices in
                the destination image.
        Returns:
            Tensor: the perspective transformation.
        Shape:
            - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
            - Output: :math:`(B, 3, 3)`
        """
        if not torch.is_tensor(src):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(src)))
        if not torch.is_tensor(dst):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(dst)))
        # if not src.shape[-2:] == (4, 2):
        #    raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
        #                    .format(src.shape))
        if not src.shape == dst.shape:
            raise ValueError("Inputs must have the same shape. Got {}"
                             .format(dst.shape))
        if not (src.shape[0] == dst.shape[0]):
            raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                             .format(src.shape, dst.shape))

        def ax(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
                 -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
                 ], dim=1)

        def ay(p, q):
            ones = torch.ones_like(p)[..., 0:1]
            zeros = torch.zeros_like(p)[..., 0:1]
            return torch.cat(
                [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
                 -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], dim=1)

        # we build matrix A by using only 4 point correspondence. The linear
        # system is solved with the least square method, so here
        # we could even pass more correspondence
        p = []
        p.append(ax(src[:, 0], dst[:, 0]))
        p.append(ay(src[:, 0], dst[:, 0]))

        p.append(ax(src[:, 1], dst[:, 1]))
        p.append(ay(src[:, 1], dst[:, 1]))

        p.append(ax(src[:, 2], dst[:, 2]))
        p.append(ay(src[:, 2], dst[:, 2]))

        p.append(ax(src[:, 3], dst[:, 3]))
        p.append(ay(src[:, 3], dst[:, 3]))

        # A is Bx8x8
        A = torch.stack(p, dim=1)

        # b is a Bx8x1
        b = torch.stack([
            dst[:, 0:1, 0], dst[:, 0:1, 1],
            dst[:, 1:2, 0], dst[:, 1:2, 1],
            dst[:, 2:3, 0], dst[:, 2:3, 1],
            dst[:, 3:4, 0], dst[:, 3:4, 1],
        ], dim=1)

        # solve the system Ax = b

        # X, LU = torch.linalg.solve(b, A)
        X = torch.linalg.solve(A, b)
        # create variable to return
        batch_size = src.shape[0]
        M = torch.ones(batch_size, 9, device=src.device, dtype=src.dtype)
        M[..., :8] = torch.squeeze(X, dim=-1)

        return M.view(-1, 3, 3)  # Bx3x3


def warp_image(img, H, out_shape=None, input_grid=None):
    if out_shape is None:
        out_shape = img.shape[-2:]
    if len(img.shape) < 4:
        img = img[None]
    if len(H.shape) < 3:
        H = H[None]
    assert img.shape[0] == H.shape[0], 'batch size of images do not match the batch size of homographies'
    batchsize = img.shape[0]
    # create grid for interpolation (in frame coordinates)
    if input_grid is None:
        y, x = torch.meshgrid([
            torch.linspace(-0.5, 0.5,
                           steps=out_shape[-2]),
            torch.linspace(-0.5, 0.5,
                           steps=out_shape[-1])
        ])
        x = x.to(img.device)
        y = y.to(img.device)
    else:
        x, y = input_grid
    x, y = x.flatten(), y.flatten()

    # append ones for homogeneous coordinates
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy = xy.repeat([batchsize, 1, 1])  # shape: (B, 3, N)
    # warp points to model coordinates
    xy_warped = torch.matmul(H, xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # build grid
    grid = torch.stack([
        x_warped.view(batchsize, *out_shape[-2:]),
        y_warped.view(batchsize, *out_shape[-2:])
    ],
        dim=-1)

    # sample warped image
    warped_img = torch.nn.functional.grid_sample(
        img, grid, mode='bilinear', padding_mode='zeros')

    # if utils.hasnan(warped_img):
    #    print('nan value in warped image! set to zeros')
    #    warped_img[utils.isnan(warped_img)] = 0

    return warped_img
