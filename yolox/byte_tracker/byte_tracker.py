import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .association import calculate_giou
from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
import random
from sklearn.metrics.pairwise import cosine_similarity
import cv2


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, reid_model, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.iou_threshold = 0.25
        self.args = args
        # self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.device = torch.device('cuda')
        self.reid_model = reid_model

    def update(self, output_results, img_info, img_size, img_folder, data, im_transforms):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        remain_inds = scores > self.args.track_thresh
        # print("remain_inds : ", remain_inds)
        inds_low = scores > 0.1
        # print("inds_low : ", inds_low)
        inds_high = scores < self.args.track_thresh
        # print("inds_high : ", inds_high)

        inds_second = np.logical_and(inds_low, inds_high)
        # print("inds_second : ", inds_second)
        dets_second = bboxes[inds_second]
        # print("dets_second : ", dets_second)
        dets = bboxes[remain_inds]
        # print("dets : ", dets)
        scores_keep = scores[remain_inds]
        # print("scores_keep : ", scores_keep)
        scores_second = scores[inds_second]
        # print("scores_second : ", scores_second)
        # print("STracks : ",  self.tracked_stracks)
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]

            # print("detections : ", detections)
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        if self.frame_id > 30:
            print("frame_id : ", self.frame_id)
            for i in range(len(output_stracks)):
                tlwh_i = output_stracks[i].tlwh
                tid_i = output_stracks[i].track_id

                x, y, w, h = tlwh_i[0], tlwh_i[1], tlwh_i[2], tlwh_i[3]
                bbox_i = (x, y, x + w, y + h)
                for j in range(i + 1, len(output_stracks)):
                    tlwh_j = output_stracks[j].tlwh
                    tid_j = output_stracks[j].track_id

                    x_j, y_j, w_j, h_j = tlwh_j[0], tlwh_j[1], tlwh_j[2], tlwh_j[3]
                    bbox_j = (x_j, y_j, x_j + w_j, y_j + h_j)

                    giou = calculate_giou(bbox_i, bbox_j)
                    if giou >= self.iou_threshold:  # 0.25
                        print(giou, " giou of : ", tid_i, tid_j)
                        first_id = tid_i
                        second_id = tid_j

                        first_id_data = [item for item in data if
                                         item['frame_id'] < self.frame_id - 1 and item['track_id'] == first_id]
                        second_id_data = [item for item in data if
                                          item['frame_id'] < self.frame_id - 1 and item['track_id'] == second_id]

                        if self.frame_id > 300:
                            # Determine the range of frames for the first_id_gallery
                            start_index = max(0, self.frame_id - 1 - len(first_id_data) // 2)
                            end_index = min(len(first_id_data), self.frame_id - 1 + len(first_id_data) // 2)
                            first_id_data = first_id_data[start_index:end_index]

                            start_index = max(0, self.frame_id - 1 - len(second_id_data) // 2)
                            end_index = min(len(second_id_data), self.frame_id - 1 + len(second_id_data) // 2)
                            second_id_data = second_id_data[start_index:end_index]

                        # get from last row
                        # first_id_gallery = first_id_data[-20 if len(first_id_data) >= 20 else len(first_id_data):]
                        # second_id_gallery = second_id_data[-20 if len(second_id_data) >= 20 else len(second_id_data):]

                        # get from first row
                        # first_id_gallery = first_id_data[:20]
                        # second_id_gallery = second_id_data[:20]

                        first_id_gallery = random.sample(first_id_data, min(15, len(first_id_data)))
                        first_id_gallery = np.sort(first_id_gallery, axis=0)
                        second_id_gallery = random.sample(second_id_data, min(15, len(second_id_data)))
                        second_id_gallery = np.sort(second_id_gallery, axis=0)

                        if len(first_id_gallery) < 15 or len(second_id_gallery) < 15:
                            print("first_id_gallery : {}, second_id_gallery : {}".format(len(first_id_gallery),
                                                                                         len(second_id_gallery)))
                            continue

                        first_query = (self.frame_id - 1, output_stracks[i].track_id, output_stracks[i].tlwh[0],
                                       output_stracks[i].tlwh[1], output_stracks[i].tlwh[2], output_stracks[i].tlwh[3],
                                       output_stracks[i].score)
                        # print("first_query : ", first_query)
                        second_query = (self.frame_id - 1, output_stracks[j].track_id, output_stracks[j].tlwh[0],
                                        output_stracks[j].tlwh[1], output_stracks[j].tlwh[2], output_stracks[j].tlwh[3],
                                        output_stracks[j].score)
                        # print("second_query : ", second_query)

                        images = []
                        # first_id_query, second_id_query, first_id_gallery, second_id_gallery in order

                        # first_id_query
                        image_filename = os.path.join(img_folder,
                                                      f"{first_query[0]:05d}.jpg")  # Construct the image filename
                        frame_image = cv2.imread(image_filename)
                        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                        x, y, w, h = (
                            int(float(first_query[2])), int(float(first_query[3])), int(float(first_query[4])),
                            int(float(first_query[5])))
                        x = max(0, x)
                        Y = max(0, y)
                        if frame_image is not None:
                            first_id_query = frame_image[y:y + h, x:x + w]
                            first_id_query = im_transforms(image=first_id_query)['image']
                            images.append(first_id_query)
                        else:
                            continue

                        # second_id_query
                        image_filename = os.path.join(img_folder,
                                                      f"{second_query[0]:05d}.jpg")  # Construct the image filename
                        frame_image = cv2.imread(image_filename)
                        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                        x, y, w, h = (
                            int(float(second_query[2])), int(float(second_query[3])), int(float(second_query[4])),
                            int(float(second_query[5])))
                        x = max(0, x)
                        Y = max(0, y)
                        if frame_image is not None:
                            second_id_query = frame_image[y:y + h, x:x + w]
                            second_id_query = im_transforms(image=second_id_query)['image']
                            images.append(second_id_query)
                        else:
                            continue

                        # first_id_gallery
                        for frame_info in first_id_gallery:
                            frame_idx = frame_info[0]  # Assuming frame_id is in the first column
                            image_filename = os.path.join(img_folder,
                                                          f"{frame_idx:05d}.jpg")  # Construct the image filename
                            frame_image = cv2.imread(image_filename)
                            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                            x, y, w, h = (
                                int(float(frame_info[2])), int(float(frame_info[3])), int(float(frame_info[4])),
                                int(float(frame_info[5])))
                            x = max(0, x)
                            Y = max(0, y)
                            bbox = (x, y, x + w, y + h)
                            if frame_image is not None:
                                im_crop = frame_image[y:y + h, x:x + w]
                                if im_crop is None:
                                    continue
                                im_crop = im_transforms(image=im_crop)['image']
                                images.append(im_crop)
                            else:
                                continue

                        # second_id_gallery
                        for frame_info in second_id_gallery:
                            frame_idx = frame_info[0]  # Assuming frame_id is in the first column
                            image_filename = os.path.join(img_folder,
                                                          f"{frame_idx:05d}.jpg")  # Construct the image filename
                            frame_image = cv2.imread(image_filename)
                            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                            x, y, w, h = (
                                int(float(frame_info[2])), int(float(frame_info[3])), int(float(frame_info[4])),
                                int(float(frame_info[5])))
                            x = max(0, x)
                            Y = max(0, y)
                            bbox = (x, y, x + w, y + h)
                            if frame_image is not None:
                                im_crop = frame_image[y:y + h, x:x + w]
                                if im_crop is None:
                                    continue
                                im_crop = im_transforms(image=im_crop)["image"]
                                images.append(im_crop)
                            else:
                                continue

                        images = torch.stack(images, dim=0)
                        # print("images : ", images.shape)

                        images = images.to(self.device, non_blocking=True)
                        features = self.reid_model(images)  # input to CLIP-ReID Model
                        features = F.normalize(features, p=2, dim=1).detach().cpu().numpy()

                        # features output from CLIP-ReID Model
                        first_query_feature = features[0, :].reshape(1, -1)  # Shape: (1, 1, 768)
                        # print("first_query_feature : ", first_query_feature.shape)
                        second_query_feature = features[1, :].reshape(1, -1)
                        # print("second_query_feature : ", second_query_feature.shape)
                        first_gallery_features = features[2:17, :]  # Shape: (15, 1, 768)
                        first_gallery_features = np.mean(first_gallery_features, axis=0).reshape(1, -1)
                        # print("first_gallery_feature : ", first_gallery_features.shape)
                        second_gallery_features = features[17:33, :]  # Shape: (15, 1, 768)
                        second_gallery_features = np.mean(second_gallery_features, axis=0).reshape(1, -1)
                        # print("second_gallery_feature : ", second_gallery_features.shape)

                        # # Calculate cosine similarities
                        similarities_first_query = [
                            cosine_similarity(first_query_feature, first_gallery_features)[0][0],
                            cosine_similarity(first_query_feature, second_gallery_features)[0][0]]
                        # print("similarities_first_query :", similarities_first_query)

                        similarities_second_query = [
                            cosine_similarity(second_query_feature, first_gallery_features)[0][0],
                            cosine_similarity(second_query_feature, second_gallery_features)[0][0]]
                        # print("similarities_second_query :", similarities_second_query)

                        # Determine the correct gallery for each query
                        correct_gallery_query1 = first_id if similarities_first_query[0] > similarities_first_query[
                            1] else second_id
                        correct_gallery_query2 = second_id if similarities_second_query[1] > similarities_second_query[
                            0] else first_id

                        if (correct_gallery_query1 == second_id) and (correct_gallery_query2 == first_id):
                            print("ID switch_occured.", second_id, first_id)
                            for out_track in output_stracks:
                                if int(out_track.track_id) == first_id:
                                    print("track_id before switched first_id: ", out_track.track_id)
                                    # swap_id = first_id
                                    out_track.track_id = second_id
                                    print("track_id after switched first_id: ", out_track.track_id)
                                elif int(out_track.track_id) == second_id:
                                    print("-------------------")
                                    print("track_id before switched second_id: ", out_track.track_id)
                                    out_track.track_id = first_id
                                    # print("first_id id_track second_id: ", first_id, out_track.track_id, second_id)
                                    print("track_id after switched second_id: ", out_track.track_id)

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
