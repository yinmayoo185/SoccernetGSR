import numpy as np
from collections import deque

from yolox.EIoU_tracker import matching
from yolox.EIoU_tracker.basetrack import BaseTrack, TrackState
from yolox.EIoU_tracker.KF import KalmanFilter_bbox, KalmanFilter_pitch

from collections import defaultdict


class STrack(BaseTrack):
    shared_kalman_bbox = KalmanFilter_bbox()
    shared_kalman_pitch = KalmanFilter_pitch()

    def __init__(self, tlwh, pitch_pos, score, feat=None, feat_history=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self._pitch_pos = np.asarray(pitch_pos, dtype=np.float64)
        self.kalman_filter = None
        self.mean_bbox, self.covariance_bbox = None, None
        self.mean_pitch, self.covariance_pitch = None, None
        self.is_activated = False

        self.last_tlwh = self._tlwh
        self.last_pitch_pos = self._pitch_pos

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.features = []
        self.times = []
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state_bbox = self.mean_bbox.copy()
        if self.state != TrackState.Tracked:
            mean_state_bbox[6] = 0
            mean_state_bbox[7] = 0
        mean_state_pitch = self.mean_pitch.copy()

        self.mean_bbox, self.covariance_bbox = self.kalman_filter_bbox.predict(mean_state_bbox, self.covariance_bbox)
        self.mean_pitch, self.covariance_pitch = self.kalman_filter_pitch.predict(mean_state_pitch, self.covariance_pitch)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean_bbox = np.asarray([st.mean_bbox.copy() for st in stracks])
            multi_covariance_bbox = np.asarray([st.covariance_bbox for st in stracks])
            multi_mean_pitch = np.asarray([st.mean_pitch.copy() for st in stracks])
            multi_covariance_pitch = np.asarray([st.covariance_pitch for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean_bbox[i][6] = 0
                    multi_mean_bbox[i][7] = 0
            multi_mean_bbox, multi_covariance_bbox = STrack.shared_kalman_bbox.multi_predict(multi_mean_bbox, multi_covariance_bbox)
            multi_mean_pitch, multi_covariance_pitch = STrack.shared_kalman_pitch.multi_predict(multi_mean_pitch, multi_covariance_pitch)
            for i, (mean_bbox, cov_bbox, mean_pitch, cov_pitch) in enumerate(zip(multi_mean_bbox, multi_covariance_bbox, multi_mean_pitch, multi_covariance_pitch)):
                stracks[i].mean_bbox = mean_bbox
                stracks[i].covariance_bbox = cov_bbox
                stracks[i].mean_pitch = mean_pitch
                stracks[i].covariance_pitch = cov_pitch

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean_bbox.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance_bbox for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter_bbox, kalman_filter_pitch, frame_id):
        """Start a new tracklet"""
        self.kalman_filter_bbox = kalman_filter_bbox
        self.kalman_filter_pitch = kalman_filter_pitch
        self.track_id = self.next_id()
        self.mean_bbox, self.covariance_bbox = self.kalman_filter_bbox.initiate(self.tlwh_to_xywh(self._tlwh))
        self.mean_pitch, self.covariance_pitch = self.kalman_filter_pitch.initiate(self._pitch_pos)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean_bbox, self.covariance_bbox = self.kalman_filter_bbox.update(self.mean_bbox, self.covariance_bbox,
                                                               self.tlwh_to_xywh(new_track.tlwh))
        self.mean_pitch, self.covariance_pitch = self.kalman_filter_pitch.update(self.mean_pitch, self.covariance_pitch,
                                                                              self._pitch_pos)

        self.last_tlwh = new_track.tlwh
        self.last_pitch_pos = new_track._pitch_pos

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
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
        new_pitch = new_track.xy

        self.last_tlwh = new_tlwh
        self.last_pitch_pos = new_pitch

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
            self.features.append(new_track.curr_feat)
            self.times.append(frame_id)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def xy(self):
        """Current estimated pitch position."""
        if self.mean_pitch is None:
            return self._pitch_pos.copy()
        return self.mean_pitch[:2].copy()

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean_bbox is None:
            return self._tlwh.copy()
        ret = self.mean_bbox[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def last_tlbr(self):
        ret = self.last_tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Deep_EIoU(object):
    def __init__(self, cfg, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()
        self.court_width = 105
        self.court_height = 68
        self.max_pitch_gap = 10.0  # pixels
        self.court_diag = np.hypot(self.court_width, self.court_height)

        self.frame_id = 0

        self.track_high_thresh = cfg['TRACK_HIGH_THRESH']
        self.track_low_thresh = cfg['TRACK_LOW_THRESH']
        self.new_track_thresh = cfg['NEW_TRACK_THRESH']
        self.match_thresh = cfg['MATCH_THRESH']
        self.with_reid = cfg['WITH_REID']


        self.buffer_size = int(frame_rate / 30.0 * cfg['TRACK_BUFFER'])
        self.max_time_lost = self.buffer_size
        self.kalman_filter_bbox = KalmanFilter_bbox()
        self.kalman_filter_pitch = KalmanFilter_pitch()

        # ReID module
        self.proximity_thresh = cfg['PROXIMITY_THRESH']
        self.appearance_thresh = cfg['APPEARANCE_THRESH']

    def get_pitch_distance(self, tracks, detections):
        """Compute distance between tracks and detections in pitch space"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))

        track_pts = np.array([t.xy for t in tracks])
        det_pts = np.array([d._pitch_pos for d in detections])
        return np.linalg.norm(track_pts[:, None] - det_pts, axis=2)

    def update(self, output_results, embedding, pitch_positions):

        '''
        output_results : [x1,y1,x2,y2,score] type:ndarray
        embdding : [emb1,emb2,...] dim:512
        '''

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]  # x1y1x2y2
            elif output_results.shape[1] == 7:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
            else:
                raise ValueError('Wrong detection size {}'.format(output_results.shape[1]))

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            pitches = pitch_positions[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            pitch_h = pitches[remain_inds]

            if self.with_reid:
                embedding = embedding[lowest_inds]
                features_keep = embedding[remain_inds]

        else:
            bboxes = []
            scores = []
            dets = []
            scores_keep = []
            features_keep = []
            pitch_h = []

        if len(dets) > 0:
            '''Detections'''
            if self.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), p, s, f) for
                              (tlbr, s, f, p) in zip(dets, scores_keep, features_keep, pitch_h)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), p, s) for
                              (tlbr, s, p) in zip(dets, scores_keep, pitch_h)]
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

        # Associate with high score detection boxes
        num_iteration = 2
        init_expand_scale = 0.7
        expand_scale_step = 0.1

        for iteration in range(num_iteration):

            cur_expand_scale = init_expand_scale + expand_scale_step * iteration

            ious_dists = matching.eiou_distance(strack_pool, detections, cur_expand_scale)
            ious_dists_mask = (ious_dists > self.proximity_thresh)
            pitch_dists = self.get_pitch_distance(strack_pool, detections)
            pitch_cost = np.clip(pitch_dists / self.court_diag, 0.0, 1.0)

            if self.with_reid:
                # if self.frame_id <= 10:
                #     emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                # else:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists, pitch_cost)
            else:
                # dists = ious_dists
                dists = np.minimum(ious_dists, pitch_cost)

            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            strack_pool = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            detections = [detections[i] for i in u_detection]

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            pitch_second = pitches[inds_second]
            scores_second = scores[inds_second]
            if self.with_reid:
                features_second = embedding[inds_second]
        else:
            dets_second = []
            scores_second = []
            features_second = []
            pitch_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if self.with_reid:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), p, s, f) for
                                     (tlbr, s, f, p) in zip(dets_second, scores_second, features_second, pitch_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), p, s) for
                                     (tlbr, s, p) in zip(dets_second, scores_second, pitch_second)]
        else:
            detections_second = []

        r_tracked_stracks = strack_pool
        ious_dists = matching.eiou_distance(r_tracked_stracks, detections_second, expand=0.5)
        pitch_dists = self.get_pitch_distance(r_tracked_stracks, detections_second)
        pitch_cost = np.clip(pitch_dists / (self.court_diag * 1.5), 0.0, 1.0)
        dists = np.minimum(ious_dists, pitch_cost)
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
        ious_dists = matching.eiou_distance(unconfirmed, detections, 0.5)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        pitch_dists = self.get_pitch_distance(unconfirmed, detections)
        pitch_cost = np.clip(pitch_dists / (self.court_diag * 1.5), 0.0, 1.0)

        if self.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists, pitch_cost)
        else:
            dists = np.minimum(ious_dists, pitch_cost)

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
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter_bbox, self.kalman_filter_pitch, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]

        # print("self.frameId : ", self.frame_id)

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
