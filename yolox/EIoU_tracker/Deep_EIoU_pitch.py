import numpy as np
from collections import deque
from yolox.EIoU_tracker import matching
from yolox.EIoU_tracker.basetrack import BaseTrack, TrackState
from yolox.EIoU_tracker.kalman_filter import KalmanFilter
from scipy.spatial.distance import cdist

class STrack(BaseTrack):
    def __init__(self, tlwh, pitch_pos, score, feat=None, feat_history=30):
        # Original image space bbox
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        # Pitch space position [x,y]
        self.pitch_pos = np.asarray(pitch_pos, dtype=np.float64)

        # Kalman filter for pitch space tracking
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # Track state
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.New
        self.start_frame = 0
        self.frame_id = 0
        self.end_frame = 0

        # ReID feature management
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        """Update appearance features"""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """Predict next position using Kalman filter"""
        if self.mean is None or not self.is_activated:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[2:4] = 0  # Zero velocity if not tracked
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.pitch_pos = self.mean[:2]

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        # Initialize mean & covariance [x, y, vx, vy]
        self.mean = np.zeros(4)
        self.mean[:2] = self.pitch_pos
        self.covariance = np.eye(4) * 100  # High initial uncertainty

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.pitch_pos
        )
        self.pitch_pos = self.mean[:2]

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Update Kalman filter
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.pitch_pos
        )
        self.pitch_pos = self.mean[:2]

        # Update appearance
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self):
        self.state = TrackState.Lost
        self.end_frame = self.frame_id

    def mark_removed(self):
        self.state = TrackState.Removed
        self.end_frame = self.frame_id


class PitchKalmanFilter(KalmanFilter):
    def __init__(self):
        super().__init__()
        # State: [x, y, vx, vy]
        self.motion_mat = np.eye(4, 4)
        self.motion_mat[0, 2] = 1  # x += vx
        self.motion_mat[1, 3] = 1  # y += vy

        self.update_mat = np.eye(2, 4)  # Measure x,y only

        # Base noise parameters
        self.base_pos_std = 0.2  # meters
        self.base_vel_std = 0.5  # m/s
        self.min_pos_std = 0.1
        self.max_pos_std = 5.0
        self.min_vel_std = 0.1
        self.max_vel_std = 10.0
        self.eps = 1e-4  # Minimum covariance floor

        # Windows for adaptive noise
        self.innovation_window = deque(maxlen=30)
        self.residual_window = deque(maxlen=30)
        self.velocity_window = deque(maxlen=10)

    def compute_process_noise(self, state):
        """Compute smoothed state-dependent process noise"""
        # Smooth velocity estimate
        self.velocity_window.append(state[2:4])
        if len(self.velocity_window) > 1:
            velocity = np.mean([np.linalg.norm(v) for v in self.velocity_window])
        else:
            velocity = np.linalg.norm(state[2:4])

        # Scale noise with velocity and innovation
        pos_std = self.base_pos_std * (1 + 0.1 * velocity)
        vel_std = self.base_vel_std * (1 + 0.2 * velocity)

        # Apply innovation-based adjustment
        if len(self.innovation_window) > 5:
            var_innov = np.var(np.stack(self.innovation_window), axis=0)
            noise_scale = 1 + var_innov.mean() / (self.base_pos_std ** 2)
            pos_std *= noise_scale
            vel_std *= noise_scale

        # Clip noise values
        pos_std = np.clip(pos_std, self.min_pos_std, self.max_pos_std)
        vel_std = np.clip(vel_std, self.min_vel_std, self.max_vel_std)

        # Construct Q matrix
        Q = np.eye(4)
        Q[0:2, 0:2] *= pos_std**2
        Q[2:4, 2:4] *= vel_std**2

        return Q

    def compute_measurement_noise(self):
        """Compute adaptive measurement noise from residuals"""
        if len(self.residual_window) > 5:
            residuals = np.array(self.residual_window)
            R = np.cov(residuals.T, ddof=1)  # Unbiased estimate
            R += np.eye(2) * max(self.base_pos_std**2, self.eps)
        else:
            R = np.eye(2) * (self.base_pos_std**2 + self.eps)

        return R

    def predict(self, mean, covariance):
        # Compute adaptive process noise
        Q = self.compute_process_noise(mean)

        # Predict
        new_mean = self.motion_mat @ mean
        new_covariance = self.motion_mat @ covariance @ self.motion_mat.T + Q

        return new_mean, new_covariance

    def update(self, mean, covariance, measurement):
        # Innovation
        innovation = measurement - self.update_mat @ mean
        self.innovation_window.append(innovation)

        # Adaptive measurement noise
        R = self.compute_measurement_noise()

        # Kalman update
        PHt = covariance @ self.update_mat.T
        S = self.update_mat @ PHt + R
        K = PHt @ np.linalg.inv(S)

        new_mean = mean + K @ innovation
        new_covariance = (np.eye(4) - K @ self.update_mat) @ covariance

        # Store residual
        residual = measurement - self.update_mat @ new_mean
        self.residual_window.append(residual)

        return new_mean, new_covariance


class Deep_EIoU(object):
    def __init__(self, cfg, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0

        # Configuration
        self.track_high_thresh = cfg['TRACK_HIGH_THRESH']
        self.track_low_thresh = cfg['TRACK_LOW_THRESH']
        self.new_track_thresh = cfg['NEW_TRACK_THRESH']
        self.match_thresh = cfg['MATCH_THRESH']
        self.max_dist = cfg['MAX_PITCH_DIST']
        self.with_reid = cfg['WITH_REID']
        self.weight_dist = 0.7
        self.weight_reid = 0.3
        self.max_removed_history = 100

        self.max_time_lost = int(frame_rate / 30.0 * cfg['TRACK_BUFFER'])
        self.kalman_filter = PitchKalmanFilter()

    def _get_distance_cost(self, tracks, detections):
        """Calculate distance matrix in pitch space using vectorized operations"""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))

        track_pos = np.stack([t.pitch_pos for t in tracks])
        det_pos = np.stack([d.pitch_pos for d in detections])
        D = cdist(track_pos, det_pos)
        return np.minimum(1.0, D / self.max_dist)

    def update(self, output_results, embedding, pitch_positions):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Create detections
        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]

            # Filter detections
            remain_inds = scores > self.track_high_thresh
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    pitch_pos,
                    s,
                    f if self.with_reid else None
                )
                for tlbr, pitch_pos, s, f in zip(
                    bboxes[remain_inds],
                    pitch_positions[remain_inds],
                    scores[remain_inds],
                    embedding[remain_inds] if self.with_reid else [None] * remain_inds.sum()
                )
            ]
        else:
            detections = []

        # Get tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Predict locations
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for track in strack_pool:
            track.predict()

        # Calculate matching cost matrix
        dists = self._get_distance_cost(strack_pool, detections)
        if self.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections)
            dist_mask = (dists > self.max_dist)
            emb_dists[dist_mask] = 1.0
            dists = self.weight_dist * dists + self.weight_reid * emb_dists

        # First association
        matches, u_tracks, u_detections = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Lost tracks
        for it in u_tracks:
            track = strack_pool[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # New tracks
        for idet in u_detections:
            det = detections[idet]
            if det.score < self.new_track_thresh:
                continue
            det.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(det)

        # Handle unconfirmed tracks
        dists = self._get_distance_cost(unconfirmed, detections)
        if self.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections)
            dist_mask = (dists > self.max_dist)
            emb_dists[dist_mask] = 1.0
            dists = self.weight_dist * dists + self.weight_reid * emb_dists

        matches, u_unconfirmed, u_detections = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Remove lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update state lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Output
        return [track for track in self.tracked_stracks if track.is_activated]


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