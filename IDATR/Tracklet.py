class Tracklet:
    def __init__(self, track_id=None, frames=None, scores=None, bboxes=None,
                 role=None, jn=None, jc=None, team=None, feats=None):
        '''
        Initialize the Tracklet with IDs, times, scores, bounding boxes, and optional features.
        If parameters are not provided, initializes them to None or empty lists.

        Args:
            track_id (int, optional): Unique identifier for the track. Defaults to None.
            frames (list or int, optional): Frame numbers where the track is present. Can be a list of frames or a single frame. Defaults to None.
            scores (list or float, optional): Detection scores corresponding to frames. Can be a list of scores or a single score. Defaults to None.
            bboxes (list of lists or list, optional): Bounding boxes corresponding to each frame. Each bounding box is a list of 4 elements. Defaults to None.
            feats (list of np.array, optional): Feature vectors corresponding to frames. Each feature should be a numpy array of shape (512,). Defaults to None.
        '''
        self.track_id = track_id
        self.parent_id = track_id
        self.scores = scores if isinstance(scores, list) else [scores] if scores is not None else []
        self.times = frames if isinstance(frames, list) else [frames] if frames is not None else []
        self.bboxes = bboxes if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list) else [bboxes] if bboxes is not None else []
        self.features = feats if feats is not None else []
        self.role = role if isinstance(role, list) else [role] if role is not None else []
        self.jn = jn if isinstance(jn, list) else [jn] if jn is not None else []
        self.jc = jc if isinstance(jc, list) else [jc] if jc is not None else []
        self.team = team if isinstance(team, list) else [team] if team is not None else []

    def append_det(self, frame, score, bbox, role, jn, jc, team):
        '''
        Appends a detection to the tracklet.

        Args:
            frame (int): Frame number for the detection.
            score (float): Detection score.
            bbox (list of float): Bounding box with four elements [x, y, width, height].
        '''
        self.scores.append(score)
        self.times.append(frame)
        self.bboxes.append(bbox)
        self.role.append(role)
        self.jn.append(jn)
        self.jc.append(jc)
        self.team.append(team)

    def append_feat(self, feat):
        '''
        Appends a feature vector to the tracklet.

        Args:
            feat (np.array): Feature vector of shape (512,).
        '''
        self.features.append(feat)

    def extract(self, start, end):
        '''
        Extracts a subtrack from the tracklet between two indices.

        Args:
            start (int): Start index for the extraction.
            end (int): End index for the extraction.

        Returns:
            Tracklet: A new Tracklet object that is a subset of the original from start to end indices.
        '''
        subtrack = Tracklet(self.track_id, self.times[start:end + 1], self.scores[start:end + 1], self.bboxes[start:end + 1], self.features[start:end + 1] if self.features else None)
        return subtrack

