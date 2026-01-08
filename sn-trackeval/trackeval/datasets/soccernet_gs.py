import json
import os
import numpy as np
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException


class SoccerNetGS(_BaseDataset):
    """Dataset class for the SoccerNet Challenge Game State (GS) task"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/SoccerNetGS'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/SoccerNetGS/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'SPLIT_TO_EVAL': 'valid',  # Valid: 'train', 'val', 'test', 'challenge'
            'EVAL_MODE': 'distance',  # Valid: 'distance' or 'classes': both are equivalent, classes is much slower.
            'EVAL_SPACE': 'pitch',  # Valid: 'image', 'pitch'
            'EVAL_SIMILARITY_METRIC': 'gaussian',  # Valid: 'iou', 'eucl', 'gaussian'
            'EVAL_DIST_TOL': 5,
            # Distance tolerance for matching predictions P and ground truth G, in meters. P and G with a larger distance will never be considered as matched when computing the HOTA.
            'USE_ROLES': True,  # Take role into account for evaluation
            'USE_TEAMS': True,  # Take team into account for evaluation
            'USE_JERSEY_NUMBERS': True,  # Take jersey numbers into account for evaluation
            'IGNORE_BALL': True,  # Ignore ball for evaluation, currently ball evaluation is not supported
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/Labels-GameState.json',  # '{gt_folder}/{seq}/gt/gt.json'
            'SKIP_SPLIT_FOL': False,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
            # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
            # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()

        print("Initializing the dataset class for the SoccerNet Game State Reconstruction task.\n"
              "IMPORTANT: The official evaluation metric for the task, i.e. the 'GS-HOTA' will appear under the 'HOTA' name in the evaluation script output.\n"
              "This happen because GS-HOTA mainly uses the same logic as the HOTA metric, the HOTA evaluation class is therefore not forked but re-used.\n"
              "The key practical difference between the GS-HOTA and the HOTA is actually the similarity metric used to match predictions with ground truth."
              "Since this similarity score is computed outside the HOTA class (i.e. inside the SoccerNetGS dataset class), there was no need to fork it into a GS-HOTA class.\n"
              "Please refer to the official paper for more information."
              )

        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = 'SoccerNetGS'
        self.split = self.config['SPLIT_TO_EVAL']
        if not self.config['SKIP_SPLIT_FOL']:
            gt_split_fol = self.split
            track_split_fol = self.benchmark + '-' + self.split
        else:
            gt_split_fol = ''
            track_split_fol = ''
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], gt_split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], track_split_fol)
        self.seq_list = self._get_seq_info()
        self.eval_mode = self.config['EVAL_MODE']
        self.eval_space = self.config['EVAL_SPACE']
        self.eval_sim_metric = self.config['EVAL_SIMILARITY_METRIC']
        self.eval_sigma = calculate_sigma(self.config['EVAL_DIST_TOL'])
        print(
            f"Using a sigma of {self.eval_sigma} for the gaussian similarity metric, based on a distance tolerance of {self.config['EVAL_DIST_TOL']} meters.")
        self.ignore_ball = self.config['IGNORE_BALL']
        self.use_jersey_numbers = self.config['USE_JERSEY_NUMBERS']
        self.use_teams = self.config['USE_TEAMS']
        self.use_roles = self.config['USE_ROLES']
        self.all_similarity_scores = []
        self.all_classes = {}
        if self.eval_mode == 'classes':
            self.all_classes = self.extract_all_classes(self.config, self.gt_fol, self.seq_list)
            self.class_name_to_class_id = {clazz["name"]: clazz["id"] for clazz in self.all_classes.values()}
            assert 0 not in self.class_name_to_class_id
            self.class_name_to_class_id["distractor"] = 0
        else:
            self.class_name_to_class_id = {
                "person": 1,
            }
        self.should_classes_combine = True
        self.use_super_categories = False  # TODO
        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']
        self.class_counter = 1

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.valid_classes = self.class_name_to_class_id.keys()
        self.class_list = self.class_name_to_class_id.keys()
        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.json')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        if self.config["SEQ_INFO"]:
            seq_list = list(self.config["SEQ_INFO"].keys())
            seq_lengths = self.config["SEQ_INFO"]
        else:
            if self.config["SEQMAP_FILE"] is None and self.config["SEQMAP_FOLDER"] is None:
                seqmap_folder = os.path.join(self.config['GT_FOLDER'], self.config['SPLIT_TO_EVAL'])
                seq_list = [seq for seq in os.listdir(seqmap_folder) if os.path.isdir(os.path.join(seqmap_folder, seq))]
            else:
                if self.config["SEQMAP_FILE"]:
                    seqmap_file = self.config["SEQMAP_FILE"][0] if isinstance(self.config["SEQMAP_FILE"], list) else \
                    self.config["SEQMAP_FILE"]
                else:
                    seqmap_file = os.path.join(self.config["SEQMAP_FOLDER"], self.config['SPLIT_TO_EVAL'],
                                               'seq_info.json')

                if not os.path.isfile(seqmap_file):
                    print('no seqmap found: ' + seqmap_file)
                    raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
                with open(seqmap_file, 'r') as f:
                    data = json.load(f)

                seq_list = [seq["name"] for seq in data]
                seq_lengths = {seq["name"]: seq["nframes"] for seq in
                               data}  # useless because computed later by reading the annotation file
        return seq_list

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if is_gt:
            file = self.config["GT_LOC_FORMAT"].format(gt_folder=self.gt_fol, seq=seq)
        else:
            file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.json')

        with open(file, 'r') as f:
            data = json.load(f)

        if is_gt:
            self.categories = {categ['id']: categ for categ in data["categories"]}

            # Keep labeled images only
            images = [img for img in data["images"] if
                      img["has_labeled_pitch"] and img["has_labeled_camera"] and img["has_labeled_person"]]

            # Sort images by frame number
            def get_frame_number(image):
                return int(image['image_id'].split('_')[-1])

            self.images = sorted(images, key=get_frame_number)

            # Create a dictionary mapping from image_id to timestep
            self.image_id_to_timestep = {image["image_id"]: i for i, image in enumerate(self.images)}

        num_timesteps = len(self.images)

        # Initialize lists with None for each timestep
        ids = [None] * num_timesteps
        classes = [None] * num_timesteps
        dets = [None] * num_timesteps
        crowd_ignore_regions = [None] * num_timesteps
        extras = [None] * num_timesteps
        confidences = [None] * num_timesteps

        key = "annotations" if is_gt else "predictions"
        for annotation in data[key]:
            if annotation["supercategory"] != "object":  # ignore pitch and camera
                continue
            image_id = annotation["image_id"]
            if image_id not in self.image_id_to_timestep:
                if is_gt or len(image_id) == 10:
                    continue
                split_id = ["train", "valid", "test", "challenge"].index(self.split) + 1
                seq_id = seq.split("-")[-1]
                frame_id = int(image_id[-4:]) + 1
                new_image_id = f"{split_id}{seq_id}{frame_id:06d}"
                if new_image_id in self.image_id_to_timestep:
                    image_id = new_image_id
                else:
                    continue
            if self.ignore_ball and annotation["attributes"]["role"] == "ball":
                continue
            timestep = self.image_id_to_timestep[image_id]
            if ids[timestep] is None:
                ids[timestep] = []
                classes[timestep] = []
                dets[timestep] = []
                crowd_ignore_regions[timestep] = []
                extras[timestep] = []
                confidences[timestep] = []

            crowd_ignore_regions[timestep].append(np.empty((0, 4)))
            if self.eval_space == 'pitch':
                bbox_pitch = annotation["bbox_pitch"]
                assert bbox_pitch is not None
                dets[timestep].append([
                    bbox_pitch["x_bottom_left"], bbox_pitch["y_bottom_left"],
                    bbox_pitch["x_bottom_middle"], bbox_pitch["y_bottom_middle"],
                    bbox_pitch["x_bottom_right"], bbox_pitch["y_bottom_right"]
                ])
            elif self.eval_space == 'image':
                bbox_image = annotation["bbox_image"]
                dets[timestep].append([bbox_image["x"], bbox_image["y"], bbox_image["w"], bbox_image["h"]])
            else:
                raise ValueError("Invalid eval_space: " + self.eval_space)
            ids[timestep].append(annotation["track_id"])

            confidence = annotation["confidence"] if not is_gt and "confidence" in annotation else 1
            confidences[timestep].append(confidence)

            # Extract extra information if needed (modify this part based on your requirements)
            role = annotation["attributes"]["role"]
            team = annotation["attributes"]["team"]
            jersey_number = annotation["attributes"]["jersey"]
            # Preprocessing: put jersey number and team to none if the role is not a player or goalkeeper, so that they are ignored for similarity calculation in 'get_raw_seq_data'
            role = role if self.use_roles else None
            # if role are disabled, use team without preprocessing. Team should be set to "None" (null in the json) if the detection is not a player nor a goalkeeper
            team = team if (self.use_teams and self.use_roles and role in {"player", "goalkeeper"}) or (
                        self.use_teams and not self.use_roles) else None
            # if role are disabled, use jersey number without preprocessing. Jersey Numbers should be set to "None" (null in the json) if the detection is not a player nor a goalkeeper
            jersey_number = jersey_number if (self.use_jersey_numbers and self.use_roles and role in {"player"}) or (
                        self.use_jersey_numbers and not self.use_roles) else None
            assert role is None or role in {"other", "player", "goalkeeper", "referee"}
            assert team is None or team in {"left", "right", 'nan'}
            assert jersey_number is None or 0 <= int(jersey_number) <= 10000
            category_id = annotation["category_id"]
            extras[timestep].append({
                "role": role,
                "team": team,
                "jersey": jersey_number,
                "category_id": category_id,
                # Add more fields as needed
            })

            if self.eval_mode == 'classes':
                class_name = self.attributes_to_class_name(role, team, jersey_number)
                class_id = self.class_name_to_class_id[class_name] if class_name in self.class_name_to_class_id else 0
                # class_id = self.class_name_to_class_id[class_name] if class_name in self.class_name_to_class_id else -1
                classes[timestep].append(class_id)
            else:
                classes[timestep].append(1)

        # Convert lists to numpy arrays
        for t in range(num_timesteps):
            if ids[t] is None:
                ids[t] = np.empty(0).astype(int)
                classes[t] = np.empty(0).astype(int)
                dets[t] = np.empty((0, 4))
                crowd_ignore_regions[t] = np.empty((0, 4))
                confidences[t] = np.empty(0)
                extras[t] = []
            ids[t] = np.array(ids[t])
            classes[t] = np.array(classes[t])
            dets[t] = np.array(dets[t])
            crowd_ignore_regions[t] = np.array(crowd_ignore_regions[t])
            confidences[t] = np.array(confidences[t])

        if is_gt:
            raw_data = {
                "gt_classes": [np.array(x) for x in classes],
                "gt_crowd_ignore_regions": crowd_ignore_regions,
                "gt_dets": dets,
                "gt_extras": extras,
                "gt_ids": [np.array(x).astype(int) for x in ids],
                "seq": seq,
                "num_timesteps": num_timesteps,
            }
        else:
            raw_data = {
                "tracker_classes": [np.array(x) for x in classes],
                "tracker_dets": dets,
                "tracker_ids": [np.array(x).astype(int) for x in ids],
                "tracker_extras": extras,
                "tracker_confidences": [np.array(x) for x in confidences],
            }
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        SoccerNetGameState:
            TODO
        """
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):
            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets
            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets  # FIXME assert not 0 size
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    @_timing.time
    def get_raw_seq_data(self, tracker, seq):
        """ Loads raw data (tracker and ground-truth) for a single tracker on a single sequence.
        Raw data includes all of the information needed for both preprocessing and evaluation, for all classes.
        A later function (get_processed_seq_data) will perform such preprocessing and extract relevant information for
        the evaluation of each class.

        This returns a dict which contains the fields:
        [num_timesteps]: integer
        [gt_ids, tracker_ids, gt_classes, tracker_classes, tracker_confidences]:
                                                                list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, tracker_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [similarity_scores]: list (for each timestep) of 2D NDArrays.
        [gt_extras]: dict (for each extra) of lists (for each timestep) of 1D NDArrays (for each det).

        gt_extras contains dataset specific information used for preprocessing such as occlusion and truncation levels.

        Note that similarities are extracted as part of the dataset and not the metric, because almost all metrics are
        independent of the exact method of calculating the similarity. However datasets are not (e.g. segmentation
        masks vs 2D boxes vs 3D boxes).
        We calculate the similarity before preprocessing because often both preprocessing and evaluation require it and
        we don't wish to calculate this twice.
        We calculate similarity between all gt and tracker classes (not just each class individually) to allow for
        calculation of metrics such as class confusion matrices. Typically the impact of this on performance is low.

        SoccerNet game state specificity: similarity score is set to 0 if the attributes (jersey number, team, ...) of the tracker and the ground do not match.
        """
        # Load raw data.
        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}  # Merges dictionaries

        # Calculate similarities for each timestep.
        similarity_scores = []
        for t, (gt_dets_t, tracker_dets_t, gt_extras_t, tracker_extras_t) in enumerate(
                zip(raw_data['gt_dets'], raw_data['tracker_dets'], raw_data['gt_extras'], raw_data['tracker_extras'])):
            ious = self._calculate_similarities(gt_dets_t, tracker_dets_t)

            if self.eval_mode == 'distance':
                # give 0 similarity when the attributes do not match
                gt_roles = np.array([gt_extra['role'] for gt_extra in gt_extras_t])
                gt_teams = np.array([gt_extra['team'] for gt_extra in gt_extras_t])
                gt_jerseys = np.array([gt_extra['jersey'] for gt_extra in gt_extras_t])
                tracker_roles = np.array([tracker_extra['role'] for tracker_extra in tracker_extras_t])
                tracker_teams = np.array([tracker_extra['team'] for tracker_extra in tracker_extras_t])
                tracker_jerseys = np.array([tracker_extra['jersey'] for tracker_extra in tracker_extras_t])

                # Ensure dimensions are compatible for broadcasting by adding an extra dimension to `gt` arrays
                gt_roles = gt_roles[:, np.newaxis]
                gt_teams = gt_teams[:, np.newaxis]
                gt_jerseys = gt_jerseys[:, np.newaxis]

                # Comparisons (True where conditions are met)
                matches = (gt_roles == tracker_roles) & (gt_teams == tracker_teams) & (gt_jerseys == tracker_jerseys)

                # Since we want to set `ious` to 0 where conditions are NOT met, invert the match matrix
                non_matches = ~matches

                ious[non_matches] = 0

            similarity_scores.append(ious)
        raw_data['similarity_scores'] = similarity_scores

        return raw_data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if self.eval_space == 'pitch':
            if self.eval_sim_metric == 'gaussian':
                similarity_scores = self._calculate_gaussian_similarity(gt_dets_t, tracker_dets_t, self.eval_sigma)
            elif self.eval_sim_metric == 'iou':
                gt_dets_t = bbox_image_bottom_projection_to_bbox_pitch(gt_dets_t)
                tracker_dets_t = bbox_image_bottom_projection_to_bbox_pitch(tracker_dets_t)
                similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
            elif self.eval_sim_metric == 'eucl':
                similarity_scores = self._calculate_normalized_euclidean_similarity(gt_dets_t, tracker_dets_t)
            else:
                raise ValueError("Invalid eval_sim_metric: " + self.eval_sim_metric)
        elif self.eval_space == 'image':
            similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        else:
            raise ValueError("Invalid eval_space: " + self.eval_space)

        # Flatten the similarity_scores and add them to the list
        self.all_similarity_scores.extend(similarity_scores.flatten())

        return similarity_scores

    def _calculate_gaussian_similarity(self, gt_dets_t, tracker_dets_t, sigma=2.5):
        # Extract the middle points
        gt_middle_points = gt_dets_t[:, 2:4]  # x_bottom_middle, y_bottom_middle
        tracker_middle_points = tracker_dets_t[:, 2:4]  # x_bottom_middle, y_bottom_middle

        # Compute the Euclidean distance between the middle points
        diff = np.expand_dims(gt_middle_points, axis=1) - np.expand_dims(tracker_middle_points, axis=0)
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Apply the Gaussian function to the distances
        similarities = np.exp(-0.5 * (distances / sigma) ** 2)

        return similarities

    def _calculate_normalized_euclidean_similarity(self, gt_dets_t, tracker_dets_t, normalization_factor=1.0,
                                                   small_value=1e-10):
        # gt_dets_t and tracker_dets_t are numpy arrays of shape (N, 6) and (M, 6) respectively
        # where N and M are the number of points in each set.
        # Each row in the arrays represents a point (x_bottom_left, y_bottom_left, x_bottom_middle, y_bottom_middle, x_bottom_right, y_bottom_right)

        # Extract the middle points
        gt_middle_points = gt_dets_t[:, 2:4]  # x_bottom_middle, y_bottom_middle
        tracker_middle_points = tracker_dets_t[:, 2:4]  # x_bottom_middle, y_bottom_middle

        # Compute the Euclidean distance between the middle points
        diff = np.expand_dims(gt_middle_points, axis=1) - np.expand_dims(tracker_middle_points, axis=0)
        euclidean_distance = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Compute the width of the ground truth detections
        gt_width = gt_dets_t[:, 4] - gt_dets_t[:, 0]  # x_bottom_right - x_bottom_left

        # Replace zeros in gt_width with small_value
        gt_width = np.where(gt_width == 0, small_value, gt_width)

        # Normalize the Euclidean distance by the width of the ground truth detections
        normalized_distance = euclidean_distance / (gt_width[:, np.newaxis] * normalization_factor)

        # Clip the final values to the range [0, 1]
        normalized_distance = 1 - np.clip(normalized_distance, 0, 1)

        return normalized_distance

    def attributes_to_class_name(self, role, team, jersey_number):
        if not self.use_jersey_numbers:
            jersey_number = None
        if not self.use_teams:
            team = None
        if not self.use_roles:
            role = "player"  # ignore the role but still uses team or jn

        if "goalkeeper" in role:
            role = "goalkeeper"  # some are tagged as "goalkeepersS"

        if role == "goalkeeper" or role == "player":
            if jersey_number is not None:
                category = f"{role}_{team}_{jersey_number}"
            else:
                category = f"{role}_{team}"
        else:
            category = f"{role}"
        return category

    def extract_all_classes(self, config, gt_fol, seq_list):
        all_classes = {}
        for seq in seq_list:
            # File location
            file = config["GT_LOC_FORMAT"].format(gt_folder=gt_fol, seq=seq)

            with open(file, 'r') as f:
                data = json.load(f)

            for annotation in data["annotations"]:
                if annotation["supercategory"] != "object":  # ignore pitch and camera
                    continue
                role = annotation["attributes"]["role"]
                jersey_number = annotation["attributes"]["jersey"]
                team = annotation["attributes"]["team"]
                class_name = self.attributes_to_class_name(role, team, jersey_number)
                if class_name not in all_classes:
                    all_classes[class_name] = {
                        "id": len(all_classes) + 1,
                        "name": class_name,
                        "supercategory": "object"
                    }
        return all_classes


def bbox_image_bottom_projection_to_bbox_pitch(dets):
    """
    Convert bounding boxes bottom left/center/right points in pitch space to a pitch bounding box.
    The pitch bbox is a square centered of the image bbox projected bottom center, with the height and width equal to the distance between the bottom left and bottom right points.
    """
    # Extract the middle points
    x_bottom_middle = dets[:, 2]
    y_bottom_middle = dets[:, 3]

    # Calculate the width and height of the bounding box
    x_bottom_left = dets[:, 0]
    y_bottom_left = dets[:, 1]
    x_bottom_right = dets[:, 4]
    y_bottom_right = dets[:, 5]
    bbox_width_height = np.sqrt((x_bottom_right - x_bottom_left) ** 2 + (y_bottom_right - y_bottom_left) ** 2)

    # Create a new array with the calculated values
    transformed_dets = np.stack((x_bottom_middle, y_bottom_middle, bbox_width_height, bbox_width_height), axis=-1)

    return transformed_dets


def calculate_sigma(x):
    """Compute gaussian kernel sigma value based on the following assumption:
    The input x value in meters should produce a similarity of 0.05.
    0.05 is the minimum acceptable similarity threshold for groundtruth/prediction matching in the HOTA metric."""
    sigma = np.sqrt(x ** 2 / (-2 * np.log(0.05)))
    return sigma