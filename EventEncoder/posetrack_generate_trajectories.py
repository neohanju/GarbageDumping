import os
import csv
import json
import glob
import progressbar

kPosetrackCSVAnnotationBasePath = '/home/neohanju/Workspace/dataset/posetrack/annotations/csv'
kCOCOKeypointsBasePath = '/home/neohanju/Workspace/dataset/posetrack/keypoints_COCO'


def load_posetrack_csv_annotation(anno_path):
    dict_list = []
    with open(anno_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dict_list.append(row)

    return {'filename': os.path.basename(anno_path).split('_relpath')[0],
            'annotations': dict_list}


def load_posetrack_cvs_annotation_all(anno_base_path=kPosetrackCSVAnnotationBasePath):
    file_paths = glob.glob(os.path.join(anno_base_path, '*.csv'))
    file_paths.sort()
    print('>> Read posetrack annotations')
    dict_list = []
    for i in progressbar.progressbar(range(len(file_paths))):
        dict_list.append(load_posetrack_csv_annotation(file_paths[i]))
    return dict_list


def load_coco_keypoints(keypoints_dir):
    dir_name = os.path.basename(keypoints_dir)
    file_paths = glob.glob(os.path.join(keypoints_dir, '*.json'))
    file_paths.sort()

    detections = []
    for file_path in file_paths:
        cur_frame_dict = {'frameNumber': os.path.basename(file_path).split('_')[0],
                          'keypoints': []}
        with open(file_path, 'r') as json_file:
            json_data = json.loads(json_file.read())

        for people_info in json_data['people']:
            cur_frame_dict['keypoints'].append(people_info['pose_keypoints_2d'])

        detections.append(cur_frame_dict)

    return {'filename': '_'.join(dir_name.split('_')[0:-1]),
            'detections': detections}


def load_coco_keypoints_all(keypoints_base_dir=kCOCOKeypointsBasePath):
    dir_name_list = next(os.walk(keypoints_base_dir))[1]
    dir_name_list.sort()

    dict_list = []
    print('>> Read keypoints from COCO model')
    for i in progressbar.progressbar(range(len(dir_name_list))):
        dict_list.append(load_coco_keypoints(os.path.join(keypoints_base_dir, dir_name_list[i])))

    return dict_list


def is_keypoints_in_bbox(keypoints, bbox):
    # keypoints; [x0, y0, confidence_0, ..., x18, y18, confidence_18]
    # bbox: [xmin, ymin, xmax, ymax]
    [xmin, ymin, xmax, ymax] = bbox
    point_check_list = [1, 2, 5]
    for check_idx in point_check_list:
        if xmin > keypoints[3*check_idx] or xmax < keypoints[3*check_idx]:
            return False
        if ymin > keypoints[3*check_idx+1] or ymax < keypoints[3*check_idx+1]:
            return False
    return True


def get_trajectories(posetrack_annotation, coco_keypoint):

    assert(posetrack_annotation['filename'] == coco_keypoint['filename'])

    # for allocation
    max_track_id = 0
    for cur_anno in posetrack_annotation['annotations']:
        if max_track_id < int(cur_anno['track_id']):
            max_track_id = int(cur_anno['track_id'])

    anno_with_ID = [[] for i in range(max_track_id + 1)]
    for cur_anno in posetrack_annotation['annotations']:
        anno_with_ID[int(cur_anno['track_id'])].append(cur_anno)

    result_trajectories = []
    for person in anno_with_ID:
        coco_idx = 0
        cur_trajectory = []
        for pose in person:
            # {frameNumber, head_x1, head_y1, head_x2, head_y2, track_id, x0, y0, is_visible_0 ... x14, y14, is_visible_14}

            # get bounding box range
            x0_idx = list(pose.keys()).index("x0")
            keypoints = list(pose.items())[x0_idx:]  # list of tuples like [('x0', '213'), ...]
            xs = [float(point[1]) for point in keypoints[0::3] if float(point[1]) != 0]
            ys = [float(point[1]) for point in keypoints[1::3] if float(point[1]) != 0]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            # find concurrent coco keypoints
            while coco_idx < len(coco_keypoint['detections']):
                if int(coco_keypoint['detections'][coco_idx]['frameNumber']) < int(pose['frameNumber']):
                    coco_idx += 1
                else:
                    break

            if int(coco_keypoint['detections'][coco_idx]['frameNumber']) > int(pose['frameNumber']):
                # there is no concurrent keypoint
                continue

            current_coco_detections = []
            while coco_idx < len(coco_keypoint['detections']):
                if int(coco_keypoint['detections'][coco_idx]['frameNumber']) == int(pose['frameNumber']):
                    current_coco_detections.append(coco_keypoint['detections'][coco_idx])
                    coco_idx += 1
                else:
                    break

            # find matched keypoint among concurrent keypoints
            for detections in current_coco_detections:
                for keypoints in detections['keypoints']:
                    if is_keypoints_in_bbox(keypoints, bbox):
                        # [track_id, is_suspect(1), imgnum, keypoints..., is_trhow_garbage(0)]
                        cur_trajectory.append(
                            [int(pose['track_id']), 1, int(pose['frameNumber'])] + keypoints + [0])

        result_trajectories.append(cur_trajectory)

    return result_trajectories


def save_trajectories(save_path, trajectories):
    with open(save_path, 'w') as txtfile:
        for trajectory in trajectories:
            for pose in trajectory:
                txtfile.write(' '.join(map(lambda x: str(x), pose)) + '\n')


def save_trajectories_from_all(save_base_path,
                               posetrack_anno_base_path=kPosetrackCSVAnnotationBasePath,
                               coco_keypoints_base_path=kCOCOKeypointsBasePath):
    posetrack_annos = load_posetrack_cvs_annotation_all(posetrack_anno_base_path)
    coco_keypoints = load_coco_keypoints_all(coco_keypoints_base_path)

    for posetrack_annotation in posetrack_annos:
        left_coco_keypoints = []
        for coco_keypoint in coco_keypoints:
            if posetrack_annotation['filename'] == coco_keypoint['filename']:
                save_trajectories(os.path.join(save_base_path, posetrack_annotation['filename'] + '.txt'),
                                  get_trajectories(posetrack_annotation, coco_keypoint))
            else:
                left_coco_keypoints.append(coco_keypoint)
        coco_keypoints = left_coco_keypoints


if "__main__" == __name__:
    save_trajectories_from_all(kCOCOKeypointsBasePath)

