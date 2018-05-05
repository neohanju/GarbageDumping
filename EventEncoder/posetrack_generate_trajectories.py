import os
import csv
import json
import glob
import progressbar
from collections import OrderedDict
from utils import intersection_over_union

kCOCOKeypointsBasePath = '/home/neohanju/Workspace/dataset/posetrack/keypoints_COCO'
kHaanjuHome = '/home/neohanju/Workspace/dataset'
kJMHome = 'C:/Users/JM/Desktop/Data/ETRIrelated/BMVC'

kCurrentHome = kJMHome
kPosetrackCSVAnnotationBasePath = os.path.join(kCurrentHome, 'posetrack/annotations/csv')
kCOCOKeypointsBasePath = os.path.join(kCurrentHome, 'posetrack/keypoints_COCO')

def load_posetrack_csv_annotation(anno_path):
    with open(anno_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        keys = next(reader)
        dict_list = [OrderedDict(zip(keys, row)) for row in reader]

    return {'setname': os.path.basename(anno_path).split('.')[0],
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

    return {'setname': '_'.join(dir_name.split('_')[0:-1]),
            'detections': detections}


def load_coco_keypoints_all(keypoints_base_dir=kCOCOKeypointsBasePath):
    parent_dir_name_list = next(os.walk(keypoints_base_dir))[1]
    parent_dir_name_list.sort()
    path_list = []
    for parent_dir in parent_dir_name_list:
        child_dir_name_list = next(os.walk(os.path.join(keypoints_base_dir, parent_dir)))[1]
        path_list += [os.path.join(keypoints_base_dir, parent_dir, current_dir) for current_dir in child_dir_name_list]

    print('>> Read keypoints from COCO model')
    dict_list = [load_coco_keypoints(path_list[i]) for i in progressbar.progressbar(range(len(path_list)))]

    return dict_list


def is_keypoints_in_bbox(keypoints, bbox):
    # keypoints; [x0, y0, confidence_0, ..., x18, y18, confidence_18]
    # bbox: [xmin, ymin, xmax, ymax]
    [xmin, ymin, xmax, ymax] = bbox
    point_check_list = [1, 2, 5]
    for check_idx in point_check_list:
        if xmin > keypoints[3 * check_idx] or xmax < keypoints[3 * check_idx]:
            return False
        if ymin > keypoints[3 * check_idx + 1] or ymax < keypoints[3 * check_idx + 1]:
            return False
    return True


def get_trajectories(posetrack_annotation, coco_keypoint):
    assert (posetrack_annotation['setname'] == coco_keypoint['setname'])

    # for allocation
    max_track_id = 0
    for cur_anno in posetrack_annotation['annotations']:
        if max_track_id < int(cur_anno['track_id']):
            max_track_id = int(cur_anno['track_id'])

    # clustering with track ID and set bounding box
    anno_with_ID = [[] for _ in range(max_track_id + 1)]
    for cur_anno in posetrack_annotation['annotations']:
        x0_idx = list(cur_anno.keys()).index("x0")
        keypoints = list(cur_anno.items())[x0_idx:x0_idx+15*3]  # list of tuples like [('x0', '213'), ...]
        xs = [float(point[1]) for point in keypoints[0::3] if float(point[1]) != 0]
        ys = [float(point[1]) for point in keypoints[1::3] if float(point[1]) != 0]
        cur_anno['bbox'] = [min(xs), min(ys), max(xs), max(ys)]
        anno_with_ID[int(cur_anno['track_id'])].append(cur_anno)

    # calculate bounding box of coco model's keypoints
    for frame_info in coco_keypoint['detections']:
        frame_info['bbox'] = []
        for keypoints in frame_info['keypoints']:
            xs, ys = [], []
            for p in range(0, len(keypoints), 3):
                if 0 == keypoints[p + 2]:
                    continue
                xs.append(keypoints[p])
                ys.append(keypoints[p + 1])
            frame_info['bbox'].append([min(xs), min(ys), max(xs), max(ys)])

    result_trajectories = []
    for person in anno_with_ID:
        coco_idx = 0
        cur_trajectory = []
        for pose in person:
            # {bbox, frameNumber, head_x1, head_y1, head_x2, head_y2, track_id, x0, y0, is_visible_0 ... x14, y14, is_visible_14}

            # find concurrent coco keypoints
            while coco_idx < len(coco_keypoint['detections']):
                if int(coco_keypoint['detections'][coco_idx]['frameNumber']) < int(pose['frameNumber']):
                    coco_idx += 1
                else:
                    break

            if int(coco_keypoint['detections'][coco_idx]['frameNumber']) > int(pose['frameNumber']):
                # there is no concurrent keypoint
                continue

            # current_coco_detections = []
            # while coco_idx < len(coco_keypoint['detections']):
            #     if int(coco_keypoint['detections'][coco_idx]['frameNumber']) == int(pose['frameNumber']):
            #         current_coco_detections.append(coco_keypoint['detections'][coco_idx])
            #         coco_idx += 1
            #     else:
            #         break

            # find matching keypoint among concurrent keypoints
            # criterion: largest I.O.U.(intersection over union)
            # but, neck and shoulders of max I.O.U. must be included by annotation box

            detection = coco_keypoint['detections'][coco_idx]
            if 0 == len(detection['keypoints']):
                continue

            bbox_iou = [intersection_over_union(pose['bbox'], detection['bbox'][i])
                        for i, keypoints in enumerate(detection['keypoints'])]
            max_iou_pos = bbox_iou.index(max(bbox_iou))
            if is_keypoints_in_bbox(detection['keypoints'][max_iou_pos], pose['bbox']):
                cur_trajectory.append(
                    [int(pose['track_id']), 1, int(pose['frameNumber'])] + detection['keypoints'][max_iou_pos] + [0])

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
            if posetrack_annotation['setname'] == coco_keypoint['setname']:
                save_trajectories(os.path.join(save_base_path, posetrack_annotation['setname'] + '.txt'),
                                  get_trajectories(posetrack_annotation, coco_keypoint))
            else:
                left_coco_keypoints.append(coco_keypoint)
        coco_keypoints = left_coco_keypoints


if "__main__" == __name__:
    save_trajectories_from_all(kCOCOKeypointsBasePath)

# ()()
# ('') HAANJU.YOO
