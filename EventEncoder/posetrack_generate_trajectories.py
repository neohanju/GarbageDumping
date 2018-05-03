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
    return dict_list


def load_posetrack_cvs_annotation_all(anno_base_path=kPosetrackCSVAnnotationBasePath):
    file_paths = glob.glob(os.path.join(anno_base_path, '*.csv'))
    file_paths.sort()

    print('>> Read posetrack annotations')
    dict_list = []
    for i in progressbar.progressbar(range(len(file_paths))):
        cur_dict = {'filename': os.path.basename(file_paths[i]).split('_relpath')[0],
                    'annotions': []}
        cur_dict['annotions'].append(load_posetrack_csv_annotation(file_paths[i]))
        dict_list.append(cur_dict)

    return dict_list


def load_coco_keypoints(keypoints_dir):
    dir_name = os.path.basename(keypoints_dir)
    cur_dict = {'filename': '_'.join(dir_name.split('_')[0:-1]),
                'detections': []}
    file_paths = glob.glob(os.path.join(keypoints_dir, '*.json'))
    file_paths.sort()
    for file_path in file_paths:
        cur_frame_dict = {'imgnum': os.path.basename(file_path).split('_')[0],
                          'keypoints': []}
        with open(file_path, 'r') as json_file:
            json_data = json.loads(json_file.read())

        for people_info in json_data['people']:
            cur_frame_dict['keypoints'].append(people_info['pose_keypoints_2d'])

        cur_dict['detections'].append(cur_frame_dict)

    return cur_dict


def load_coco_keypoints_all(keypoints_base_dir=kCOCOKeypointsBasePath):
    dir_name_list = next(os.walk(keypoints_base_dir))[1]
    dir_name_list.sort()

    dict_list = []
    print('>> Read keypoints from COCO model')
    for i in progressbar.progressbar(range(len(dir_name_list))):
        dict_list.append(load_coco_keypoints(os.path.join(keypoints_base_dir, dir_name_list[i])))

    return dict_list


# def get_trajectories_from_posetrack_annotation(posetrack_anno_path, coco_keypoints_base_path):
#
#     anno_data = load_posetrack_csv_annotation(posetrack_anno_path)
#     keypoint_data = load_coco_keypoints(coco_keypoints_base_path)
#
#     # for allocation
#     max_track_id = 0
#     for cur_anno in anno_data:
#         if max_track_id < cur_anno['track_id']:
#             max_track_id = cur_anno['track_id']
#
#     trajectories = [] * (max_track_id + 1)
#     for cur_anno in anno_data:
#         trajectories[cur_anno['track_id']].append(cur_anno)
#
#     return trajectories


if "__main__" == __name__:
    posetrack_annos = load_posetrack_cvs_annotation_all()
    coco_keypoints = load_coco_keypoints_all()

