k_posetrack_anno_path = '/home/neohanju/Workspace/dataset/posetrack/annotations/val';
anno_files = dir(fullfile(k_posetrack_anno_path, '*.mat'));

for i = 1:length(anno_files)
    fprintf('Process %s...[%3d/%3d]\n', anno_files(i).name, i, length(anno_files));
    posetrack_mat_to_csv(fullfile(k_posetrack_anno_path, anno_files(i).name));
end

%()()
%('')HAANJU.YOO
