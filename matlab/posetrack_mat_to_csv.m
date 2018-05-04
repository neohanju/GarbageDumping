function posetrack_mat_to_csv(anno_path)

if nargin < 1
    anno_path = '/home/neohanju/Workspace/dataset/posetrack/annotations/train/000003_bonn_relpath_5sec_trainsub.mat';
end

if ~exist(anno_path, 'file')
    error('Cannot find annotation file. Check the path: %s', anno_path);
end

% csv column titles
attributes = {...
    'frameNumber', 'head_x1', 'head_y1', 'head_x2', 'head_y2', 'track_id', ...
    'x0',  'y0',  'is_visible_0' ...
    'x1',  'y1',  'is_visible_1' ...
    'x2',  'y2',  'is_visible_2' ...
    'x3',  'y3',  'is_visible_3' ...
    'x4',  'y4',  'is_visible_4' ...
    'x5',  'y5',  'is_visible_5' ...
    'x6',  'y6',  'is_visible_6' ...
    'x7',  'y7',  'is_visible_7' ...
    'x8',  'y8',  'is_visible_8' ...
    'x9',  'y9',  'is_visible_9' ...
    'x10', 'y10', 'is_visible_10' ...
    'x11', 'y11', 'is_visible_11' ...
    'x12', 'y12', 'is_visible_12' ...
    'x13', 'y13', 'is_visible_13' ...
    'x14', 'y14', 'is_visible_14'};
kX0Pos = 7;

% load annotation
[filepath, filename, ~] = fileparts(anno_path);
load(anno_path);  % read 'annolist'

% read annotation data and save them into 2D matrix
num_frames = length(annolist);
csvdata = zeros(0, length(attributes));
% fprintf('Process %s...\n', filename);
for i = 1:num_frames
%     fprintf('%d/%d\n', i, num_frames);
    if ~annolist(i).is_labeled
        continue;
    end
    for j = 1:length(annolist(i).annorect)
        
        if isempty(annolist(i).annorect(j).annopoints)
            continue;
        end
        
        [~, frame_number, ~] = fileparts(annolist(i).image.name);
        
        cur_rect = annolist(i).annorect(j);        
        cur_record = zeros(1, length(attributes));
        cur_record(1) = str2num(frame_number);
        cur_record(2:6) = ...
            [cur_rect.x1, cur_rect.y1, cur_rect.x2, cur_rect.y2, cur_rect.track_id];
        
        cur_points = cur_rect.annopoints.point;
        for k = 1:length(cur_points)
            cur_record(kX0Pos+3*cur_points(k).id:kX0Pos+3*cur_points(k).id+2) = ...
                [cur_points(k).x, cur_points(k).y, 1];  % last element = is_visible
        end
        
        csvdata(end+1,:) = cur_record;
    end
end

T = array2table(csvdata, 'VariableNames', attributes);
writetable(T, fullfile(filepath, [filename, '.csv']));

end

%()()
%('')HAANJU.YOO

