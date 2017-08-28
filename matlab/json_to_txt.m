DATA_ROOT = 'D:/Workspace/Dataset/ETRI/GarbageDumping/CPM_results';
NUM_POINT_TYPES = 18;
DO_VIDEO_DEBUG = true;

x_pos = zeros(0, NUM_POINT_TYPES);
y_pos = zeros(0, NUM_POINT_TYPES);
confidences = zeros(0, NUM_POINT_TYPES);

if DO_VIDEO_DEBUG
    VIDEO_BASE_PATH = 'D:\Workspace\Dataset\ETRI\GarbageDumping\numbering';
end

for i = 1:219
    folder_path = fullfile(DATA_ROOT, sprintf('%03d', i));
    files = dir(fullfile(folder_path, '*.json'));
    filenames = {files(:).name};
    fprintf('processing %s\n', folder_path);
    
    if DO_VIDEO_DEBUG
        video_path = fullfile(VIDEO_BASE_PATH, sprintf('%03d.mp4', i));
        video_object = VideoReader(video_path);
    end
    
    for j = 1:length(filenames)
        json_body = loadjson(fullfile(folder_path, filenames{j}));
        
        if DO_VIDEO_DEBUG
            image_frame = read(video_object, j);
            imshow(image_frame, 'border', 'tight');
            hold on;
        else
            fp = fopen(fullfile(folder_path, strrep(filenames{j}, ...
                '.json', '.txt')), 'wt');
            % num objects
            fprintf(fp, '%d\n', length(json_body.people));
        end    
        
        for pIdx = 1:length(json_body.people)
            cur_keypoints_read = json_body.people{pIdx}.pose_keypoints;
            cur_x_pos = zeros(1, NUM_POINT_TYPES);
            cur_y_pos = zeros(1, NUM_POINT_TYPES);
            cur_confidences = zeros(1, NUM_POINT_TYPES);
            cur_keypoints = zeros(1, 3*NUM_POINT_TYPES);
            for k = 1:NUM_POINT_TYPES
                cur_x_pos(k) = cur_keypoints_read{3*k-2};
                cur_y_pos(k) = cur_keypoints_read{3*k-1};
                cur_confidences(k) = cur_keypoints_read{3*k};
                cur_keypoints(3*k-2:3*k) = ...
                    [cur_x_pos(k), cur_y_pos(k), cur_confidences(k)];
                                
                if DO_VIDEO_DEBUG
                    text(cur_x_pos(k), cur_y_pos(k), ...
                        ['\fontsize{10}\color{white}' int2str(k)], ...
                        'BackgroundColor', 'k');
                else
                    fprintf(fp, '%f,%f,%f,', ...
                        cur_x_pos(k), cur_y_pos(k), cur_confidences(k));
                end
            end
            
            if ~DO_VIDEO_DEBUG
                fprintf(fp, '\n');
            end
            
%             x_pos(end+1,:) = cur_x_pos;
%             y_pos(end+1,:) = cur_y_pos;
%             confidences(end+1,:) = cur_confidences;            
        end
               
        if DO_VIDEO_DEBUG
            hold off;
        else
            fclose(fp);
        end
    end
end

%()()
%('')HAANJU.YOO
