function draw_frame(video_path, frame_index, xs, ys)

video_object = VideoReader(video_path);
image_frame = read(video_object, frame_index);
imshow(image_frame, 'border', 'tight');
hold on;
for i = 1:length(xs)
    rectangle('Position', [xs(i), ys(i), 5, 5], 'EdgeColor', 'r', 'LineWidth', 1, 'LineStyle', '-');
    text(xs(i), ys(i), ['\fontsize{10}\color{white}' int2str(i)], 'BackgroundColor', 'k');
end
hold off;

end

%()()
%('')HAANJU.YOO
