clear variables, close all

video_directory = 'C:\path\to\videos';

run_test_1_pca = true;
run_test_2_pca = true;
run_test_3_pca = true;
run_test_4_pca = true;
% To show the mass tracking for a given test, the pca must also be run
show_test_1_mass_tracking = false;
show_test_2_mass_tracking = false;
show_test_3_mass_tracking = false;
show_test_4_mass_tracking = false;
% To show the scatter plot for a given test, the pca must also be run
show_test_1_scatter_plot = false;
show_test_2_scatter_plot = false;
show_test_3_scatter_plot = false;
show_test_4_scatter_plot = false;
show_test_1_videos = false;
show_test_2_videos = false;
show_test_3_videos = false;
show_test_4_videos = false;

%% Loading Data and Playing Videos

% The videos are matrices correspondingly named but with the prefix 
% vidFrames instead of cam
% The videos are all different lengths, so we trim them to be all the
% length of the shortest video, which is 226 frames, by aligning the start 
% of the videos based on observation
video_trim_length = 226; % frames
if run_test_1_pca || show_test_1_videos
    load(strcat(video_directory, 'cam1_1.mat'))
    vidFrames1_1_start_frame = 1;
    vidFrames1_1 = trim_video(vidFrames1_1, vidFrames1_1_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam2_1.mat'))
    vidFrames2_1_start_frame = 10;
    vidFrames2_1 = trim_video(vidFrames2_1, vidFrames2_1_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam3_1.mat'))
    vidFrames3_1_start_frame = 1;
    vidFrames3_1 = trim_video(vidFrames3_1, vidFrames3_1_start_frame, video_trim_length);
end
if run_test_2_pca || show_test_2_videos
    load(strcat(video_directory, 'cam1_2.mat'))
    vidFrames1_2_start_frame = 1;
    vidFrames1_2 = trim_video(vidFrames1_2, vidFrames1_2_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam2_2.mat'))
    vidFrames2_2_start_frame = 24;
    vidFrames2_2 = trim_video(vidFrames2_2, vidFrames2_2_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam3_2.mat'))
    vidFrames3_2_start_frame = 5;
    vidFrames3_2 = trim_video(vidFrames3_2, vidFrames3_2_start_frame, video_trim_length);
end
if run_test_3_pca || show_test_3_videos
    load(strcat(video_directory, 'cam1_3.mat'))
    vidFrames1_3_start_frame = 1;
    vidFrames1_3 = trim_video(vidFrames1_3, vidFrames1_3_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam2_3.mat'))
    vidFrames2_3_start_frame = 29;
    vidFrames2_3 = trim_video(vidFrames2_3, vidFrames2_3_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam3_3.mat'))
    vidFrames3_3_start_frame = 1;
    vidFrames3_3 = trim_video(vidFrames3_3, vidFrames3_3_start_frame, video_trim_length);
end
if run_test_4_pca || show_test_4_videos
    load(strcat(video_directory, 'cam1_4.mat'))
    vidFrames1_4_start_frame = 1;
    vidFrames1_4 = trim_video(vidFrames1_4, vidFrames1_4_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam2_4.mat'))
    vidFrames2_4_start_frame = 3;
    vidFrames2_4 = trim_video(vidFrames2_4, vidFrames2_4_start_frame, video_trim_length);
    load(strcat(video_directory, 'cam3_4.mat'))
    vidFrames3_4_start_frame = 1;
    vidFrames3_4 = trim_video(vidFrames3_4, vidFrames3_4_start_frame, video_trim_length);
end

if show_test_1_videos
    show_movies(vidFrames1_1, vidFrames2_1, vidFrames3_1, [], [], [])
end
if show_test_2_videos
    show_movies(vidFrames1_2, vidFrames2_2, vidFrames3_2, [], [], [])
end
if show_test_3_videos
    show_movies(vidFrames1_3, vidFrames2_3, vidFrames3_3, [], [], [])
end
if show_test_4_videos
    show_movies(vidFrames1_4, vidFrames2_4, vidFrames3_4, [], [], [])
end

%% Performing the PCA

if run_test_1_pca
    test_1_explained = run_pca(vidFrames1_1, vidFrames2_1, vidFrames3_1, show_test_1_mass_tracking, show_test_1_scatter_plot);
    disp('Variance explained by each principal component for test 1:')
    disp(test_1_explained)
end
if run_test_2_pca
    test_2_explained = run_pca(vidFrames1_2, vidFrames2_2, vidFrames3_2, show_test_2_mass_tracking, show_test_2_scatter_plot);
    disp('Variance explained by each principal component for test 2:')
    disp(test_2_explained)
end
if run_test_3_pca
    test_3_explained = run_pca(vidFrames1_3, vidFrames2_3, vidFrames3_3, show_test_3_mass_tracking, show_test_3_scatter_plot);
    disp('Variance explained by each principal component for test 3:')
    disp(test_3_explained)
end
if run_test_4_pca
    test_4_explained = run_pca(vidFrames1_4, vidFrames2_4, vidFrames3_4, show_test_4_mass_tracking, show_test_4_scatter_plot);
    disp('Variance explained by each principal component for test 4:')
    disp(test_4_explained)
end

%% Functions

function centroids = find_centroids(filtered_video)
    % filtered_video is expected to be a 3D matrix of only logical values
    % The rows of the returned matrix correspond to 
    
    centroids = zeros(size(filtered_video, 4), 2);
    for i=1:size(filtered_video, 3)
        filtered_frame = filtered_video(:,:,i);
        [rows, cols] = size(filtered_frame);
        
        if ~any(filtered_frame, 'all') && i~= 1
            % If there are no pixels that match our filter, just set the
            % coordinates to the last value, which will reduce our variance
            % further than having an extraneous entry of (0,0). We check
            % that i isn't 1 to avoid an indexing error 
            centroids(i, :) = centroids(i-1, :);
            continue
        end

        x = 1:cols;
        y = 1:rows;

        [X, Y] = meshgrid(x,y);

        centroid = [mean(X(boolean(filtered_frame))) mean(Y(boolean(filtered_frame)))];
        centroids(i, :) = centroid;
    end
end

function explained = run_pca(video1, video2, video3, show_mass_tracking, show_scatter_plot)

    % Our first task is to track the mass during each video, to do this we
    % exploit the yellow band present on the mass by filtering the video by
    % the color yellow, then finding the centroid of all yellow present on
    % each frame. To get good results with our filtering, we have to use
    % different values for each video
    
    video1_yellow_filter = yellow_filter(video1);
    video2_yellow_filter = yellow_filter(video2);
    video3_yellow_filter = yellow_filter(video3);
    
    video1_centroids = find_centroids(video1_yellow_filter);
    video2_centroids = find_centroids(video2_yellow_filter);
    video3_centroids = find_centroids(video3_yellow_filter);
    
    if show_mass_tracking
        show_movies(video1, video2, video3, video1_centroids, video2_centroids, video3_centroids)
    end
    if show_scatter_plot
        figure()
        x = [video1_centroids(:, 1); video2_centroids(:, 1); video3_centroids(:, 1)];
        y = [video2_centroids(:, 2); video3_centroids(:, 2); video3_centroids(:, 2)];
        scatter(x, y)
        xlabel('Pixel x index)'), ylabel('Pixel y index')
        drawnow
    end
    
    % This matrix is the transpose of the form presented in class, but is
    % the form expected by MATLAB
    coordinates_matrix = [video1_centroids video2_centroids video3_centroids];
    [~, ~, ~, ~, explained] = pca(coordinates_matrix);
end

function show_movies(video1, video2, video3, video1_centroids, video2_centroids, video3_centroids)
    % Note that we don't control the speed that the videos play at, so the
    % video will play at the speed MATLAB renders the images at, but this
    % has looked reasonable on my computer
    
    % If not plotting the tracked centroids, the centroid matrices are
    % expected to be empty
    
    figure()
    video1_frames = size(video1, 4);
    video2_frames = size(video2, 4);
    video3_frames = size(video3, 4);
    frames = max([video1_frames video2_frames video3_frames]);
    for i=1:frames
        subplot(1, 3, 1)
        if i<=video1_frames
            imshow(video1(:, :, :, i))
            if ~isempty(video1_centroids)
                hold on
                plot(video1_centroids(i, 1), video1_centroids(i, 2), 'ro')
                hold off
            end
        else
            imshow(zeros(size(video1(:, :, :, 1))))
        end
        drawnow
        subplot(1, 3, 2)
        if i<=video2_frames
            imshow(video2(:, :, :, i))
            if ~isempty(video2_centroids)
                hold on
                plot(video2_centroids(i, 1), video2_centroids(i, 2), 'ro')
                hold off
            end
        else
            imshow(zeros(size(video2(:, :, :, 1))))
        end
        drawnow
        subplot(1, 3, 3)
        if i<=video3_frames
            imshow(video3(:, :, :, i))
            if ~isempty(video3_centroids)
                hold on
                plot(video3_centroids(i, 1), video3_centroids(i, 2), 'ro')
                hold off
            end
        else
            imshow(zeros(size(video3(:, :, :, 1))))
        end
        drawnow
    end
end

function trimmed_video = trim_video(video, start_frame, length)
    trimmed_video = video(:,:,:,start_frame:start_frame+length-1);
end

function filtered_video = yellow_filter(video)
    % To filter the video by the color yellow we note that yellow is
    % rgb(255, 255, 0). To get good results with this filtering we have to
    % use different values for every video. To be filtered pixels must have
    % a value above red_threshold and green_threshold and below
    % blue_threshold.
    
    red_threshold = 200;
    green_threshold = 200;
    blue_threshold = 175;

    filtered_video = zeros(size(video, 1), size(video, 2), size(video, 4)); 
    for i=1:size(video, 4)
        red_frame = video(:,:,1, i);
        filtered_red_frame = red_frame>=red_threshold;
        green_frame = video(:,:,2,i);
        filtered_green_frame = green_frame >= green_threshold;
        blue_frame = video(:,:,3,i);
        filtered_blue_frame = blue_frame <= blue_threshold;
        filtered_video(:,:,i) = filtered_red_frame & filtered_green_frame & filtered_blue_frame;
    end

end
