close all, clear variables

run_monte_carlo_dmd = false;
run_ski_drop_dmd = true;
show_pca_modes = true;
show_separated_movies = true;

path_to_data = 'C:\path\to\data';

if run_monte_carlo_dmd
    vid = VideoReader(strcat(path_to_data, 'monte_carlo_low.mp4'));
    monte_carlo = read(vid, [1 Inf]);
    omega_threshold = 0.1;
    video_dmd(monte_carlo, 'Monte Carlo', omega_threshold, show_pca_modes, show_separated_movies)
end

if run_ski_drop_dmd
    vid = VideoReader(strcat(path_to_data, 'ski_drop_low.mp4'));
    ski_drop = read(vid, [1 Inf]);
    omega_threshold = 0.1;
    video_dmd(ski_drop, 'Ski Drop', omega_threshold, show_pca_modes, show_separated_movies)
end

function video_dmd(video, video_name, omega_threshold, show_pca_modes, show_separated_movies)

    video_height = size(video, 1);
    video_width = size(video, 2);
    video_frames = size(video, 4);
    video_pixels = video_height*video_width;

    video_2d = zeros(video_pixels, video_frames);
    for i=1:video_frames
        video_2d(:,i) = reshape(rgb2gray(video(:,:,:,i)), video_pixels, 1);
    end
    
    if show_pca_modes
    
        [~, S, ~] = svd(video_2d, 'econ');
        
        % Plotting PCA Modes
        figure()
        plot(diag(S), 'o')
        xlabel('Singular Value Index')
        ylabel('Value')
        title_text = strcat(video_name, ' Video Singular Value Spectrum');
        title(title_text)
    
    end
    
    % These values were determined from the above plots
    if strcmp(video_name, 'Monte Carlo')
        r = 175;
    elseif strcmp(video_name, 'Ski Drop')
        r = 250;
    end
    
    X1 = video_2d(:,1:end-1);
    X2 = video_2d(:,2:end);
    
    [U,S,V] = svd(X1, 'econ');
    
    Ur = U(:,1:r);
    Sr = S(1:r,1:r);
    Vr = V(:,1:r);
    
    Atilde = Ur'*X2*Vr/Sr;
    [W,D] = eig(Atilde);
    Phi = X2*Vr/Sr*W;
    
    dt = 1;
    lambda = diag(D);
    omega = log(lambda)/dt;
    
    x1 = video_2d(:,1);
    b = Phi\x1;
    
    u_modes = zeros(r, video_frames);
    t = 1:video_frames;
    for i=1:video_frames
        u_modes(:,i) = b.*exp(omega*t(i));
    end
    
    % Normal
    % u_dmd = Phi*u_modes;
    
    Phi_background = Phi(:,abs(omega)<omega_threshold);
    background_modes = u_modes(abs(omega)<omega_threshold, :);
    
    background_video = Phi_background*background_modes;

    foreground_video = video_2d - abs(background_video);
    
    [residual_row, residual_col] = find(foreground_video<0);
    
    foreground_residuals = zeros(size(foreground_video));
    for i=1:length(residual_row)
        foreground_residuals(residual_row(i), residual_col(i)) = foreground_video(residual_row(i), residual_col(i));
    end
    
    background_video = foreground_residuals + abs(background_video);
    foreground_video = foreground_video - foreground_residuals;

    if show_separated_movies
        background_video_3d = zeros(video_height, video_width, video_frames);
        max_pixel_value = max(foreground_video, [], 'all');
        for i=1:video_frames
            % max_pixel_value = max(foreground_video(:,i), [], 'all');
            background_video_3d(:,:,i) = reshape(background_video(:,i), video_height, video_width)/max_pixel_value;
        end
        
        foreground_video_3d = zeros(video_height, video_width, video_frames);
        for i=1:video_frames
            % max_pixel_value = max(foreground_video(:,i), [], 'all');
            foreground_video_3d(:,:,i) = reshape(foreground_video(:,i), video_height, video_width)/max_pixel_value;
        end
        
        figure()
        for i=1:video_frames
            subplot(1, 2, 1)
            imshow(background_video_3d(:,:,i))
            title('Background Video')
            drawnow
            subplot(1, 2, 2)
            imshow(foreground_video_3d(:,:,i))
            title('Foreground Video')
            drawnow
            sgtitle(strcat(video_name, ' Separated Videos'));
        end
    end

end