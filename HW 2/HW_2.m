%% Provided Code
close all; clear variables;

play_and_show_floyd = false;
play_and_show_GNR = false;
play_floyd_filtered_bass = false;
play_floyd_filtered_guitar = false;
play_gnr_filtered = false;
show_floyd_frequencies = false;
show_gnr_frequencies = false;
show_floyd_filtered_bass_frequencies = false;
show_floyd_filtered_guitar_frequencies = false;
show_gnr_filtered_frequencies = false;
show_floyd_bass_sheet_music = true;
show_floyd_guitar_sheet_music = true;
show_gnr_guitar_sheet_music = true;
% A sliding filter on our filtered songs can be shown through turning on
% show_sliding_filter in the gaussian_filter helper function

if play_and_show_floyd
    % Provided Code run on Floyd.m4a
    figure()
    [y, Fs] = audioread('Floyd.m4a');
    tr.gnr = length(y)/Fs; % record time in seconds
    plot((1:length(y))/Fs, y);
    xlabel('Time [sec]'); ylabel('Amplitude');
    title("Comfortably Numb");
    p8 = audioplayer(y, Fs); playblocking(p8);
end

if play_and_show_GNR
    % Provided Code run on GNR.m4a
    figure()
    [y, Fs] = audioread('GNR.m4a');
    tr.gnr = length(y)/Fs; % record time in seconds
    plot((1:length(y))/Fs, y);
    xlabel('Time [sec]'); ylabel('Amplitude');
    title("Sweet Child O' Mine"); % I fixed the parantheses
    p8 = audioplayer(y, Fs); playblocking(p8);
end

%% Creating Sheet Music for Pink Floyd and Guns and Roses

floyd_L = 60; % The Floyd clip is 60 seconds long
gnr_L = 14; % The GNR clip is 14 seconds long

% Identifying the Guitar in Guns and Roses and Bass in Pink Floyd

[floyd, floyd_rate] = audioread('Floyd.m4a');
[gnr, gnr_rate] = audioread('GNR.m4a');

floyd_transformed = fft(floyd);
gnr_transformed = fft(gnr);

% MATLAB's fftshift adds the extra value in vectors with an odd length
% to the second half of the vector, so we do the same
% We also scale the k vectors to be in ordinary frequency Hz (as opposed to
% angular frequency, which is also technically in Hz)
floyd_k = (1/floyd_L)*[0:(length(floyd)/2 - 1) -round(length(floyd)/2):-1];
floyd_kshift = fftshift(floyd_k);

gnr_k = (1/gnr_L)*[0:(length(gnr)/2 - 1) -length(gnr)/2:-1];
gnr_kshift = fftshift(gnr_k); 

if show_floyd_frequencies
    figure()
    plot(floyd_kshift, abs(fftshift(floyd_transformed)))
    title('Comfortably Numb Frequency Domain')
    xlabel('Frequency (Hz)'); ylabel('Power Spectral Density');
end
if show_gnr_frequencies
    figure()
    plot(gnr_kshift, abs(fftshift(gnr_transformed)))
    title("Sweet Child O' Mine Frequency Domain")
    xlabel('Frequency (Hz)'); ylabel('Power Spectral Density');
end

% Looking at the above plots the bass of the Pink Floyd clip seems to be
% centered around +-125 Hz
% Looking at the above plots, the guitar of the Guns and Roses clip seems
% to be centered around +- 3750 Hz

% Filtering Around the Frequencies Identified

% Testing reveals that these filters need to be wider than might be 
% expected
floyd_bass_filter_frequency = 125; % Hz
floyd_bass_filter_width = 3000; % Hz
floyd_center1 = find(floyd_k==floyd_bass_filter_frequency);
floyd_center2 = find(floyd_k==-floyd_bass_filter_frequency);
floyd_transformed_filtered_bass = gaussian_filter(floyd_transformed, floyd_center1, floyd_bass_filter_width) + ...
    gaussian_filter(floyd_transformed, floyd_center2, floyd_bass_filter_width);
floyd_bass = ifft(floyd_transformed_filtered_bass);

gnr_guitar_filter_frequency = 3750; % Hz
gnr_guitar_filter_width = 16000; % Hz
gnr_center1 = find(gnr_k==gnr_guitar_filter_frequency);
gnr_center2 = find(gnr_k==-gnr_guitar_filter_frequency);
gnr_transformed_filtered_guitar = gaussian_filter(gnr_transformed, gnr_center1, gnr_guitar_filter_width) + ... 
    gaussian_filter(gnr_transformed, gnr_center2, gnr_guitar_filter_width);
gnr_guitar = ifft(gnr_transformed_filtered_guitar);

if show_floyd_filtered_bass_frequencies
    figure()
    plot(floyd_kshift, abs(fftshift(floyd_transformed_filtered_bass)))
    title('Comfortably Numb Frequency Domain (Filtered Bass)')
    xlabel('Frequency (Hz)'); ylabel('Power Spectral Density');
end
if show_gnr_filtered_frequencies
    figure()
    plot(gnr_kshift, abs(fftshift(gnr_transformed_filtered_guitar)))
    title("Sweet Child O' Mine Frequency Domain (Filtered Guitar)")
    xlabel('Frequency (Hz)'); ylabel('Power Spectral Density');
end

if play_floyd_filtered_bass
    p8 = audioplayer(floyd_bass, floyd_rate); playblocking(p8);
end
if play_gnr_filtered
    p8 = audioplayer(gnr_guitar, gnr_rate); playblocking(p8);
end

% Gabor Filtering The Filtered Data

% Comfortably Numb is 127 BPM (beats per minute) and the clip is floyd_L 
% seconds long (60)
floyd_beats = 127/60*floyd_L;
floyd_indices_per_beat = length(floyd_k)/floyd_beats;
% We need a relatively long filter to pick up bass, so we filter with a
% Full Width at one Tenth Maximum (FWTM) of 1 bars
floyd_bass_filter_width = (4*floyd_indices_per_beat)/(2*sqrt(2*log(10)));
floyd_bass_spectrogram = zeros(length(floyd_k), floyd_beats);
for beat=1:floyd_beats
    floyd_beat_filtered = gaussian_filter(floyd_bass, beat*floyd_indices_per_beat, floyd_bass_filter_width);
    floyd_bass_spectrogram(:, beat) = fft(floyd_beat_filtered);
end

% We want one filter per beat. Sweet Child O Mine is 127 BPM (beats per 
% minute) and the clip is gnr_L seconds long (14)
gnr_beats = 123/60*gnr_L;
% We don't need as long of a filter to pick up guitar, so we filter with a
% FWTM of 1/2 beat
gnr_indices_per_beat = length(gnr_k)/floyd_beats;
gnr_guitar_filter_width = (1/2*gnr_indices_per_beat)/(2*sqrt(2*log(10)));
gnr_guitar_spectrogram = zeros(length(gnr_k), floor(gnr_beats));
% The Guns and Roses clip is 14.7 beats according to the information above,
% but we filter only for 14 beats
for beat=1:floor(gnr_beats) 
    gnr_beat_filtered = gaussian_filter(gnr_guitar, beat*gnr_indices_per_beat, gnr_guitar_filter_width);
    gnr_guitar_spectrogram(:, beat) = fft(gnr_beat_filtered);
end

% Plotting Spectrograms (Sheet Music)

if show_floyd_bass_sheet_music
    % We only want to look at the relevant frequencies, which are around
    % 125 Hz for the bass in this clip
    floyd_bass_low_frequency = 70; % Hz
    floyd_bass_high_frequency = 135; % Hz
    floyd_bass_low_index = find(floyd_k==floyd_bass_low_frequency);
    floyd_bass_high_index = find(floyd_k==floyd_bass_high_frequency);
    figure()
    pcolor(1:floyd_beats, floyd_k(floyd_bass_low_index:floyd_bass_high_index),...
        abs(floyd_bass_spectrogram(floyd_bass_low_index:floyd_bass_high_index, :)))
    shading interp
    colormap(flipud(gray))
    title('Comfortably Numb Bass Sheet Music')
    xlabel('Beat'); ylabel('Frequency (Hz)')
    yyaxis right
    ylim([floyd_bass_low_frequency floyd_bass_high_frequency])
    % We try to tick only notes that are played to avoid clutter
    yticks([82 87 92 98 110 124])
    yticklabels({'E', 'F', 'F#', 'G', 'A', 'B'})
    ylabel('Notes')
end

if show_gnr_guitar_sheet_music
    % We only want to look at the relevant frequencies, which are around 
    gnr_guitar_low_frequency = 2000;
    gnr_guitar_high_frequency = 5000;
    gnr_guitar_low_index = find(gnr_k==gnr_guitar_low_frequency);
    gnr_guitar_high_index = find(gnr_k==gnr_guitar_high_frequency);
    figure()
    pcolor(1:floor(gnr_beats), gnr_k(gnr_guitar_low_index:gnr_guitar_high_index),...
        abs(gnr_guitar_spectrogram(gnr_guitar_low_index:gnr_guitar_high_index, :)))
    shading interp
    colormap(flipud(gray))
    title("Sweet Child O' Mine Guitar Sheet Music")
    xlabel('Beat'), ylabel('Frequency (Hz)')
    yyaxis right
    ylim([gnr_guitar_low_frequency gnr_guitar_high_frequency])
    % We try to tick only notes that are played to avoid clutter
    yticks([2093 2489 2794 2960 3136 3322 3520 3730 3950 4185 4435 4698 4978])
    yticklabels({'C', 'D#', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#'})
    ylabel('Notes')
end


%% Isolating the Bass in Comfortably Numb

% We've already done this as part of part 1, the result is
% floyd_bass

%% Isolating the Guitar in Comfortably Numb

% We use the same process as in first part of part 1, but with a new filter
% There's a background around the low end of the guitar solo's frequency
% range, which makes this more difficult (in addition to this being faster
% notes)

floyd_guitar_filter_frequency = 2100; % Hz
floyd_guitar_filter_width = 12500; % Hz
floyd2_center1 = find(floyd_k==floyd_guitar_filter_frequency);
floyd2_center2 = find(floyd_k==-floyd_guitar_filter_frequency);
floyd_transformed_filtered_guitar = gaussian_filter(floyd_transformed, floyd2_center1, floyd_guitar_filter_width) + ...
    gaussian_filter(floyd_transformed, floyd2_center2, floyd_guitar_filter_width);
floyd_guitar = ifft(floyd_transformed_filtered_guitar);

if show_floyd_filtered_guitar_frequencies
    figure()
    plot(floyd_kshift, abs(fftshift(floyd_transformed_filtered_guitar)))
    title('Comfortably Numb Frequency Domain (Filtered Guitar)')
    xlabel('Frequency (Hz)'); ylabel('Power Spectral Density');
end

if play_floyd_filtered_guitar
    p8 = audioplayer(floyd_guitar, floyd_rate); playblocking(p8);
end

% Gabor Filtering
% We don't need as long of a filter to pick up guitar, so we filter with a
% FWTM of 2 beats
floyd_guitar_filter_width = (2*floyd_indices_per_beat)/(2*sqrt(2*log(10)));
floyd_guitar_spectrogram = zeros(length(floyd_k), floyd_beats);
for beat=1:floyd_beats
    floyd_beat_filtered = gaussian_filter(floyd_guitar, beat*floyd_indices_per_beat, floyd_guitar_filter_width);
    floyd_guitar_spectrogram(:, beat) = fft(floyd_beat_filtered);
end

if show_floyd_guitar_sheet_music
    floyd_guitar_low_frequency = 1700;
    floyd_guitar_high_frequency = 2550;
    floyd_guitar_low_index = find(floyd_k==floyd_guitar_low_frequency);
    floyd_guitar_high_index = find(floyd_k==floyd_guitar_high_frequency);
    figure()
    pcolor(1:floyd_beats, floyd_k(floyd_guitar_low_index:floyd_guitar_high_index),...
        abs(floyd_guitar_spectrogram(floyd_guitar_low_index:floyd_guitar_high_index, :)))
    shading interp
    colormap(flipud(gray))
    title('Comfortably Numb Guitar Sheet Music')
    xlabel('Beat'), ylabel('Frequency (Hz)')
    yyaxis right
    ylim([floyd_guitar_low_frequency floyd_guitar_high_frequency])
    % We try to tick only notes that are played to avoid clutter
    yticks([1760 1865 1975 2093 2218 2349 2489])
    yticklabels({'A', 'A#', 'B', 'C', 'C#', 'D', 'D#'})
    ylabel('Notes')
end

%% Helper Functions

function vec = gaussian_filter(x, center, width)
% x is the vector the Gaussian filter is being applied to
% center is the index of the center of the Gaussian filter (a scalar)
% width is the standard deviation (in terms of indices) of the Gaussian
% filter (a scalar)
    scale = 1:length(x);
    filter = exp(-(scale-center).^2/(2*width^2));
    vec = x.*transpose(filter);
    show_sliding_filter = false; 
    % If this is on, other plots should be off since this clears the figure
    % This has to be here due to MATLAB's variable scoping
    if show_sliding_filter
        plot(scale, filter)
        yyaxis right
        plot(scale, x)
        drawnow
        pause(0.1)
        clf
    end
end