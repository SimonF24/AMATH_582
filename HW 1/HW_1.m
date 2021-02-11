%% Initial Setup

clear variables; close all; clc
% clearing variables instead of all for performance reasons

show_provided_isosurface = false;
show_average_spectrum_isosurface = false;
show_filter_isosurface = false;
show_path_plot = false;
show_aircraft_path = false;

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix named subdata
% This matrix is too big to upload to GitHub

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

if show_provided_isosurface
    figure()
    for j=1:49
        Un(:,:,:) = reshape(subdata(:,j),n,n,n);
        M = max(abs(Un),[],'all');
        clf, isosurface(X,Y,Z,abs(Un)/M,0.7)
        % clf clears the plot without creating a new window
        axis([-20 20 -20 20 -20 20]), grid on, drawnow
        pause(1)
    end
end

%% Averaging the Spectrum

% Taking the Fourier Transform of the data
subdata_transformed = zeros(n, n, n, 49);
subdata_transformed_ave = zeros(n, n, n);
if show_average_spectrum_isosurface
    figure()
end
for j=1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    subdata_transformed(:,:,:,j) = fftn(Un);
    subdata_transformed_ave = subdata_transformed_ave + subdata_transformed(:,:,:,j);
    if show_average_spectrum_isosurface
        clf, isosurface(Kx, Ky, Kz, abs(subdata_transformed_ave)/max(abs(subdata_transformed_ave), [], 'all'), 0.7)
        xlabel('kx (unshifted)'), ylabel('ky (unshifted)'), zlabel('kz (unshifted)')
        % axis([-10 10 -10 10 -10 10])
        grid on, drawnow
        pause(1)
    end
end

% Inspection of the above isosurface gives us the center frequency as 
% located at (-5, 3, -8) in unshifted frequency space ((5, -7, 2) in 
% shifted frequency space). This is can be seen with the axis command 
% uncommented but is most resolved with the axis command uncommented, which
% allows MATLAB to "zoom in" on the peak resulting from the center 
% frequency
 
%% Filtering the Data Around the Center Frequency

% We use a 3D Gaussian with variance 1 for our filter
filter = exp(-(X+5).^2./2).*exp(-(Y-3).^2./2).*exp(-(Z+8).^2./2);
if show_filter_isosurface
    figure()
    clf, isosurface(X, Y, Z, filter, 0.7);
    xlabel('x'), ylabel('y'), zlabel('z')
    grid on, drawnow
end

% Applying the filter to our data
full_filter = repmat(filter, 1, 1, 1, 49);
subdata_transformed_filtered = subdata_transformed.*full_filter;

% Taking the inverse Fourier Transform of our data
subdata_filtered = zeros(size(subdata_transformed_filtered));
path_3d = zeros(49, 3);
for i=1:49
   subdata_filtered(:,:,:,i) = ifftn(subdata_transformed_filtered(:,:,:,i));
   [~, linear_index] = max(subdata_filtered(:,:,:,i), [], 'all', 'linear');
   [x_index, y_index, z_index] = ind2sub([n, n, n], linear_index);
   path_3d(i, :) = [x_index, y_index, z_index];
   % Plotting this shows nothing
end

if show_path_plot
    figure()
    plot3(path_3d(:, 1), path_3d(:, 2), path_3d(:, 3))
    xlabel('x'), ylabel('y'), zlabel('z')
end


%% Where to Send our P-8 Orion Sub-Tracking Aircraft

% This is just x and y components of the path we've already found

aircraft_path = path_3d(:,1:2);

if show_aircraft_path
    figure()
    plot(aircraft_path(:,1), aircraft_path(:,2))
    xlabel('x'), ylabel('y')
end
