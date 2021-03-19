%% Initial Setup

close all, clear variables

run_control_svm_and_decision_tree = true;
run_fft_svm_and_decision_tree = false;
run_svd_svm_and_decision_tree = false;
show_average_fft_spectrum = false;
show_singular_value_spectrum = false;

number_of_modes = 150;

path_to_data = 'C:\path\to\data\';
[train_images, train_labels] = mnist_parse(strcat(path_to_data, 'train-images.idx3-ubyte'), ...
    strcat(path_to_data, 'train-labels.idx1-ubyte')); 
[test_images, test_labels] = mnist_parse(strcat(path_to_data, 't10k-images.idx3-ubyte'), ...
    strcat(path_to_data, 't10k-labels.idx1-ubyte')); 

% We assume that the train and test images are the same size
image_width = size(train_images, 1);
image_height = size(train_images, 2);
image_pixels = image_width*image_height;
num_train_images = size(train_images, 3);
num_test_images = size(test_images, 3);

train_matrix = zeros(image_pixels, num_train_images);
for i=1:size(train_images, 3)
    train_matrix(:, i) = reshape(train_images(:,:,i), image_pixels, 1);
end

test_matrix = zeros(image_pixels, num_test_images);
for i=1:size(test_images, 3)
    test_matrix(:,i) = reshape(test_images(:,:,i), image_pixels, 1);
end

%% Performing SVD

% We subtract out the mean image before performing our SVD
mean_image = mean(train_matrix, 2);
svd_train_matrix = train_matrix - repmat(mean_image, [1 num_train_images]);

svd_test_matrix = test_matrix - repmat(mean_image, [1 num_test_images]);

[U, S, V] = svd(train_matrix, 'econ');
Vstar = V';

if show_singular_value_spectrum
    figure()
    plot(diag(S), 'o')
    xlabel('Singular Value Index')
    ylabel('Value')
    title('Singular Value Spectrum')
end

%% Fitting SVM and Decision Tree on SVD Data

if run_svd_svm_and_decision_tree

    projected_train_images = svd_train_matrix'*Vstar(:,1:number_of_modes);
    projected_test_images = svd_test_matrix'*Vstar(:,1:number_of_modes);
    scaled_projected_train_images = projected_train_images./max(projected_train_images, [], 'all');
    scaled_projected_test_images = projected_test_images./max(projected_test_images, [], 'all');

    % SVM

    SVM_Models = cell(10,1);
    classes = 0:9;

    fprintf('\n')
    for i=classes
        SVM_Models{i+1} = fitcsvm(scaled_projected_train_images, train_labels==i, 'ClassNames', [false true]);

        % We include the below text as a measure of progress
        fprintf('Fit SVD SVM for Digit %g\n', i)
    end

    test_scores = zeros(size(scaled_projected_test_images, 1), length(classes));
    train_scores = zeros(size(scaled_projected_train_images, 1), length(classes));
    for i=classes
        [~, score] = predict(SVM_Models{i+1}, scaled_projected_test_images);
        test_scores(:,i+1) = score(:,2);
        [~, score] = predict(SVM_Models{i+1}, scaled_projected_train_images);
        train_scores(:,i+1) = score(:,2);
    end

    [~, svm_test_classifications] = max(test_scores, [], 2);
    svm_test_classifications = svm_test_classifications-1; % To account for MATLAB being 1-indexed

    [~, svm_train_classifications] = max(train_scores, [], 2);
    svm_train_classifications = svm_train_classifications-1; % To account for MATLAB being 1-indexed

    svm_test_accuracy = nnz(svm_test_classifications==test_labels)/length(test_labels);
    svm_train_accuracy = nnz(svm_train_classifications==train_labels)/length(train_labels);

    fprintf('\nSVD SVM Results\n')
    fprintf('On all 10 digits, our SVM was %g%% accurate on the SVD test data\n', svm_test_accuracy*100)
    fprintf('On all 10 digits, our SVM was %g%% accurate on the SVD training data\n', svm_train_accuracy*100)

    % Decision Tree

    tree = fitctree(scaled_projected_train_images, train_labels);
    tree_test_classifications = predict(tree, scaled_projected_test_images);
    tree_train_classifications = predict(tree, scaled_projected_train_images);

    tree_test_accuracy = nnz(tree_test_classifications==test_labels)/length(test_labels);
    tree_train_accuracy = nnz(tree_train_classifications==train_labels)/length(train_labels);

    fprintf('\nSVD Decision Tree Results:\n')
    fprintf('On all 10 digits our decision tree was %g%% accurate on the test data\n', tree_test_accuracy*100)
    fprintf('On all 10 digits our decision tree was %g%% accurate on the training data\n', tree_train_accuracy*100)

end

%% Performing FFT

fft_train_images = abs(fft(train_matrix));
fft_test_images = abs(fft(test_matrix));

if show_average_fft_spectrum
    
    average_fft_spectrum = mean(fft_train_images, 2);

    plot(average_fft_spectrum)
    xlabel('Index')
    ylabel('Spectral Power Density')
    title('Average FFT Spectrum')

end

%% FFT SVM and Decision Tree

if run_fft_svm_and_decision_tree

    fft_projected_train_images = fft_train_images(1:number_of_modes,:);
    fft_projected_test_images = fft_test_images(1:number_of_modes,:);
    scaled_fft_projected_train_images = fft_projected_train_images./max(fft_projected_train_images, [], 'all');
    scaled_fft_projected_test_images = fft_projected_test_images./max(fft_projected_test_images, [], 'all');
    scaled_fft_projected_train_images = scaled_fft_projected_train_images';
    scaled_fft_projected_test_images = scaled_fft_projected_test_images';

    % SVM

    SVM_Models = cell(10,1);
    classes = 0:9;

    fprintf('\n')
    for i=classes
        SVM_Models{i+1} = fitcsvm(scaled_fft_projected_train_images, train_labels==i, 'ClassNames', [false true]);

        % We include the below text as a measure of progress
        fprintf('Fit FFT SVM for Digit %g\n', i)
    end

    test_scores = zeros(size(scaled_fft_projected_test_images, 1), length(classes));
    train_scores = zeros(size(scaled_fft_projected_train_images, 1), length(classes));
    for i=classes
        [~, score] = predict(SVM_Models{i+1}, scaled_fft_projected_test_images);
        test_scores(:,i+1) = score(:,2);
        [~, score] = predict(SVM_Models{i+1}, scaled_fft_projected_train_images);
        train_scores(:,i+1) = score(:,2);
    end

    [~, svm_test_classifications] = max(test_scores, [], 2);
    svm_test_classifications = svm_test_classifications-1; % To account for MATLAB being 1-indexed

    [~, svm_train_classifications] = max(train_scores, [], 2);
    svm_train_classifications = svm_train_classifications-1; % To account for MATLAB being 1-indexed

    svm_test_accuracy = nnz(svm_test_classifications==test_labels)/length(test_labels);
    svm_train_accuracy = nnz(svm_train_classifications==train_labels)/length(train_labels);

    fprintf('\nFFT SVM Results\n')
    fprintf('On all 10 digits, our SVM was %g%% accurate on the test data\n', svm_test_accuracy*100)
    fprintf('On all 10 digits, our SVM was %g%% accurate on the training data\n', svm_train_accuracy*100)

    % Decision Tree

    tree = fitctree(scaled_fft_projected_train_images, train_labels);
    tree_test_classifications = predict(tree, scaled_fft_projected_test_images);
    tree_train_classifications = predict(tree, scaled_fft_projected_train_images);

    tree_test_accuracy = nnz(tree_test_classifications==test_labels)/length(test_labels);
    tree_train_accuracy = nnz(tree_train_classifications==train_labels)/length(train_labels);

    fprintf('\nFFT Decision Tree Results:\n')
    fprintf('On all 10 digits our decision tree was %g%% accurate on the test data\n', tree_test_accuracy*100)
    fprintf('On all 10 digits our decision tree was %g%% accurate on the training data\n', tree_train_accuracy*100)

end

%% Control SVM and Decision Tree

% Not used, takes an infeasible amount of time to run

if run_control_svm_and_decision_tree
    
    control_train_matrix = train_matrix';
    control_test_matrix = test_matrix';

    % SVM

    SVM_Models = cell(10,1);
    classes = 0:9;

    fprintf('\n')
    for i=classes
        SVM_Models{i+1} = fitcsvm(control_train_matrix, train_labels==i, 'ClassNames', [false true]);

        % We include the below text as a measure of progress
        fprintf('Fit Control SVM for Digit %g\n', i)
    end

    test_scores = zeros(size(control_test_matrix, 1), length(classes));
    train_scores = zeros(size(control_train_matrix, 1), length(classes));
    for i=classes
        [~, score] = predict(SVM_Models{i+1}, control_test_matrix);
        test_scores(:,i+1) = score(:,2);
        [~, score] = predict(SVM_Models{i+1}, control_train_matrix);
        train_scores(:,i+1) = score(:,2);
    end

    [~, svm_test_classifications] = max(test_scores, [], 2);
    svm_test_classifications = svm_test_classifications-1; % To account for MATLAB being 1-indexed

    [~, svm_train_classifications] = max(train_scores, [], 2);
    svm_train_classifications = svm_train_classifications-1; % To account for MATLAB being 1-indexed

    svm_test_accuracy = nnz(svm_test_classifications==test_labels)/length(test_labels);
    svm_train_accuracy = nnz(svm_train_classifications==train_labels)/length(train_labels);

    fprintf('\nControl SVM Results\n')
    fprintf('On all 10 digits, our SVM was %g%% accurate on the test data\n', svm_test_accuracy*100)
    fprintf('On all 10 digits, our SVM was %g%% accurate on the training data\n', svm_train_accuracy*100)

    % Decision Tree

    tree = fitctree(control_train_matrix, train_labels);
    tree_test_classifications = predict(tree, control_test_matrix);
    tree_train_classifications = predict(tree, control_train_matrix);

    tree_test_accuracy = nnz(tree_test_classifications==test_labels)/length(test_labels);
    tree_train_accuracy = nnz(tree_train_classifications==train_labels)/length(train_labels);

    fprintf('\nControl Decision Tree Results:\n')
    fprintf('On all 10 digits our decision tree was %g%% accurate on the test data\n', tree_test_accuracy*100)
    fprintf('On all 10 digits our decision tree was %g%% accurate on the training data\n', tree_train_accuracy*100)

end