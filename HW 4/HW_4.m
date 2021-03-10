%% Initial Setup

close all, clear variables

selected_v_modes = [2 3 5];
image_reconstruction_modes = 100;
run_2_digit_linear_discriminant = false;
run_3_digit_linear_discriminant = false;
run_easiest_hardest_digits = false;
run_method_comparison = false;
run_svm_and_decision_tree = true;
show_image_reconstruction = false;
show_singular_value_spectrum = false;
show_v_mode_projection = false;

path_to_data = 'C:\path\to\data';
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

%% Performing SVD

train_svd_matrix = zeros(image_pixels, num_train_images);
for i=1:size(train_images, 3)
    train_svd_matrix(:, i) = reshape(train_images(:,:,i), image_pixels, 1);
end

% We subtract out the mean image before performing our SVD
mean_image = mean(train_svd_matrix, 2);
train_svd_matrix = train_svd_matrix - repmat(mean_image, [1 num_train_images]);

[U, S, Vstar] = svd(train_svd_matrix, 'econ');
V = Vstar';

%% Preparing Test Data

test_matrix = zeros(image_pixels, num_test_images);
for i=1:size(test_images, 3)
    test_matrix(:,i) = reshape(test_images(:,:,i), image_pixels, 1);
end

test_matrix = test_matrix - repmat(mean_image, [1 num_test_images]);

%% Plotting Singular Value Spectrum

if show_singular_value_spectrum
    figure()
    plot(diag(S), 'o')
    xlabel('Singular Value Index')
    ylabel('Value')
    title('Singular Value Spectrum')
end

%% Reconstructing Images from SVD Modes

if show_image_reconstruction
    % We show the first three training images and their reconstructions
    % with the selected number of SVD modes
    
    figure()
    subplot(2, 3, 1)
    imshow(train_images(:,:,1))
    subplot(2, 3, 2)
    imshow(train_images(:,:,2))
    subplot(2, 3, 3)
    imshow(train_images(:,:,3))
    
    reconstructed_images = U(:,1:image_reconstruction_modes)*...
        S(1:image_reconstruction_modes, 1:image_reconstruction_modes)*...
        Vstar(1:3, 1:image_reconstruction_modes)';
    
    % We don't add back the mean image to this since it is largely
    % unnecessary 
    %reconstructed_images = reconstructed_images + repmat(mean_image, [1,3]);
    
    subplot(2, 3, 4)
    imshow(reshape(reconstructed_images(:,1), image_width, image_height))
    subplot(2, 3, 5)
    imshow(reshape(reconstructed_images(:,2), image_width, image_height))
    subplot(2, 3, 6)
    imshow(reshape(reconstructed_images(:,3), image_width, image_height))
    sgtitle('Original and Reconstructed Images')
    
end

% Checking different values and examining the principal component spectrum 
% leads us to the conclusion that around 100 mode are needed for tolerable 
% image reconstruction.
% The U matrix is a transformation from the original space of the matrix A
% to the principal component space. The S matrix represents the stretching
% of the transformation in its principal component space. The Vstar matrix
% represents a transformation from the principal component space to the 
% final space of the matrix A (so the matrix V represents a transformation
% from the final space of matrix A to the principal component space)

%% Projecting onto V-Modes

if show_v_mode_projection

    colors=[1 0 0;
            0 1 0;
            0 0 1;
            1 1 0;
            0 1 1;
            1 0 1;
            0 0 0;
            0.5 0.5 1;
            1 0.5 0.5;
            0.5 1 0.5];
    
    projected_images = train_svd_matrix'*V(:, selected_v_modes);
    figure()
    hold on
    for i=0:9
        relevant_images = projected_images(train_labels == i,:);
        plot3(relevant_images(:,1), relevant_images(:,2), relevant_images(:,3), 'o', 'Color', colors(i+1,:))
    end
    title_text = sprintf('Images Projected onto V-Modes %g, %g, and %g', selected_v_modes);
    title(title_text)
    xlabel_text = sprintf('V Mode %g', selected_v_modes(1));
    xlabel(xlabel_text)
    ylabel_text = sprintf('V Mode %g', selected_v_modes(2));
    ylabel(ylabel_text)
    zlabel_text = sprintf('V Mode %g', selected_v_modes(3));
    zlabel(zlabel_text)
    legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    % Although a bit difficult to see, clumping is discernable in this plot
end

%% Linear Classifier to Discern 2 Digits

% We choose 0 and 1 as our two digits to discern

% We use 100 modes for V since that was our above determined value of
% significant modes

if run_2_digit_linear_discriminant
    
    projected_train_images = train_svd_matrix'*V(:,1:100);
    projected_test_images = test_matrix'*V(:,1:100);
    train_images_0 = projected_train_images(train_labels==0, :);
    train_images_1 = projected_train_images(train_labels==1, :);
    test_images_0 = projected_test_images(test_labels==0, :);
    test_images_1 = projected_test_images(test_labels==1, :);
    
    sample = [test_images_0; test_images_1];
    training = [train_images_0; train_images_1];
    group = [zeros(size(train_images_0, 1), 1); ones(size(train_images_1, 1), 1)];

    class = classify(sample, training, group);
    
    num_zero_test_images = size(test_images_0,1);
    num_one_test_images = size(test_images_1,1);
    
    zero_accuracy = nnz(class(1:num_zero_test_images)==0)/num_zero_test_images;
    one_accuracy = nnz(class(end-num_one_test_images:end)==1)/length(test_images_1);
    overall_accuracy = nnz([class(1:num_zero_test_images)==0; class(end-num_one_test_images:end)==1])...
        /(num_zero_test_images+num_one_test_images);
    
    fprintf('\n2 Digit Linear Discriminant Results:\n')
    fprintf('Accuracy on zeros: %g%%\n', zero_accuracy*100)
    fprintf('Accuracy on ones: %g%%\n', one_accuracy*100)
    fprintf('Overall accuracy: %g%%\n', overall_accuracy*100)

    % The first 980 elements of this bar graph should be 0 and the last
    % 1135 elements should be 1
    figure()
    bar(class)
    title('2 Digit Linear Discriminant Results')
    xlabel('Test Image Number')
    ylabel('Classification (0 or 1)')

end

% This linear classifier does exceptionally well, with a 99.86% overall
% accuracy

%% Linear Classifier to Discern 3 Digits

% We choose 0, 1 and 2 to be our digits to discern

% We use 100 modes for V since that was our above determined value of
% significant modes

if run_3_digit_linear_discriminant
    
    projected_train_images = train_svd_matrix'*V(:,1:100);
    projected_test_images = test_matrix'*V(:,1:100);
    train_images_0 = projected_train_images(train_labels==0, :);
    train_images_1 = projected_train_images(train_labels==1, :);
    train_images_2 = projected_train_images(train_labels==2, :);
    test_images_0 = projected_test_images(test_labels==0, :);
    test_images_1 = projected_test_images(test_labels==1, :);
    test_images_2 = projected_test_images(test_labels==2, :);
    
    sample = [test_images_0; test_images_1; test_images_2];
    training = [train_images_0; train_images_1; train_images_2];
    group = [zeros(size(train_images_0, 1), 1); ones(size(train_images_1, 1), 1); 2*ones(size(train_images_2, 1), 1)];

    class = classify(sample, training, group);
    
    num_zero_test_images = size(test_images_0,1);
    num_one_test_images = size(test_images_1,1);
    num_two_test_images = size(test_images_2,1);
    
    zero_accuracy = nnz(class(1:num_zero_test_images)==0)/num_zero_test_images;
    one_accuracy = nnz(class(num_zero_test_images+1:num_zero_test_images+1+num_one_test_images)==1)/length(test_images_1);
    two_accuracy = nnz(class(end-num_two_test_images:end)==2)/num_two_test_images;
    overall_accuracy = nnz([class(1:num_zero_test_images)==0; ...
        class(num_zero_test_images+1:num_zero_test_images+1+num_one_test_images)==1; ...
        class(end-num_two_test_images:end)==2])...
        /(num_zero_test_images+num_one_test_images+num_two_test_images);
    
    fprintf('\n3 Digit Linear Discriminant Results:\n') 
    fprintf('Accuracy on zeros: %g%%\n', zero_accuracy*100)
    fprintf('Accuracy on ones: %g%%\n', one_accuracy*100)
    fprintf('Accuracy on twos: %g%%\n', two_accuracy*100)
    fprintf('Overall accuracy: %g%%\n', overall_accuracy*100)

    % The first 980 elements of this bar graph should be 0, the next 1135
    % elements should be 1, and the last 1032 elements should be 2
    figure()
    bar(class)
    title('3 Digit Linear Discriminant Results')
    xlabel('Test Image Number')
    ylabel('Classification (0, 1, or 2)')
    
    % This classifier also does very well, with a 96.25% overall accuracy
    
end

%% Finding Easiest and Hardest Digits to Classify

if run_easiest_hardest_digits
    
    permutations = nchoosek(0:9, 2);
    
    % We continue to use 100 V modes as that was the selected number
    % earlier and we are getting good results

    projected_train_images = train_svd_matrix'*V(:,1:100);
    projected_test_images = test_matrix'*V(:,1:100);

    accuracies = zeros(size(permutations,1),1);
    for i=1:size(permutations,1)
        digits = permutations(i,:);
        digit_one = digits(1);
        digit_two = digits(2);

        digit_one_train_images = projected_train_images(train_labels==digit_one,:);
        digit_two_train_images = projected_train_images(train_labels==digit_two,:);
        digit_one_test_images = projected_test_images(test_labels==digit_one,:);
        digit_two_test_images = projected_test_images(test_labels==digit_two,:);
        
        sample = [digit_one_test_images; digit_two_test_images];
        training = [digit_one_train_images; digit_two_train_images];
        group = [digit_one*ones(size(digit_one_train_images,1), 1); digit_two*ones(size(digit_two_train_images, 1), 1)];
        
        class = classify(sample, training, group);
        
        num_digit_one_test_images = size(digit_one_test_images, 1);
        num_digit_two_test_images = size(digit_two_test_images, 1);
        
        overall_accuracy = nnz([class(1:num_digit_one_test_images)==digit_one; ...
            class(end-num_digit_two_test_images:end)==digit_two])...
        /(num_digit_one_test_images+num_digit_two_test_images);
    
        accuracies(i) = overall_accuracy;
    end
    
    [best_classification, best_index] = max(accuracies);
    [worst_classification, worst_index] = min(accuracies);
    
    easiest_digits = permutations(best_index,:);
    hardest_digits = permutations(worst_index,:);
    
    fprintf('\nEasiest and Hardest Digit Results:\n')
    fprintf('The easiest digits to classify are %g and %g\n', easiest_digits(1), easiest_digits(2))
    fprintf('These digits can be classified with an accuracy of %g%%\n', best_classification*100)
    fprintf('The hardest digits to classify are %g and %g\n', hardest_digits(1), hardest_digits(2))
    fprintf('These digits can be classified with an accuracy of %g%%\n', worst_classification*100)
    
    % The easiest digits are 0 and 1 (99.85%) and the hardest digits are 5
    % and 8 (92.87%)
    
end

%% Fitting SVM and Decision Tree

if run_svm_and_decision_tree
    
    % We continue to use 100 V modes as that was the selected number
    % earlier and we are getting good results

    projected_train_images = train_svd_matrix'*V(:,1:100);
    projected_test_images = test_matrix'*V(:,1:100);
    scaled_projected_train_images = projected_train_images./max(projected_train_images, [], 'all');
    scaled_projected_test_images = projected_test_images./max(projected_test_images, [], 'all');
    
    % SVM
    
    SVM_Models = cell(10,1);
    classes = 0:9;
    
    fprintf('\n')
    for i=classes
        SVM_Models{i+1} = fitcsvm(scaled_projected_train_images, train_labels==i, 'ClassNames', [false true]);
        
        % We include the below text as a measure of progress
        fprintf('Fit SVM for Digit %g\n', i)
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
    
    fprintf('\n10 Digit SVM Results\n')
    fprintf('On all 10 digits, our SVM was %g%% accurate on the test data\n', svm_test_accuracy*100)
    fprintf('On all 10 digits, our SVM was %g%% accurate on the training data\n', svm_train_accuracy*100)
    
    % Decision Tree
    
    tree = fitctree(scaled_projected_train_images, train_labels);
    tree_test_classifications = predict(tree, scaled_projected_test_images);
    tree_train_classifications = predict(tree, scaled_projected_train_images);
    
    tree_test_accuracy = nnz(tree_test_classifications==test_labels)/length(test_labels);
    tree_train_accuracy = nnz(tree_train_classifications==train_labels)/length(train_labels);
    
    fprintf('\n10 Digit Decision Tree Results:\n')
    fprintf('On all 10 digits our decision tree was %g%% accurate on the test data\n', tree_test_accuracy*100)
    fprintf('On all 10 digits our decision tree was %g%% accurate on the training data\n', tree_train_accuracy*100)
    
end

%% Comparison Between LDA, SVM, and Decision Tree on Hardest and Easiest Digits

if run_method_comparison
    
    % We continue to use 100 V modes as that was the selected number
    % earlier and we are getting good results
    
    projected_train_images = train_svd_matrix'*V(:,1:100);
    projected_test_images = test_matrix'*V(:,1:100);
    scaled_projected_train_images = projected_train_images./max(projected_train_images, [], 'all');
    scaled_projected_test_images = projected_test_images./max(projected_test_images, [], 'all');
    
    % We begin with the easiest digits to separate, which we determined
    % earlier to be 0 and 1
    
    train_images_0 = scaled_projected_train_images(train_labels==0,:);
    train_images_1 = scaled_projected_train_images(train_labels==1,:);
    test_images_0 = scaled_projected_test_images(test_labels==0,:);
    test_images_1 = scaled_projected_test_images(test_labels==1,:);
    
    easiest_test_images = [test_images_0; test_images_1];
    easiest_test_labels = [zeros(size(test_images_0,1), 1); ones(size(test_images_1, 1), 1)];
    easiest_train_images = [train_images_0; train_images_1];
    easiest_train_labels = [zeros(size(train_images_0,1), 1); ones(size(train_images_1, 1), 1)];
    
    % LDA
    
    % We previously used unscaled data with the LDA, but being linear, this
    % should make a difference in which digits are hardest or easiest
        
    easiest_classifications = classify(easiest_test_images, easiest_train_images, easiest_train_labels);
    
    num_test_images_0 = size(test_images_0, 1);
    num_test_images_1 = size(test_images_1, 1);
    
    lda_easiest_accuracy = nnz([easiest_classifications(1:num_test_images_0)==0; ...
        easiest_classifications(end-num_test_images_1:end)==1])...
        /(num_test_images_0+num_test_images_1);
    
    % SVM
    
    easiest_svm = fitcsvm(easiest_train_images, easiest_train_labels);
    svm_easiest_classifications = predict(easiest_svm, easiest_test_images);
    svm_easiest_accuracy = nnz(svm_easiest_classifications==easiest_test_labels)/length(easiest_test_labels);
    
    % Decision Tree
    
    easiest_tree = fitctree(easiest_train_images, easiest_train_labels);
    tree_easiest_classifications = predict(easiest_tree, easiest_test_images);
    tree_easiest_accuracy = nnz(tree_easiest_classifications==easiest_test_labels)/length(easiest_test_labels);
    
    % We now consider the hardest digits to separate, which we determined
    % earlier were 5 and 8
    
    train_images_5 = scaled_projected_train_images(train_labels==5,:);
    train_images_8 = scaled_projected_train_images(train_labels==8,:);
    test_images_5 = scaled_projected_test_images(test_labels==5,:);
    test_images_8 = scaled_projected_test_images(test_labels==8,:);
    
    hardest_test_images = [test_images_5; test_images_8];
    hardest_test_labels = [zeros(size(test_images_5,1), 1); ones(size(test_images_8, 1), 1)];
    hardest_train_images = [train_images_5; train_images_8];
    hardest_train_labels = [zeros(size(train_images_5,1), 1); ones(size(train_images_8, 1), 1)];
    
    % LDA
    
    hardest_classifications = classify(hardest_test_images, hardest_train_images, hardest_train_labels);
    
    num_test_images_5 = size(test_images_5, 1);
    num_test_images_8 = size(test_images_8, 1);
    
    lda_hardest_accuracy = nnz([hardest_classifications(1:num_test_images_5)==0; ...
        hardest_classifications(end-num_test_images_8:end)==1])...
        /(num_test_images_5+num_test_images_8);
    
    % SVM
    
    hardest_svm = fitcsvm(hardest_train_images, hardest_train_labels);
    svm_hardest_classifications = predict(hardest_svm, hardest_test_images);
    svm_hardest_accuracy = nnz(svm_hardest_classifications==hardest_test_labels)/length(hardest_test_labels);
    
    % Decision Tree
    
    hardest_tree = fitctree(hardest_train_images, hardest_train_labels);
    tree_hardest_classifications = predict(hardest_tree, hardest_test_images);
    tree_hardest_accuracy = nnz(tree_hardest_classifications==hardest_test_labels)/length(hardest_test_labels);
    
    % Reporting Results
    
    fprintf('\nMethod Comparison Results:\n')
    fprintf(['On the easiest digits 0 and 1, our LDA was %g%% accurate,\n'...
        'our SVM was %g%% accurate, and our decision tree was %g%% accurate\n'],...
        lda_easiest_accuracy*100, svm_easiest_accuracy*100, tree_easiest_accuracy*100);
    fprintf(['On the hardest digits 5 and 8, our LDA was %g%% accurate,\n'...
        'our SVM was %g%% accurate, and our decision tree was %g%% accurate\n'],...
        lda_hardest_accuracy*100, svm_hardest_accuracy*100, tree_hardest_accuracy*100);
    
end