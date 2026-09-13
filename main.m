%% Working with dataset

img = imread("s5\1.pgm");
imshow(img)   % show image

%% Generating matrices

all_A_matrices = {};
mean_faces = {};

for i = 1:40
    A = [];
    for j = 1:10
        adr = "s"+i+"\"+j+".pgm";
        img = imread(adr);
        img = im2double(img);
        vector = img(:);
        A = [A vector];
    end

    meanFace = mean(A,2);
    A = A - repmat(meanFace,1,size(A,2));

    mean_faces{i} = meanFace;
    all_A_matrices{i} = A;
end


    adr = "s"+5+"\"+1+".pgm";
    img = imread(adr);
    B = reshape(mean_faces{5},size(img));
    imshow(B)





person_number = 5;

A_person = all_A_matrices{person_number};  % Size: [pixels × 10]
mean_person = mean_faces{person_number};

[U, S, V] = svd(A_person, 'econ');

k_values = [3, 6];

figure;

for idx = 1:length(k_values)
    k = k_values(idx);
    
    S_k = S;
    S_k(k+1:end, k+1:end) = 0;
    
    A_k = U * S_k * V';
    
    A_k_with_mean = A_k ;
    

    j = 1;  
    
    reconstructed_vector = A_k_with_mean(:, j);
    
    reconstructed_img = reshape(reconstructed_vector, size(img));
    
    subplot(2, length(k_values) , idx );
    imshow(reconstructed_img, []);
    title(['k = ' num2str(k)]);
end

subplot(2, length(k_values) + 1, 1);
original_img = all_images{person_number}{1};  % First image of person 5
imshow(original_img);
title('Original Image');







person_number = 5;


A_person   = all_A_matrices{person_number};   % [pixels x 10]
meanFace   = mean_faces{person_number};       % [pixels x 1]

img0 = im2double(imread("s"+person_number+"\1.pgm"));
[H, W] = size(img0);

[U, S, V] = svd(A_person, 'econ');

k_values = [3, 6];

figure;

for idx = 1:length(k_values)
    k = k_values(idx);

    U_k = U(:, 1:k);


    coeffs = U_k' * A_person;          % [k x 10]

    coeff_mean = mean(coeffs, 2);      % [k x 1]

    rep_vec = meanFace + U_k * coeff_mean;     % [pixels x 1]
    rep_img = reshape(rep_vec, [H, W]);

    subplot(1, length(k_values), idx);
    imshow(rep_img, []);
    title(['Representative face (k = ' num2str(k) ')']);
end


figure;
num_show = 6;
for i = 1:num_show
    ef_img = reshape(U(:, i), [H, W]);  % eigenface i
    subplot(2, 3, i);
    imshow(ef_img, []);
    title(['Eigenface #' num2str(i)]);
end






%% Recognition

k = 6;            
num_people = 40;

img_ref = im2double(imread("s1\1.pgm"));
[H, W] = size(img_ref);

U_k_people = cell(1, num_people);

for i = 1:num_people
    A = all_A_matrices{i};        % mean-centered matrix of person i
    [U, S, V] = svd(A, 'econ');  
    U_k_people{i} = U(:,1:k);     % first k eigenfaces
end


rec_folder = "recognition\";
rec_images = ["p1.jpg","p2.jpg","p3.jpg"];

predicted_person = zeros(1,length(rec_images));

for t = 1:length(rec_images)

    adr = rec_folder + rec_images(t);
    img = imread(adr);

    if ndims(img) == 3
        img = rgb2gray(img);
    end

    img = im2double(img);

    if size(img,1) ~= H || size(img,2) ~= W
        img = imresize(img,[H W]);
    end

    x = img(:);   

    errors = zeros(1,num_people);

    for i = 1:num_people

        meanFace = mean_faces{i};
        U_k = U_k_people{i};

        a = x - meanFace;

        a_k = U_k * (U_k' * a);
        x_hat = meanFace + a_k;

        errors(i) = norm(x - x_hat,2);
    end

    [~, predicted_person(t)] = min(errors);

    fprintf("Image %s -> Predicted person = %d\n", ...
            rec_images(t), predicted_person(t));
end



figure;
for t = 1:length(rec_images)

    adr = rec_folder + rec_images(t);
    img = imread(adr);
    if ndims(img)==3, img = rgb2gray(img); end
    img = im2double(img);
    img = imresize(img,[H W]);

    subplot(1,3,t);
    imshow(img,[]);
    title(sprintf('%s -> Person %d',rec_images(t),predicted_person(t)));
end
