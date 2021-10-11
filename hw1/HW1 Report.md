# HW1 Report
### Joyce Huang

### a)
> *using the training data in TrainingSamplesDCT 8.mat, what are reasonable estimates for the prior probabilities?*

Based on the given dataset, the prior probabilities of cheetah and grass are as follow:

- $P_Y(cheetah) = \frac{250}{1053+250} = 0.1919$
- $P_Y(grass) = \frac{1053}{1053+250} = 0.8081$  

### b)
> *using the training data in TrainingSamplesDCT 8.mat, compute and plot the index histograms PX|Y (x|cheetah) and PX|Y (x|grass).*

![](https://github.com/ChungYujoyce/UCSD-ECE-271A/blob/main/hw1/prob_.png)

### c) 
>  *for each block in the image cheetah.bmp, compute the feature X (index of the DCT coefficient with 2 nd greatest energy). Compute the state variable Y using the minimum probability of error rule based on the probabilities obtained in a) and b). Store the state in an array A. Using the commands imagesc and colormap(gray (255)) create a picture of that array.*
![](https://github.com/ChungYujoyce/UCSD-ECE-271A/blob/main/hw1/pred.jpg)

### d)
> *The array A contains a mask that indicates which blocks contain grass and which contain the cheetah. Compare it with the ground truth provided in image cheetah mask.bmp (shown below on the right) and compute the probability of error of your algorithm.*

$ errorrate = \frac{\sum {xor(predict, groundtruth)}}{all-pixels} = \frac{11473}{60750} = 0.1666$


# Code

~~~matlab
data = load("TrainingSamplesDCT_8.mat");
zigzag = readmatrix('Zig-Zag Pattern.txt');
img = imread('cheetah.bmp');
img = im2double(img);

%% (a) prior probabilities
front = data.TrainsampleDCT_FG;
back = data.TrainsampleDCT_BG;

% get the dimensions
FH = height(front);
BH = height(back);

% count the proportions
prior_grass = BH / (FH + BH);
prior_cheetah = FH / (FH + BH);

% (b) compute histograms
% initialization
X_front = zeros(1, 64);
X_back = zeros(1, 64);

% sort each value and selct the 2nd largest abs value
for i = 1:FH
    [value, sorted_idx] = sort(abs(front(i, :)));
    % index 63 has the 2nd largest value
    % plus one on counts for histogram 
    X_front(sorted_idx(63)) = X_front(sorted_idx(63)) + 1;
end

% do it again for back data
for i = 1:BH
    [value, sorted_idx] = sort(abs(back(i, :)));
    X_back(sorted_idx(63)) = X_back(sorted_idx(63)) + 1;
end

% compute PX|Y (x|cheetah) and PX|Y (x|grass)
% each count of 2nd largest divided by total counts
X_front = X_front ./ FH;
X_back = X_back ./ BH;


% draw the graphs (two combined)
f = figure;
bar([X_front(:), X_back(:)], 'BarWidth', 1.5);
xlabel('2nd largest x counts & position', 'FontSize', 15);
ylabel('Probability', 'FontSize', 15);
title('Index Histogram');
legend('P(X|cheetah)', 'P(X|grass)');
set(gca,'XLim',[0,40]);
saveas(f, "prob_", "png");

% (c) create picture
threshold = prior_grass / prior_cheetah;

[imgH, imgW] = size(img);
% index 0~63 to index~64
zigzag = zigzag + 1;
%8x8 block go through whole image
frame = zeros((imgH - 7)*(imgW - 7) , 64);
% 8x8 block turn to 1*64 vector
vector = zeros(1, 64);

for i = 1:(imgH - 7)
    for j = 1:(imgW - 7)
        % compute DCT for each block
        dct_process = dct2(img(i:i+7, j:j+7));
        for x = 1:8
            for y = 1:8
                % store the DCT values in zigzag pattern
                vector(zigzag(x, y)) = dct_process(x, y);
            end
        end
        % transform the structure from vectoor to block type
        frame((i-1)*(imgH-7) + j, :) = vector;
    end
end

result = zeros(imgH, imgW);

for i = 1:size(frame, 1)
    [sorted_pos, sorted_idx] = sort(abs(frame(i, :)));
    if((X_front(sorted_idx(63)))/(X_back(sorted_idx(63)))> threshold)
        result(floor(i / 248) + 1, rem(i, 248) + 1) = 1;
    end
end


imshow(result);
imwrite(result, 'pred.jpg');
result = mat2gray(result);

% (d) prob of error 

ground_truth = imread('cheetah_mask.bmp');
ground_truth = im2double(ground_truth);
% compute false rate
% use XOR to detect if pred and ground_truth are same or not
false_rate = sum(sum(xor(ground_truth, result))) / (imgH * imgW);
disp(false_rate);
~~~