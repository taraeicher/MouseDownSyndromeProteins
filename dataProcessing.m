% Programmer: Tara Eicher
% WSU ID: z847x563
% Class: Machine Learning (CS 697)
% Professor: Dr. Kaushik Sinha
% Program Description: Process a CSV file containing values of -1 for all
%    missing data points and containing 1 unneeded column at the beginning
%    and just before the end y-value into files compatible with the
%    LIBSVM library (which takes y at the beginning and x afterward).
%    Split the data set at 80/20 and replace missing values with the
%    averages for the respective column.

% Load the data and format so that the Y-value comes first.
in = load('Data_Cortex_Nuclear_modified.csv');
X = in(1:size(in, 1), 1:size(in, 2) - 1);
Y = in(1:size(in, 1), size(in, 2));
data = horzcat(Y, X);

% Extract all samples of class 1 - c-CS-m.
data1 = data(data(:, 1) == 1, :);
X1 = data1(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos1 = X1 >= 0;
pos_X1 = X1 .* pos1; 
col_means1 = mean(pos_X1);
[~, c1] = find(X1 < 0);
X1(X1 < 0) = col_means1(c1);
% Normalize all x-values.
X1 = (X1-repmat(min(X),size(X1,1),1))./(repmat(max(X)-min(X),size(X1,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data1 = horzcat(data1(:,1), X1);
data1 = data1(randperm(size(data1, 1)), :);
% Put 80 percent into training set, 20 into testing.
train1 = data1(1:ceil(size(data1, 1) * 8 / 10), :);
test1 = data1((size(train1, 1) + 1):size(data1, 1), :);

% Extract all samples of class 2 - c-SC-m.
data2 = data(data(:, 1) == 2, :);
X2 = data2(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos2 = X2 >= 0;
pos_X2 = X2 .* pos2; 
col_means2 = mean(pos_X2);
[~, c2] = find(X2 < 0);
X2(X2 < 0) = col_means2(c2);
% Normalize all x-values.
X2 = (X2-repmat(min(X),size(X2,1),1))./(repmat(max(X)-min(X),size(X2,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data2 = horzcat(data2(:,1), X2);
data2 = data2(randperm(size(data2, 1)), :);
% Put 80 percent into training set, 20 into testing.
train2 = data2(1:ceil(size(data2, 1) * 8 / 10), :);
test2 = data2((size(train2, 1) + 1):size(data2, 1), :);

% Extract all samples of class 3 - c-CS-s.
data3 = data(data(:, 1) == 3, :);
X3 = data3(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos3 = X3 >= 0;
pos_X3 = X3 .* pos3; 
col_means3 = mean(pos_X3);
[~, c3] = find(X3 < 0);
X3(X3 < 0) = col_means3(c3);
% Normalize all x-values.
X3 = (X3-repmat(min(X),size(X3,1),1))./(repmat(max(X)-min(X),size(X3,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data3 = horzcat(data3(:,1), X3);
data3 = data3(randperm(size(data3, 1)), :);
% Put 80 percent into training set, 20 into testing.
train3 = data3(1:ceil(size(data3, 1) * 8 / 10), :);
test3 = data3((size(train3, 1) + 1):size(data3, 1), :);

% Extract all samples of class 4 - c-SC-s.
data4 = data(data(:, 1) == 4, :);
X4 = data4(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos4 = X4 >= 0;
pos_X4 = X4 .* pos4; 
col_means4 = mean(pos_X4);
[~, c4] = find(X4 < 0);
X4(X4 < 0) = col_means4(c4);
% Normalize all x-values.
X4 = (X4-repmat(min(X),size(X4,1),1))./(repmat(max(X)-min(X),size(X4,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data4 = horzcat(data4(:,1), X4);
data4 = data4(randperm(size(data4, 1)), :);
% Put 80 percent into training set, 20 into testing.
train4 = data4(1:ceil(size(data4, 1) * 8 / 10), :);
test4 = data4((size(train4, 1) + 1):size(data4, 1), :);

% Extract all samples of class 5 - t-CS-m.
data5 = data(data(:, 1) == 5, :);
X5 = data5(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos5 = X5 >= 0;
pos_X5 = X5 .* pos5; 
col_means5 = mean(pos_X5);
[~, c5] = find(X5 < 0);
X5(X5 < 0) = col_means5(c5);
% Normalize all x-values.
X5 = (X5-repmat(min(X),size(X5,1),1))./(repmat(max(X)-min(X),size(X5,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data5 = horzcat(data5(:,1), X5);
data5 = data5(randperm(size(data5, 1)), :);
% Put 80 percent into training set, 20 into testing.
train5 = data5(1:ceil(size(data5, 1) * 8 / 10), :);
test5 = data5((size(train5, 1) + 1):size(data5, 1), :);

% Extract all samples of class 6 - t-SC-m.
data6 = data(data(:, 1) == 6, :);
X6 = data6(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos6 = X6 >= 0;
pos_X6 = X6 .* pos6; 
col_mean6 = mean(pos_X6);
[~, c6] = find(X6 < 0);
X6(X6 < 0) = col_mean6(c6);
% Normalize all x-values.
X6 = (X6-repmat(min(X),size(X6,1),1))./(repmat(max(X)-min(X),size(X6,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data6 = horzcat(data6(:,1), X6);
data6 = data6(randperm(size(data6, 1)), :);
% Put 80 percent into training set, 20 into testing.
train6 = data6(1:ceil(size(data6, 1) * 8 / 10), :);
test6 = data6((size(train6, 1) + 1):size(data6, 1), :);

% Extract all samples of class 7 - t-CS-s.
data7 = data(data(:, 1) == 7, :);
X7 = data7(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos7 = X7 >= 0;
pos_X7 = X7 .* pos7; 
col_mean7 = mean(pos_X7);
[~, c7] = find(X7 < 0);
X7(X7 < 0) = col_mean7(c7);
% Normalize all x-values.
X7 = (X7-repmat(min(X),size(X7,1),1))./(repmat(max(X)-min(X),size(X7,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data7 = horzcat(data7(:,1), X7);
data7 = data7(randperm(size(data7, 1)), :);
% Put 80 percent into training set, 20 into testing.
train7 = data7(1:ceil(size(data7, 1) * 8 / 10), :);
test7 = data7((size(train7, 1) + 1):size(data7, 1), :);

% Extract all samples of class 8 - t-SC-s.
data8 = data(data(:, 1) == 8, :);
X8 = data8(:, 2:size(data, 2));
% For all missing x-values, replace them with the column means for their
% class.
pos8 = X8 >= 0;
pos_X8 = X8 .* pos8; 
col_mean8 = mean(pos_X8);
[~, c8] = find(X8 < 0);
X8(X8 < 0) = col_mean8(c8);
% Normalize all x-values.
X8 = (X8-repmat(min(X),size(X8,1),1))./(repmat(max(X)-min(X),size(X8,1),1));
% Scramble the rows of the dataset so that they are random (not organized
% by mouse ID).
data8 = horzcat(data8(:,1), X8);
data8 = data8(randperm(size(data8, 1)), :);
% Put 80 percent into training set, 20 into testing.
train8 = data8(1:ceil(size(data8, 1) * 8 / 10), :);
test8 = data8((size(train8, 1) + 1):size(data8, 1), :);

% For each combo we're looking at, concat the data sets and scramble again.
% This needs to be done separately for training and testing.
% Output in sparse format.

% Output a training file in which c-CS-s = +1 and c-SC-s = -1.
CFCeffectSalineNormal_train = vertcat(train3, train4);
CFCeffectSalineNormal_train = CFCeffectSalineNormal_train(randperm(size(CFCeffectSalineNormal_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file1 = fopen('CFCeffectSalineNormal_train.txt','w');
for r = 1:size(CFCeffectSalineNormal_train, 1)
    if CFCeffectSalineNormal_train(r,1) == 3
        fprintf(file1,'+1 ');
    else
        fprintf(file1, '-1 ');
    end
    for c = 2:size(CFCeffectSalineNormal_train, 2)
        if CFCeffectSalineNormal_train(r,c) ~= 0
            fprintf(file1,'%d:%6.4f ', c - 1, CFCeffectSalineNormal_train(r,c));
        end
    end
    fprintf(file1, '\n');
end
fclose(file1);

% Output a testing file in which c-CS-s = +1 and c-SC-s = -1.
CFCeffectSalineNormal_test = vertcat(test3, test4);
CFCeffectSalineNormal_test = CFCeffectSalineNormal_test(randperm(size(CFCeffectSalineNormal_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file2 = fopen('CFCeffectSalineNormal_test.txt','w');
for r = 1:size(CFCeffectSalineNormal_test, 1)
    if CFCeffectSalineNormal_test(r,1) == 3
        fprintf(file2,'+1 ');
    else
        fprintf(file2, '-1 ');
    end
    for c = 2:size(CFCeffectSalineNormal_test, 2)
        if CFCeffectSalineNormal_test(r,c) ~= 0
            fprintf(file2,'%d:%6.4f ', c - 1, CFCeffectSalineNormal_test(r,c));
        end
    end
    fprintf(file2, '\n');
end
fclose(file2);

% Output a training file in which c-CS-m = +1 and c-SC-m = -1.
CFCeffectMemantineNormal_train = vertcat(train1, train2);
CFCeffectMemantineNormal_train = CFCeffectMemantineNormal_train(randperm(size(CFCeffectMemantineNormal_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file3 = fopen('CFCeffectMemantineNormal_train.txt','w');
for r = 1:size(CFCeffectMemantineNormal_train, 1)
    if CFCeffectMemantineNormal_train(r,1) == 1
        fprintf(file3,'+1 ');
    else
        fprintf(file3, '-1 ');
    end
    for c = 2:size(CFCeffectMemantineNormal_train, 2)
        if CFCeffectMemantineNormal_train(r,c) ~= 0
            fprintf(file3,'%d:%6.4f ', c - 1, CFCeffectMemantineNormal_train(r,c));
        end
    end
    fprintf(file3, '\n');
end
fclose(file3);

% Output a testing file in which c-CS-m = +1 and c-SC-m = -1.
CFCeffectMemantineNormal_test = vertcat(test1, test2);
CFCeffectMemantineNormal_test = CFCeffectMemantineNormal_test(randperm(size(CFCeffectMemantineNormal_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file4 = fopen('CFCeffectMemantineNormal_test.txt','w');
for r = 1:size(CFCeffectMemantineNormal_test, 1)
    if CFCeffectMemantineNormal_test(r,1) == 1
        fprintf(file4,'+1 ');
    else
        fprintf(file4, '-1 ');
    end
    for c = 2:size(CFCeffectMemantineNormal_test, 2)
        if CFCeffectMemantineNormal_test(r,c) ~= 0
            fprintf(file4,'%d:%6.4f ', c - 1, CFCeffectMemantineNormal_test(r,c));
        end
    end
    fprintf(file4, '\n');
end
fclose(file4);

% Output a training file in which c-SC-m = +1 and c-SC-s = -1.
MemantineEffectNormal_train = vertcat(train2, train4);
MemantineEffectNormal_train = MemantineEffectNormal_train(randperm(size(MemantineEffectNormal_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file5 = fopen('MemantineEffectNormal_train.txt','w');
for r = 1:size(MemantineEffectNormal_train, 1)
    if MemantineEffectNormal_train(r,1) == 2
        fprintf(file5,'+1 ');
    else
        fprintf(file5, '-1 ');
    end
    for c = 2:size(MemantineEffectNormal_train, 2)
        if MemantineEffectNormal_train(r,c) ~= 0
            fprintf(file5,'%d:%6.4f ', c - 1, MemantineEffectNormal_train(r,c));
        end
    end
    fprintf(file5, '\n');
end
fclose(file5);

% Output a testing file in which c-SC-m = +1 and c-SC-s = -1.
MemantineEffectNormal_test = vertcat(test2, test4);
MemantineEffectNormal_test = MemantineEffectNormal_test(randperm(size(MemantineEffectNormal_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file6 = fopen('MemantineEffectNormal_test.txt','w');
for r = 1:size(MemantineEffectNormal_test, 1)
    if MemantineEffectNormal_test(r,1) == 2
        fprintf(file6,'+1 ');
    else
        fprintf(file6, '-1 ');
    end
    for c = 2:size(MemantineEffectNormal_test, 2)
        if MemantineEffectNormal_test(r,c) ~= 0
            fprintf(file6,'%d:%6.4f ', c - 1, MemantineEffectNormal_test(r,c));
        end
    end
    fprintf(file6, '\n');
end
fclose(file6);

% Output a training file in which c-CS-m = +1 and c-CS-s = -1.
MemantineEffectCFCnormal_train = vertcat(train1, train3);
MemantineEffectCFCnormal_train = MemantineEffectCFCnormal_train(randperm(size(MemantineEffectCFCnormal_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file7 = fopen('MemantineEffectCFCnormal_train.txt','w');
for r = 1:size(MemantineEffectCFCnormal_train, 1)
    if MemantineEffectCFCnormal_train(r,1) == 1
        fprintf(file7,'+1 ');
    else
        fprintf(file7, '-1 ');
    end
    for c = 2:size(MemantineEffectCFCnormal_train, 2)
        if MemantineEffectCFCnormal_train(r,c) ~= 0
            fprintf(file7,'%d:%6.4f ', c - 1, MemantineEffectCFCnormal_train(r,c));
        end
    end
    fprintf(file7, '\n');
end
fclose(file7);

% Output a testing file in which c-CS-m = +1 and c-CS-s = -1.
MemantineEffectCFCnormal_test = vertcat(test1, test3);
MemantineEffectCFCnormal_test = MemantineEffectCFCnormal_test(randperm(size(MemantineEffectCFCnormal_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file8 = fopen('MemantineEffectCFCnormal_test.txt','w');
for r = 1:size(MemantineEffectCFCnormal_test, 1)
    if MemantineEffectCFCnormal_test(r,1) == 1
        fprintf(file8,'+1 ');
    else
        fprintf(file8, '-1 ');
    end
    for c = 2:size(MemantineEffectCFCnormal_test, 2)
        if MemantineEffectCFCnormal_test(r,c) ~= 0
            fprintf(file8,'%d:%6.4f ', c - 1, MemantineEffectCFCnormal_test(r,c));
        end
    end
    fprintf(file8, '\n');
end
fclose(file8);

% Output a training file in which t-CS-s = +1 and t-SC-s = -1.
CFCeffectSalineTrisomy_train = vertcat(train7, train8);
CFCeffectSalineTrisomy_train = CFCeffectSalineTrisomy_train(randperm(size(CFCeffectSalineTrisomy_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file9 = fopen('CFCeffectSalineTrisomy_train.txt','w');
for r = 1:size(CFCeffectSalineTrisomy_train, 1)
    if CFCeffectSalineTrisomy_train(r,1) == 7
        fprintf(file9,'+1 ');
    else
        fprintf(file9, '-1 ');
    end
    for c = 2:size(CFCeffectSalineTrisomy_train, 2)
        if CFCeffectSalineTrisomy_train(r,c) ~= 0
            fprintf(file9,'%d:%6.4f ', c - 1, CFCeffectSalineTrisomy_train(r,c));
        end
    end
    fprintf(file9, '\n');
end
fclose(file9);

% Output a testing file in which t-CS-s = +1 and t-SC-s = -1.
CFCeffectSalineTrisomy_test = vertcat(test7, test8);
CFCeffectSalineTrisomy_test = CFCeffectSalineTrisomy_test(randperm(size(CFCeffectSalineTrisomy_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file10 = fopen('CFCeffectSalineTrisomy_test.txt','w');
for r = 1:size(CFCeffectSalineTrisomy_test, 1)
    if CFCeffectSalineTrisomy_test(r,1) == 7
        fprintf(file10,'+1 ');
    else
        fprintf(file10, '-1 ');
    end
    for c = 2:size(CFCeffectSalineTrisomy_test, 2)
        if CFCeffectSalineTrisomy_test(r,c) ~= 0
            fprintf(file10,'%d:%6.4f ', c - 1, CFCeffectSalineTrisomy_test(r,c));
        end
    end
    fprintf(file10, '\n');
end
fclose(file10);

% Output a training file in which t-CS-m = +1 and t-SC-m = -1.
CFCeffectMemantineTrisomy_train = vertcat(train5, train6);
CFCeffectMemantineTrisomy_train = CFCeffectMemantineTrisomy_train(randperm(size(CFCeffectMemantineTrisomy_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file11 = fopen('CFCeffectMemantineTrisomy_train.txt','w');
for r = 1:size(CFCeffectMemantineTrisomy_train, 1)
    if CFCeffectMemantineTrisomy_train(r,1) == 5
        fprintf(file11,'+1 ');
    else
        fprintf(file11, '-1 ');
    end
    for c = 2:size(CFCeffectMemantineTrisomy_train, 2)
        if CFCeffectMemantineTrisomy_train(r,c) ~= 0
            fprintf(file11,'%d:%6.4f ', c - 1, CFCeffectMemantineTrisomy_train(r,c));
        end
    end
    fprintf(file11, '\n');
end
fclose(file11);

% Output a testing file in which t-CS-m = +1 and t-SC-m = -1.
CFCeffectMemantineTrisomy_test = vertcat(test5, test6);
CFCeffectMemantineTrisomy_test = CFCeffectMemantineTrisomy_test(randperm(size(CFCeffectMemantineTrisomy_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file12 = fopen('CFCeffectMemantineTrisomy_test.txt','w');
for r = 1:size(CFCeffectMemantineTrisomy_test, 1)
    if CFCeffectMemantineTrisomy_test(r,1) == 5
        fprintf(file12,'+1 ');
    else
        fprintf(file12, '-1 ');
    end
    for c = 2:size(CFCeffectMemantineTrisomy_test, 2)
        if CFCeffectMemantineTrisomy_test(r,c) ~= 0
            fprintf(file12,'%d:%6.4f ', c - 1, CFCeffectMemantineTrisomy_test(r,c));
        end
    end
    fprintf(file12, '\n');
end
fclose(file12);

% Output a training file in which t-SC-m = +1 and t-SC-s = -1.
MemantineEffectTrisomy_train = vertcat(train6, train8);
MemantineEffectTrisomy_train = MemantineEffectTrisomy_train(randperm(size(MemantineEffectTrisomy_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file13 = fopen('MemantineEffectTrisomy_train.txt','w');
for r = 1:size(MemantineEffectTrisomy_train, 1)
    if MemantineEffectTrisomy_train(r,1) == 6
        fprintf(file13,'+1 ');
    else
        fprintf(file13, '-1 ');
    end
    for c = 2:size(MemantineEffectTrisomy_train, 2)
        if MemantineEffectTrisomy_train(r,c) ~= 0
            fprintf(file13,'%d:%6.4f ', c - 1, MemantineEffectTrisomy_train(r,c));
        end
    end
    fprintf(file13, '\n');
end
fclose(file13);

% Output a testing file in which t-SC-m = +1 and t-SC-s = -1.
MemantineEffectTrisomy_test = vertcat(test6, test8);
MemantineEffectTrisomy_test = MemantineEffectTrisomy_test(randperm(size(MemantineEffectTrisomy_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file14 = fopen('MemantineEffectTrisomy_test.txt','w');
for r = 1:size(MemantineEffectTrisomy_test, 1)
    if MemantineEffectTrisomy_test(r,1) == 6
        fprintf(file14,'+1 ');
    else
        fprintf(file14, '-1 ');
    end
    for c = 2:size(MemantineEffectTrisomy_test, 2)
        if MemantineEffectTrisomy_test(r,c) ~= 0
            fprintf(file14,'%d:%6.4f ', c - 1, MemantineEffectTrisomy_test(r,c));
        end
    end
    fprintf(file14, '\n');
end
fclose(file14);

% Output a training file in which t-CS-m = +1 and t-CS-s = -1.
MemantineEffectCFCtrisomy_train = vertcat(train5, train7);
MemantineEffectCFCtrisomy_train = MemantineEffectCFCtrisomy_train(randperm(size(MemantineEffectCFCtrisomy_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file15 = fopen('MemantineEffectCFCtrisomy_train.txt','w');
for r = 1:size(MemantineEffectCFCtrisomy_train, 1)
    if MemantineEffectCFCtrisomy_train(r,1) == 5
        fprintf(file15,'+1 ');
    else
        fprintf(file15, '-1 ');
    end
    for c = 2:size(MemantineEffectCFCtrisomy_train, 2)
        if MemantineEffectCFCtrisomy_train(r,c) ~= 0
            fprintf(file15,'%d:%6.4f ', c - 1, MemantineEffectCFCtrisomy_train(r,c));
        end
    end
    fprintf(file15, '\n');
end
fclose(file15);

% Output a testing file in which t-CS-m = +1 and t-CS-s = -1.
MemantineEffectCFCtrisomy_test = vertcat(test5, test7);
MemantineEffectCFCtrisomy_test = MemantineEffectCFCtrisomy_test(randperm(size(MemantineEffectCFCtrisomy_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file16 = fopen('MemantineEffectCFCtrisomy_test.txt','w');
for r = 1:size(MemantineEffectCFCtrisomy_test, 1)
    if MemantineEffectCFCtrisomy_test(r,1) == 5
        fprintf(file16,'+1 ');
    else
        fprintf(file16, '-1 ');
    end
    for c = 2:size(MemantineEffectCFCtrisomy_test, 2)
        if MemantineEffectCFCtrisomy_test(r,c) ~= 0
            fprintf(file16,'%d:%6.4f ', c - 1, MemantineEffectCFCtrisomy_test(r,c));
        end
    end
    fprintf(file16, '\n');
end
fclose(file16);

% Output a training file in which t-SC-s = +1 and c-SC-s = -1.
InitialTrisomyDifference_train = vertcat(train4, train8);
InitialTrisomyDifference_train = InitialTrisomyDifference_train(randperm(size(InitialTrisomyDifference_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file17 = fopen('InitialTrisomyDifference_train.txt','w');
for r = 1:size(InitialTrisomyDifference_train, 1)
    if InitialTrisomyDifference_train(r,1) == 8
        fprintf(file17,'+1 ');
    else
        fprintf(file17, '-1 ');
    end
    for c = 2:size(InitialTrisomyDifference_train, 2)
        if InitialTrisomyDifference_train(r,c) ~= 0
            fprintf(file17,'%d:%6.4f ', c - 1, InitialTrisomyDifference_train(r,c));
        end
    end
    fprintf(file17, '\n');
end
fclose(file17);

% Output a testing file in which t-SC-s = +1 and c-SC-s = -1.
InitialTrisomyDifference_test = vertcat(test4, test8);
InitialTrisomyDifference_test = InitialTrisomyDifference_test(randperm(size(InitialTrisomyDifference_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file18 = fopen('InitialTrisomyDifference_test.txt','w');
for r = 1:size(InitialTrisomyDifference_test, 1)
    if InitialTrisomyDifference_test(r,1) == 8
        fprintf(file18,'+1 ');
    else
        fprintf(file18, '-1 ');
    end
    for c = 2:size(InitialTrisomyDifference_test, 2)
        if InitialTrisomyDifference_test(r,c) ~= 0
            fprintf(file18,'%d:%6.4f ', c - 1, InitialTrisomyDifference_test(r,c));
        end
    end
    fprintf(file18, '\n');
end
fclose(file18);

% Output a training file in which c-CS-m, c-CS-s, and t-CS-m = +1 and c-SC-s and t-CS-s = -1.
SuccessfulLearningDifference_train = vertcat(train1, train3, train5, train4, train7);
SuccessfulLearningDifference_train = SuccessfulLearningDifference_train(randperm(size(SuccessfulLearningDifference_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file19 = fopen('SuccessfulLearningDifference_train.txt','w');
for r = 1:size(SuccessfulLearningDifference_train, 1)
    if SuccessfulLearningDifference_train(r,1) == 1 || SuccessfulLearningDifference_train(r,1) == 3 || SuccessfulLearningDifference_train(r,1) == 5
        fprintf(file19,'+1 ');
    else
        fprintf(file19, '-1 ');
    end
    for c = 2:size(SuccessfulLearningDifference_train, 2)
        if SuccessfulLearningDifference_train(r,c) ~= 0
            fprintf(file19,'%d:%6.4f ', c - 1, SuccessfulLearningDifference_train(r,c));
        end
    end
    fprintf(file19, '\n');
end
fclose(file19);

% Output a testing file in which c-CS-m, c-CS-s, and t-CS-m = +1 and c-SC-s and t-CS-s = -1.
SuccessfulLearningDifference_test = vertcat(test1, test3, test5, test4, test7);
SuccessfulLearningDifference_test = SuccessfulLearningDifference_test(randperm(size(SuccessfulLearningDifference_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file20 = fopen('SuccessfulLearningDifference_test.txt','w');
for r = 1:size(SuccessfulLearningDifference_test, 1)
    if SuccessfulLearningDifference_test(r,1) == 1 || SuccessfulLearningDifference_test(r,1) == 3 || SuccessfulLearningDifference_test(r,1) == 5
        fprintf(file20,'+1 ');
    else
        fprintf(file20, '-1 ');
    end
    for c = 2:size(SuccessfulLearningDifference_test, 2)
        if SuccessfulLearningDifference_test(r,c) ~= 0
            fprintf(file20,'%d:%6.4f ', c - 1, SuccessfulLearningDifference_test(r,c));
        end
    end
    fprintf(file20, '\n');
end
fclose(file20);

% Output a training file in which c-CS-m and c-CS-s = +1 and t-CS-s = -1.
NormalLearningDifference_train = vertcat(train1, train3, train7);
NormalLearningDifference_train = NormalLearningDifference_train(randperm(size(NormalLearningDifference_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file21 = fopen('NormalLearningDifference_train.txt','w');
for r = 1:size(NormalLearningDifference_train, 1)
    if NormalLearningDifference_train(r,1) == 1 || NormalLearningDifference_train(r,1) == 3
        fprintf(file21,'+1 ');
    else
        fprintf(file21, '-1 ');
    end
    for c = 2:size(NormalLearningDifference_train, 2)
        if NormalLearningDifference_train(r,c) ~= 0
            fprintf(file21,'%d:%6.4f ', c - 1, NormalLearningDifference_train(r,c));
        end
    end
    fprintf(file21, '\n');
end
fclose(file21);

% Output a testing file in which c-CS-m and c-CS-s = +1 and t-CS-s = -1.
NormalLearningDifference_test = vertcat(test1, test3, test7);
NormalLearningDifference_test = NormalLearningDifference_test(randperm(size(NormalLearningDifference_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file22 = fopen('NormalLearningDifference_test.txt','w');
for r = 1:size(NormalLearningDifference_test, 1)
    if NormalLearningDifference_test(r,1) == 1 || NormalLearningDifference_test(r,1) == 3
        fprintf(file22,'+1 ');
    else
        fprintf(file22, '-1 ');
    end
    for c = 2:size(NormalLearningDifference_test, 2)
        if NormalLearningDifference_test(r,c) ~= 0
            fprintf(file22,'%d:%6.4f ', c - 1, NormalLearningDifference_test(r,c));
        end
    end
    fprintf(file22, '\n');
end
fclose(file22);

% Output a training file in which c-SC-m, c-SC-s, and t-SC-m = +1 and t-CS-s = -1.
SuccessfulLearningPromoDiff_train = vertcat(train2, train4, train6, train8);
SuccessfulLearningPromoDiff_train = SuccessfulLearningPromoDiff_train(randperm(size(SuccessfulLearningPromoDiff_train, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file23 = fopen('SuccessfulLearningPromoDiff_train.txt','w');
for r = 1:size(SuccessfulLearningPromoDiff_train, 1)
    if SuccessfulLearningPromoDiff_train(r,1) == 2 || SuccessfulLearningPromoDiff_train(r,1) == 4 || SuccessfulLearningPromoDiff_train(r,1) == 6
        fprintf(file23,'+1 ');
    else
        fprintf(file23, '-1 ');
    end
    for c = 2:size(SuccessfulLearningPromoDiff_train, 2)
        if SuccessfulLearningPromoDiff_train(r,c) ~= 0
            fprintf(file23,'%d:%6.4f ', c - 1, SuccessfulLearningPromoDiff_train(r,c));
        end
    end
    fprintf(file23, '\n');
end
fclose(file23);

% Output a testing file in which c-SC-m, c-SC-s, and t-SC-m = +1 and t-CS-s = -1.
SuccessfulLearningPromoDiff_test = vertcat(test2, test4, test6, test8);
SuccessfulLearningPromoDiff_test = SuccessfulLearningPromoDiff_test(randperm(size(SuccessfulLearningPromoDiff_test, 1)), :);
% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
file24 = fopen('SuccessfulLearningPromoDiff_test.txt','w');
for r = 1:size(SuccessfulLearningPromoDiff_test, 1)
    if SuccessfulLearningPromoDiff_test(r,1) == 2 || SuccessfulLearningPromoDiff_test(r,1) == 4 || SuccessfulLearningPromoDiff_test(r,1) == 6
        fprintf(file24,'+1 ');
    else
        fprintf(file24, '-1 ');
    end
    for c = 2:size(SuccessfulLearningPromoDiff_test, 2)
        if SuccessfulLearningPromoDiff_test(r,c) ~= 0
            fprintf(file24,'%d:%6.4f ', c - 1, SuccessfulLearningPromoDiff_test(r,c));
        end
    end
    fprintf(file24, '\n');
end
fclose(file24);
