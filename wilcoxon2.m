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

%Aggregate groups as needed. Note: NormalLearningNeg would simply correspond to X7 and SuccessfulLearningPromoNeg to X8
SuccessfulLearningPos = vertcat(X1, X3, X5);
SuccessfulLearningPos = SuccessfulLearningPos(randperm(size(SuccessfulLearningPos,1)),:);
SuccessfulLearningNeg = vertcat(X4, X7);
SuccessfulLearningNeg = SuccessfulLearningNeg(randperm(size(SuccessfulLearningNeg,1)),:);
NormalLearningPos = vertcat(X1, X3);
NormalLearningPos = NormalLearningPos(randperm(size(NormalLearningPos,1)),:);
SuccessfulLearningPromoPos = vertcat(X2, X4, X6);
SuccessfulLearningPromoPos = SuccessfulLearningPromoPos(randperm(size(SuccessfulLearningPromoPos,1)),:);


% Perform Wilcoxin test for each protein higher in c-CS-s than c-SC-s.
% Output a file with p-values.

file1 = fopen('CFCeffectSalineNormal/CFCeffectSalineNormal_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X3(:,i),X4(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X3(:,i),X4(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file1, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file1);

file2 = fopen('CFCeffectMemantineNormal/CFCeffectSalineNormal_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X1(:,i),X2(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X1(:,i),X2(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file2, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file2);

file3 = fopen('MemantineEffectNormal/MemantineEffectNormal_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X2(:,i),X4(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X2(:,i),X4(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file3, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file3);

file4 = fopen('MemantineEffectCFCnormal/MemantineEffectCFCnormal_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X1(:,i),X3(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X1(:,i),X3(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file4, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file4);

file5 = fopen('CFCeffectSalineTrisomy/CFCeffectSalineTrisomy_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X7(:,i),X8(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X7(:,i),X8(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file5, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file5);

file6 = fopen('CFCeffectMemantineTrisomy/CFCeffectMemantineTrisomy_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X5(:,i),X6(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X5(:,i),X6(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file5, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file6);

file7 = fopen('MemantineEffectTrisomy/MemantineEffectTrisomy_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X6(:,i),X8(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X6(:,i),X8(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file7, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file7);

file8 = fopen('MemantineEffectCFCtrisomy/MemantineEffectCFCtrisomy_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X5(:,i),X7(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X5(:,i),X7(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file8, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file8);

file9 = fopen('InitialTrisomyDifference/InitialTrisomyDifference_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(X8(:,i),X4(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(X8(:,i),X4(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file9, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file9);

file10 = fopen('SuccessfulLearningDifference/SuccessfulLearningDifference_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(SuccessfulLearningPos(:,i),SuccessfulLearningNeg(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(SuccessfulLearningPos(:,i),SuccessfulLearningNeg(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file10, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file10);

file11 = fopen('NormalLearningDifference/NormalLearningDifference_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(NormalLearningPos(:,i),X7(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(NormalLearningPos(:,i),X7(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file11, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file11);

file12 = fopen('SuccessfulLearningPromoDiff/SuccessfulLearningPromoDiff_ranksumvals2.csv','w');
for i = 1:77
	[P1, H1] = ranksum(SuccessfulLearningPromoPos(:,i),X8(:,i), 'alpha', 0.025, 'tail', 'left');
	[P2, H2] = ranksum(SuccessfulLearningPromoPos(:,i),X8(:,i), 'alpha', 0.025, 'tail', 'right');
	fprintf(file12, '%d, %d,%.8f, %d,%.8f\n',i,H1,P1,H2,P2);
end
fclose(file12);