% Programmer: Tara Eicher
% WSU ID: z847x563
% Class: Machine Learning (CS 697)
% Professor: Dr. Kaushik Sinha
% Program Description: Builds an SVM coordinate matrix and alpha vector
%    for the model file output by LIBSVM. Multiplies these to obtain the
%    w vector. Also obtains the b vector. Outputs these data to files.

% For each row, first print y-value with space, then print the index of
% each nonzero x, followed by its value. Print a newline at the end of each
% row.
%model= fopen('Documents/MachineLearning/project/CFCeffectSalineNormal/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/CFCeffectSalineNormal/hyper','w');
%model= fopen('Documents/MachineLearning/project/CFCeffectMemantineNormal/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/CFCeffectMemantineNormal/hyper','w');
%model= fopen('Documents/MachineLearning/project/MemantineEffectNormal/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/MemantineEffectNormal/hyper','w');
%model= fopen('Documents/MachineLearning/project/MemantineEffectCFCnormal/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/MemantineEffectCFCnormal/hyper','w');
%model= fopen('Documents/MachineLearning/project/CFCeffectSalineTrisomy/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/CFCeffectSalineTrisomy/hyper','w');
%model= fopen('Documents/MachineLearning/project/CFCeffectMemantineTrisomy/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/CFCeffectMemantineTrisomy/hyper','w');
%model= fopen('Documents/MachineLearning/project/MemantineEffectTrisomy/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/MemantineEffectTrisomy/hyper','w');
%model= fopen('Documents/MachineLearning/project/MemantineEffectCFCtrisomy/C2model','r');
%hyperplane = fopen('Documents/MachineLearning/project/MemantineEffectCFCtrisomy/hyper','w');
%model= fopen('Documents/MachineLearning/project/InitialTrisomyDifference/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/InitialTrisomyDifference/hyper','w');
%model= fopen('Documents/MachineLearning/project/SuccessfulLearningDifference/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/SuccessfulLearningDifference/hyper','w');
%model= fopen('Documents/MachineLearning/project/NormalLearningDifference/C5model','r');
%hyperplane = fopen('Documents/MachineLearning/project/NormalLearningDifference/hyper','w');
model= fopen('Documents/MachineLearning/project/SuccessfulLearningPromoDiff/C5model','r');
hyperplane = fopen('Documents/MachineLearning/project/SuccessfulLearningPromoDiff/hyper','w');

% Read junk lines, then beta = -rho
%junk = textscan(model1,'%s', 7);
for ln = 1:3
    fgets(model);
end
svs_line = fgets(model);
elements = textscan(svs_line, '%s %d', 1);
svs_count = elements{2};
svs_line = fgets(model);
elements = textscan(svs_line, '%s %f', 1);
rho = elements{2};
beta = rho * -1;
for ln = 1:3
    fgets(model);
end
svs = zeros(svs_count, 77);
alpha = zeros(svs_count, 1);
%For each new line (with \n), first char is concatenated to alpha.
%Rest of characters become a vector with indices at labels. This vector is 
%concatenated with svm matrix.
for n = 1:svs_count
    line = fgets(model);
    l = size(line);
    [alph_i, pos] = textscan(line, '%f', 1);
    alpha_i = alph_i{1};
    alpha(n) = alpha_i;
    array = textscan(line(pos:l(2)), '%d:%f');
    i = array{1};
    val = array{2};
    svs(n, i) = val;
end
% Finally, get w.
svs_with_alphas = repmat(alpha, 1, 77).*svs;
w = sum(svs_with_alphas);
size(w)
% Print w, then b, to a file.
fprintf(hyperplane, 'w = ');
fprintf(hyperplane, '%f ', w);
fprintf(hyperplane, '\n');
fprintf(hyperplane, 'b = %f', beta);

% Verify output by checking classification of support vectors.
crossCheck = zeros(size(svs, 1), 2);
crossCheck(:, 1) = (w * svs')' + repmat(beta, size(svs, 1), 1);
crossCheck(:, 2) = alpha./abs(alpha);
crossCheck


