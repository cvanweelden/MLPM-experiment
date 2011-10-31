%Meta-trainer
clear


%Parameters:
k_max = 5;
xmin = -1;
xmax = 1;
sdmin = 0.1;
sdmax = 0.4;
p_max = 5;
sample_sizes = [5 50 500 5000];
train_sets = 50;
test_size = 500;

no_of_tests = 10;
no_sizes = length(sample_sizes);
no_of_methods = 1; %[Naive Bayes, SVM]

%Initialize sample matrices 4D (sample_size x |param|+|class| x train_sets
%x no_of_tests)
sample_1 = zeros(sample_sizes(1)*2,3,train_sets, no_of_tests);
sample_2 = zeros(sample_sizes(2)*2,3,train_sets, no_of_tests);
sample_3 = zeros(sample_sizes(3)*2,3,train_sets, no_of_tests);
sample_4 = zeros(sample_sizes(4)*2,3,train_sets, no_of_tests);
% Initialize test matrices 3D (test_size x |param|+|true class|+|bayes
% optimal| x no_of_tests)
sample_t = zeros(test_size *2, 4, no_of_tests);

% Results matrix 3D (|performance, bias, variance, noise| x sample_sizes x
% no_of_tests)
decomposition = zeros(4,no_sizes*no_of_methods, k_max, no_of_tests);
tic
for iter= 1:no_of_tests
    for k = 1:k_max
        gmm_sampling;
        naive_bayestrainer;
        %svmtrainer;
            
        fprintf('iteration %d, k=%d:\n',iter, k);
        time_spent = toc
    end

    
end

decomposition

%get mean, remove NaN values for this..
if sum(isnan(decomposition(:))) > 0
    fprintf('NaN detected in decomposition\n')
    decomposition(isnan(decomposition(:))) = 0;
end

meandecomp = mean(decomposition, 4)