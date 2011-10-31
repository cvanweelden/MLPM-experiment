%No classify, only generator to check effect of components on noise
clear


%Parameters:
k = 1;
k_max = 50;
xmin = -1;
xmax = 1;
sdmin = 0.1;
sdmax = 0.4;
p_max = 5;
sample_sizes = 5000;
train_sets = 10000;
test_size = 500;

no_of_tests = 25;
no_sizes = length(sample_sizes);
no_of_methods = 2; %[Naive Bayes, SVM]

%Initialize sample matrices 4D (sample_size x |param|+|class| x train_sets
%x no_of_tests)
%sample_1 = zeros(sample_sizes(1)*2,3,train_sets, no_of_tests);
%sample_2 = zeros(sample_sizes(2)*2,3,train_sets, no_of_tests);
%sample_3 = zeros(sample_sizes(3)*2,3,train_sets, no_of_tests);
%sample_4 = zeros(sample_sizes(4)*2,3,train_sets, no_of_tests);
% Initialize test matrices 3D (test_size x |param|+|true class|+|bayes
% optimal| x no_of_tests)
sample_t = zeros(test_size *2, 4, no_of_tests);

% Results matrix 3D (|performance, bias, variance, noise| x sample_sizes x
% no_of_tests)
decomposition = zeros(4,no_sizes*no_of_methods, no_of_tests);
for iter = 1:no_of_tests
    for k = 1:k_max
    
     gmm_sampling;
    
    end
end

kkk = mean(kiloknaller,1);
plot(kkk)