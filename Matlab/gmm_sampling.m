    
    %Parameters:
    k = 1;
    xmin = -1;
    xmax = 1;
    sdmax = 0.25;
    p_max = 5;
    sample_sizes = [5 50 500 5000];
    train_sets = 100;
    test_size = 500;
    
    sample_it_str = sprintf('sample%d',iter);
    
    % Create means, covariances and prior distributions, Cov = round now
    mu0 = (rand(k,2).*(xmax-xmin))+xmin;
    sd0 = rand(k,1).*sdmax;
    sigma0 = reshape((sd0*[1 0 0 1])',2,2,k);
    p0 = ceil(rand(1,k)*p_max);
    
    mu1 = (rand(k,2).*(xmax-xmin))+xmin;
    sd1 = rand(k,1).*sdmax;
    sigma1 = reshape((sd1*[1 0 0 1])',2,2,k);
    p1 = ceil(rand(1,k)*p_max);
    
    %Step 1: create pdf's
    gmdist0 = gmdistribution(mu0, sigma0, p0);
    gmdist1 = gmdistribution(mu1, sigma1, p1);
    
    %Step 2: sample from distributions for the training set
    
    for siz = 1:length(sample_sizes)
        
        sample_1 = zeros(sample_sizes(1)*2,3,train_sets);
    sample_2 = zeros(sample_sizes(2)*2,3,train_sets);
    sample_3 = zeros(sample_sizes(3)*2,3,train_sets);
    sample_4 = zeros(sample_sizes(4)*2,3,train_sets);

    for i=1:train_sets
        sample_1(:,:,i) = [random(gmdist0,sample_sizes(1)) zeros(sample_sizes(1),1) ; random(gmdist1,sample_sizes(1)) ones(sample_sizes(1),1)];
        sample_2(:,:,i) = [random(gmdist0,sample_sizes(2)) zeros(sample_sizes(2),1) ; random(gmdist1,sample_sizes(2)) ones(sample_sizes(2),1)];
        sample_3(:,:,i) = [random(gmdist0,sample_sizes(3)) zeros(sample_sizes(3),1) ; random(gmdist1,sample_sizes(3)) ones(sample_sizes(3),1)];
        sample_4(:,:,i) = [random(gmdist0,sample_sizes(4)) zeros(sample_sizes(4),1) ; random(gmdist1,sample_sizes(4)) ones(sample_sizes(4),1)];
    end
    
    %Step 3: sample for the test set
    sample_t = [random(gmdist0, test_size) zeros(test_size, 1) ; random(gmdist1, test_size) ones(test_size, 1)];
    
    %Step 4: compute the bayes-optimal outcome of the test set
    pdf0_t = pdf(gmdist0, sample_t(:,1:2));
    pdf1_t = pdf(gmdist1, sample_t(:,1:2));
    [max index] = max([pdf0_t pdf1_t]');
    y_star = index' - 1;
    
    
    %Step 5: Compute the noise: E|t-y*|
    noise = mean(abs(sample_t(:,3)-y_star))