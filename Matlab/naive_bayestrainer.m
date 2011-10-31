
%Step 6: Train Naive Bayes on all training sets
predictions = zeros(test_size*2, length(sample_sizes), train_sets);

for evaliter = 1:train_sets
    nb1 = NaiveBayes.fit(sample_1(:,1:2,evaliter, iter),sample_1(:,3,evaliter, iter),'Distribution','kernel');
    predictions(:,1,evaliter) = predict(nb1, sample_t(:,1:2, iter));
    
    nb2 = NaiveBayes.fit(sample_2(:,1:2,evaliter, iter),sample_2(:,3,evaliter, iter),'Distribution','kernel');
    predictions(:,2,evaliter) = predict(nb2, sample_t(:,1:2, iter));
    
    nb3 = NaiveBayes.fit(sample_3(:,1:2,evaliter, iter),sample_3(:,3,evaliter, iter),'Distribution','kernel');
    predictions(:,3,evaliter) = predict(nb3, sample_t(:,1:2, iter));
    
    nb4 = NaiveBayes.fit(sample_4(:,1:2,evaliter, iter),sample_4(:,3,evaliter, iter),'Distribution','kernel');
    predictions(:,4,evaliter) = predict(nb4, sample_t(:,1:2, iter));
end

if (sum(isnan(predictions(:))) > 0)
    fprintf('NaN detected in predictions\n');
    predictions(isnan(predictions(:))) = 0;
end


%Step 7: Evaluate classifications

% loss = L(y, t)
decomposition(1, 1:no_sizes, k,iter) = mean(mean(abs(predictions - repmat(sample_t(:,3, iter),[1,length(sample_sizes), train_sets]))),3);

ym = round(mean(predictions,3));

% bias = L(y*,ym)
decomposition(2, 1:no_sizes, k,iter) = mean(abs(ym - repmat(sample_t(:,3,iter),[1,length(sample_sizes)])),1);

% variance = ED[L(ym,y)]
decomposition(3, 1:no_sizes, k,iter) = mean(mean(abs(repmat(ym,[1,1,train_sets]) - predictions),1),3);

