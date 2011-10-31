
%Step 6: Train SVM on all training sets
predictions = zeros(test_size*2, length(sample_sizes), train_sets);

for evaliter = 1:train_sets
    svm1 = svmtrain(sample_1(:,1:2,evaliter, iter),sample_1(:,3,evaliter, iter));
    predictions(:,1,evaliter) = svmclassify(svm1, sample_t(:,1:2, iter));
    
    svm2 = svmtrain(sample_2(:,1:2,evaliter, iter),sample_2(:,3,evaliter, iter));
    predictions(:,2,evaliter) = svmclassify(svm2, sample_t(:,1:2, iter));
    
    svm3 = svmtrain(sample_3(:,1:2,evaliter, iter),sample_3(:,3,evaliter, iter));
    predictions(:,3,evaliter) = svmclassify(svm3, sample_t(:,1:2, iter));
    
    svm4 = svmtrain(sample_4(:,1:2,evaliter, iter),sample_4(:,3,evaliter, iter));
    predictions(:,4,evaliter) = svmclassify(svm4, sample_t(:,1:2, iter));
end

if (sum(isnan(predictions(:))) > 0)
    fprintf('NaN detected in predictions\n');
    predictions(isnan(predictions(:))) = 0;
end


%Step 7: Evaluate classifications

% loss = L(y, t)
decomposition(1, (no_sizes+1):no_sizes*2, iter) = mean(mean(abs(predictions - repmat(sample_t(:,3, iter),[1,length(sample_sizes), train_sets]))),3);

ym = round(mean(predictions,3));

% bias = L(y*,ym)
decomposition(2, (no_sizes+1):no_sizes*2, iter) = mean(abs(ym - repmat(sample_t(:,3,iter),[1,length(sample_sizes)])),1);

% variance = ED[L(ym,y)]
decomposition(3, (no_sizes+1):no_sizes*2, iter) = mean(mean(abs(repmat(ym,[1,1,train_sets]) - predictions),1),3);

