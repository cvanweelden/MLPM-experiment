
%Step 6: Train Naive Bayes on all training sets
tic
predictions = zeros(test_size*2, length(sample_sizes), train_sets);

for i = 1:train_sets
    bla = 'iteration no'
    i
    nb1 = NaiveBayes.fit(sample_1(:,1:2,i),sample_1(:,3,i),'Distribution','kernel');
    predictions(:,1,i) = predict(nb1, sample_t(:,1:2));
    
    nb2 = NaiveBayes.fit(sample_2(:,1:2,i),sample_2(:,3,i),'Distribution','kernel');
    predictions(:,2,i) = predict(nb2, sample_t(:,1:2));
    
    nb3 = NaiveBayes.fit(sample_3(:,1:2,i),sample_3(:,3,i),'Distribution','kernel');
    predictions(:,3,i) = predict(nb3, sample_t(:,1:2));
    
    nb4 = NaiveBayes.fit(sample_4(:,1:2,i),sample_4(:,3,i),'Distribution','kernel');
    predictions(:,4,i) = predict(nb4, sample_t(:,1:2));
end

%Step 7: Evaluate classifications

% L(y, t)
loss = mean(abs(predictions - repmat(sample_t(:,3),[1,length(sample_sizes), train_sets])))

ym = round(mean(predictions,3))

% bias = L(y*,ym)
bias = mean(abs(ym - repmat(sample_t(:,3),[1,length(sample_sizes)])),1)

% variance = ED[L(ym,y)]
variance = mean(mean(abs(repmat(ym,[1,1,train_sets]) - predictions),1),3)

time_spent = toc