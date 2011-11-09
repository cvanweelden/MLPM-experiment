%Step 6: run kNN (with oracle value for k)
predictions = zeros(test_size*2, length(sample_sizes), train_sets);
best_loss = Inf .* ones(length(sample_sizes), no_of_tests, train_sets);


for evaliter = 1:train_sets
    for kN = 1:round(sample_sizes(1)*2/10):sample_sizes(1)*2
        idx = knnsearch(sample_1(:, 1:2, evaliter, iter), sample_t(:, 1:2, iter), 'k', kN);
        labels = sample_1(:,3,evaliter,iter);
        cur_predictions = mode(labels(idx),2);
        curloss = mean(abs(cur_predictions - sample_t(:,3, iter)));
        if curloss < best_loss(1, iter, evaliter)
            best_loss(1,iter,evaliter) = curloss;
            predictions(:,1,evaliter) = cur_predictions;
            KKNN(1,k,iter,evaliter) = kN;
        end
    end
    
    for kN = 1:round(sample_sizes(2)*2/10):sample_sizes(2)*2
        idx = knnsearch(sample_2(:, 1:2, evaliter, iter), sample_t(:, 1:2, iter), 'k', kN);
        labels = sample_2(:,3,evaliter,iter);
        cur_predictions = mode(labels(idx),2);
        curloss = mean(abs(cur_predictions - sample_t(:,3, iter)));
        if curloss < best_loss(2, iter, evaliter)
            best_loss(2,iter,evaliter) = curloss;
            predictions(:,2,evaliter) = cur_predictions;
            KKNN(2,k,iter,evaliter) = kN;
        end
    end
    
    for kN = 1:round(sample_sizes(3)*2/10):sample_sizes(3)*2
        idx = knnsearch(sample_3(:, 1:2, evaliter, iter), sample_t(:, 1:2, iter), 'k', kN);
        labels = sample_3(:,3,evaliter,iter);
        cur_predictions = mode(labels(idx),2);
        curloss = mean(abs(cur_predictions - sample_t(:,3, iter)));
        if curloss < best_loss(3, iter, evaliter)
            best_loss(3,iter,evaliter) = curloss;
            predictions(:,3,evaliter) = cur_predictions;
            KKNN(3,k,iter,evaliter) = kN;
        end
    end
    
    for kN = 1:round(sample_sizes(4)*2/10):sample_sizes(4)*2
        idx = knnsearch(sample_4(:, 1:2, evaliter, iter), sample_t(:, 1:2, iter), 'k', kN);
        labels = sample_4(:,3,evaliter,iter);
        cur_predictions = mode(labels(idx),2);
        curloss = mean(abs(cur_predictions - sample_t(:,3, iter)));
        if curloss < best_loss(4, iter, evaliter)
            best_loss(4,iter,evaliter) = curloss;
            predictions(:,4,evaliter) = cur_predictions;
            KKNN(4,k,iter,evaliter) = kN;
        end
    end
end

%Step 7: Evaluate classifications

% loss = L(y, t)
decomposition(1, 1:no_sizes, k,iter) = mean(mean(abs(predictions - repmat(sample_t(:,3, iter),[1,length(sample_sizes), train_sets]))),3);

ym = round(mean(predictions,3));

% bias = L(y*,ym)
decomposition(2, 1:no_sizes, k,iter) = mean(abs(ym - repmat(sample_t(:,3,iter),[1,length(sample_sizes)])),1);

% variance = ED[L(ym,y)]
decomposition(3, 1:no_sizes, k,iter) = mean(mean(abs(repmat(ym,[1,1,train_sets]) - predictions),1),3);