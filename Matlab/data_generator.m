function [points, label] = data_generator(func, noise, min_range, max_range, sample_size, mode)
    if mode == 'regression'
        sample = random('unif', min_range, max_range, sample_size, 1);
        points = func(sample);
        points = noise(points);
        label = NaN;
    else
        % mode == 'classification'
        
    end
    
return