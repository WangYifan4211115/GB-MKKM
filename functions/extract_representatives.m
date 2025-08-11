function [reps, labels] = extract_representatives(X, y, clusters)
    num_features = size(X, 2);
    reps = zeros(length(clusters), num_features);
    labels = zeros(length(clusters), 1);
    for i = 1:length(clusters)
        indices = clusters{i}.indices;
        features = X(indices, :);
        center = mean(features, 1);
        distances = sqrt(sum((features - center).^2, 2));
        [~, min_idx] = min(distances);
        rep_idx = indices(min_idx);
        reps(i, :) = X(rep_idx, :);
        labels(i) = y(rep_idx);
        fprintf('Cluster %d: Representative sample index = %d, Label = %d\n', i, rep_idx, y(rep_idx));
    end
end
