function clusters = initializeGB(X, y)
    num_samples = size(X, 1);
    num_clusters = ceil(sqrt(num_samples));
    clusters = cell(num_clusters, 1);
    sample_importance = mean(X, 2);
    [~, sorted_indices] = sort(sample_importance, 'descend');
    for i = 1:num_clusters
        clusters{i}.indices = [];
        clusters{i}.features = [];
        clusters{i}.labels = [];
    end
    for i = 1:num_samples
        target_cluster = mod(i - 1, num_clusters) + 1;
        idx = sorted_indices(i);
        clusters{target_cluster}.indices(end + 1) = idx;
        clusters{target_cluster}.features = [clusters{target_cluster}.features; X(idx, :)];
        clusters{target_cluster}.labels = [clusters{target_cluster}.labels; y(idx)];
    end
end
