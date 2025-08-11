function current_clusters = adaptively_split_clusters(X, y, clusters)
    max_iterations = 500;
    min_samples_per_ball = 1;
    current_clusters = clusters;
    for iteration = 1:max_iterations
        fprintf('--- Iteration %d ---\n', iteration);
        new_clusters = {};
        cluster_changed = false;
        consistencies = zeros(length(current_clusters), 1);
        for i = 1:length(current_clusters)
            features = X(current_clusters{i}.indices, :);
            avg_distance = mean(pdist(features));
            max_distance = max(pdist(features));
            consistencies(i) = avg_distance / max_distance;
        end
        consistency_threshold = median(consistencies);
        fprintf('Consistency threshold (median): %.4f\n', consistency_threshold);
        for i = 1:length(current_clusters)
            cluster = current_clusters{i};
            indices = cluster.indices;
            if length(indices) <= min_samples_per_ball
                new_clusters{end+1} = cluster;
                continue;
            end
            consistency = consistencies(i);
            fprintf('Consistency : %.4f\n', consistency);
            if consistency < 2 * consistency_threshold
                [sub_idx, ~] = kmeans(X(indices, :), 2, 'Start', 'plus', 'Replicates', 1);
                sub_indices_1 = indices(sub_idx == 1);
                sub_indices_2 = indices(sub_idx == 2);
                if length(sub_indices_1) > min_samples_per_ball && length(sub_indices_2) > min_samples_per_ball
                    cluster_changed = true;
                    new_clusters{end+1} = struct('indices', sub_indices_1, 'features', X(sub_indices_1, :), 'labels', y(sub_indices_1));
                    new_clusters{end+1} = struct('indices', sub_indices_2, 'features', X(sub_indices_2, :), 'labels', y(sub_indices_2));
                else
                    new_clusters{end+1} = cluster;
                end
            else
                new_clusters{end+1} = cluster;
            end
        end
        current_clusters = new_clusters;
        if ~cluster_changed
            fprintf('无新粒球被分裂，停止迭代\n');
            break;
        end
    end
end

