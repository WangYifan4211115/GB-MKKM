function [res_mean, res_std] = NMIACCC(U, y, numclass)
    % 输入检查
    if size(U, 2) ~= numclass
        error('Number of clusters (columns in U) does not match numclass.');
    end

    % 归一化 U
    U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1, size(U, 2));

    % 执行 k-means 聚类
    maxIter = 20;
    nmi_results = zeros(maxIter, 1);
    ari_results = zeros(maxIter, 1);

    for iter = 1:maxIter
        cluster_labels = kmeans(U_normalized, numclass, 'MaxIter', 100, 'Replicates', 5);
        cluster_labels = cluster_labels(:);

        % 计算 NMI 和 ARI
        nmi_results(iter) = MutualInfo(y, cluster_labels);
        ari_results(iter) = adjrandindex(y, cluster_labels);
    end

    % 返回结果
    res_mean = [mean(nmi_results), mean(ari_results)];
    res_std = [std(nmi_results), std(ari_results)];
end


