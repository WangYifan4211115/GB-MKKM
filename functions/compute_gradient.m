function grad = compute_gradient(KH, H, Sigma)
    % 计算核权重的梯度
    % 输入：
    %   KH: 核矩阵堆栈，大小为 [num_samples, num_samples, num_kernels]
    %   H: 聚类分配矩阵，大小为 [num_samples, num_clusters]
    %   Sigma: 当前核权重，大小为 [num_kernels, 1]
    % 输出：
    %   grad: 核权重的梯度

    [num_samples, ~, num_kernels] = size(KH); % 获取核矩阵的维度
    grad = zeros(num_kernels, 1); % 初始化梯度

    for p = 1:num_kernels
        % 提取第p个核矩阵
        Kp = KH(:, :, p);

        % 计算梯度
        grad(p) = -2 * trace(Kp * (H * H') * Sigma(p));
    end
end

