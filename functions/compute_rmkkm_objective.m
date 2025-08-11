
function obj = compute_rmkkm_objective(KH, H, Sigma, lambda)
    % 计算RMKKM的目标函数值
    % 输入：
    %   KH: 核矩阵堆栈，大小为 [num_samples, num_samples, num_kernels]
    %   H: 聚类分配矩阵，大小为 [num_samples, num_clusters]
    %   Sigma: 当前核权重，大小为 [num_kernels, 1]
    %   lambda: 正则化参数
    % 输出：
    %   obj: 当前目标函数值

    % 初始化加权核矩阵 K_sigma
    [num_samples, ~, num_kernels] = size(KH);
    K_sigma = zeros(num_samples, num_samples);

    % 计算加权核矩阵 K_sigma
    for p = 1:num_kernels
        K_sigma = K_sigma + Sigma(p)^2 * KH(:, :, p);
    end

    % 计算目标函数值
    obj = trace(K_sigma * (H * H')) + lambda * sum(Sigma.^2);
end

