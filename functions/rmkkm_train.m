function [H_normalized, Sigma, obj] = rmkkm_train(KH, numclass, options)
    % 输入：
    %   KH: 核矩阵堆栈，大小为 [num_samples, num_samples, num_kernels]
    %   numclass: 聚类的目标数量
    %   options: 优化参数
    % 输出：
    %   H_normalized: 归一化的聚类隶属矩阵
    %   Sigma: 核权重
    %   obj: 每次迭代的目标函数值

    [num_samples, ~, num_kernels] = size(KH);

    % 初始化核权重
    Sigma = ones(num_kernels, 1) / num_kernels;

    % 初始化目标函数值存储
    obj = [];

    % 设置正则化参数
    lambda = options.lambda;

    % 初始化聚类隶属矩阵 H
    H = rand(num_samples, numclass);
    H = H ./ sqrt(sum(H.^2, 2)); % 归一化

    for iter = 1:options.nbitermax
        % Step 1: 计算加权核矩阵 K_sigma
        K_sigma = zeros(num_samples, num_samples);
        for p = 1:num_kernels
            K_sigma = K_sigma + Sigma(p)^2 * KH(:, :, p);
        end

        % Step 2: 使用加权核进行K-means更新聚类矩阵 H
        [H, ~] = eigs(K_sigma, numclass, 'LA');

        % Step 3: 计算梯度
        grad = compute_gradient(KH, H, Sigma);

        % Step 4: 更新 Sigma
        Sigma = Sigma - options.threshold * grad; % 梯度下降更新
        Sigma = max(Sigma, 0); % 确保非负
        Sigma = Sigma / sum(Sigma); % 归一化

        % Step 5: 计算当前目标函数值
        obj(iter) = compute_rmkkm_objective(KH, H, Sigma, lambda);

        % Step 6: 检查收敛
        if iter > 1 && abs(obj(iter) - obj(iter - 1)) < options.seuildiffsigma
            fprintf('目标函数在第 %d 次迭代后收敛.\n', iter);
            break;
        end
    end

    % 返回归一化的聚类隶属矩阵
    H_normalized = H ./ sqrt(sum(H.^2, 2));
end


