function [res_mean, res_std, res1, res2, res3, res4] = ACC(U, Y, numclass)
    % 数据归一化
    U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1, numclass);
    
    % 参数设置
    maxIter = 20; % 最大迭代次数
    res1 = zeros(maxIter, 1); % 准确率
    res2 = zeros(maxIter, 1); % 互信息
    res3 = zeros(maxIter, 1); % 纯度
    res4 = zeros(maxIter, 1); % 调整兰德指数

    % 聚类评估
    for it = 1:maxIter
        % 使用 litekmeans 聚类
        indx = litekmeans(U_normalized, numclass, 'MaxIter', 100, 'Replicates', 10);
        indx = indx(:);

        % 映射聚类结果到真实标签
        newIndx = bestMap(Y, indx);

        % 计算指标
        res1(it) = mean(Y == newIndx);                % 准确率
        res2(it) = MutualInfo(Y, newIndx);            % 互信息
        res3(it) = purFuc(Y, newIndx);                % 纯度
        res4(it) = adjrandindex(Y, newIndx);          % 调整兰德指数
    end

    % 计算平均值和标准差
    res_mean(1) = mean(res1); % 平均准确率
    res_mean(2) = mean(res2); % 平均互信息
    res_mean(3) = mean(res3); % 平均纯度
    res_mean(4) = mean(res4); % 平均调整兰德指数

    res_std(1) = std(res1); % 准确率标准差
    res_std(2) = std(res2); % 互信息标准差
    res_std(3) = std(res3); % 纯度标准差
    res_std(4) = std(res4); % 调整兰德指数标准差
end

