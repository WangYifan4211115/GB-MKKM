function [Hstar, Sigma, obj] = simpleMKKMplot(KH, numclass, option)
    % 输入是核矩阵堆叠KH，聚类数量numclass和配置选项option。
    % 返回聚类指示矩阵Hstar，核权重Sigma，和目标函数值obj。
    
    numker = size(KH, 3); % 确定核矩阵的数量，即KH的第三维大小。
    Sigma = ones(numker, 1) / numker; % 初始化核权重Sigma为均匀分布。

    % 初始化变量
    nloop = 1; % 初始化迭代计数器
    loop = 1;  % 控制主循环的变量

    % 初始化 Kernel K-means
    Kmatrix = sumKbeta(KH, Sigma.^2); % 计算所有核矩阵加权求和后的结果，作为核K均值的输入。
    [Hstar, obj1] = newmykernelkmeans(Kmatrix, numclass); % 核K均值算法。
    obj(nloop) = obj1; % 将初始目标函数值存入obj数组。
    
    [grad] = simpleMKKMGrad(KH, Hstar, Sigma); % 计算核矩阵权重的梯度。
    Sigmaold = Sigma; % 保存当前的核矩阵权重，用于下一次迭代比较。

    % 主循环：更新Sigma并记录目标函数值。
    while loop
        nloop = nloop + 1; % 增加迭代计数器。
        [Sigma, Hstar, obj(nloop)] = simpleMKKMupdate(KH, Sigmaold, grad, obj(nloop-1), numclass, option); % 更新权重。
        
        [grad] = simpleMKKMGrad(KH, Hstar, Sigma); % 更新梯度。
        
        % 检查收敛条件
        if max(abs(Sigma - Sigmaold)) < option.seuildiffsigma
            loop = 0; % 结束循环。
            fprintf(1, 'Variation convergence criteria reached\n');
        end

        % 更新Sigmaold以便下一次迭代。
        Sigmaold = Sigma;
    end
end


