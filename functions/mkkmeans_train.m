
%这段代码是多核K均值聚类（MKKM）算法的MATLAB实现,从多个核矩阵中学习聚类模型。代码流程包括初始化、迭代更新和收敛检测
function [H_normalized,theta,objective]= mkkmeans_train(Km,cluster_count)
%输入参数:
%Km: 包含多个核矩阵的三维数组。
%cluster_count: 聚类数目
%输出参数:
%H_normalized: 归一化的聚类指示矩阵。
%theta: 各核矩阵的权重。
%objective: 目标函数值的历史，用于监控算法进展

%--------初始化变量-------------
numker = size(Km, 3);  %核矩阵的数量
theta = ones(numker,1)/numker;  %初始化所有核的权重，均等分配。
K_theta = mycombFun(Km, theta.^2); %使用初始权重组合所有核矩阵

%-------设置选项和变量-----------
opt.disp = 0; %禁用某些函数的显示输出
iteration_count = 0; %记录迭代次数
flag =1; %控制循环的条件变量

% %%---
% maxIter = 30;
% res_mean = zeros(4,maxIter);
% res_std = zeros(4,maxIter);

while flag  %使用一个布尔变量flag控制循环。flag初始化为1，当满足停止条件时设为0以终止循环
    iteration_count = iteration_count+1;  %在每次循环开始，迭代计数器iteration_count增加1，用于跟踪迭代的次数。
    fprintf(1, 'running iteration %d...\n', iteration_count);  %打印当前的迭代次数，帮助用户跟踪算法进度
    [H, ~] = eigs(K_theta, cluster_count, 'LA', opt);  
    %使用eigs函数计算核矩阵K_theta的最大的cluster_count个特征向量，这些向量构成了聚类指示矩阵H。
    % 参数'LA'指定计算最大特征值，opt是控制输出显示的选项
    % [res_mean(:,iteration_count),res_std(:,iteration_count)] = myNMIACCV2(H,Y,cluster_count);
    %     resH(iteration_count,:) = myNMIACC(H,Y,cluster_count);
    %     Q = zeros(numker);
    %     for m = 1:numker
    %         Q(m, m) = trace(Km(:, :, m)) - trace(H' * Km(:, :, m) * H);
    %     end
    %     res = mskqpopt(Q, zeros(numker, 1), ones(1, numker), 1, 1, zeros(numker, 1), ones(numker, 1), [], 'minimize echo(0)');
    %     theta = res.sol.itr.xx;
    [theta] = updateabsentkernelweightsV3(H,Km); %调用自定义函数，输入当前的聚类指示矩阵H和原始核矩阵Km，输出新的核权重theta
    K_theta = mycombFun(Km, theta.^2); %通过自定义函数mycombFun重新计算核矩阵的组合
    % 函数根据核权重的平方theta.^2和原始核矩阵Km，计算加权和形成新的组合核矩阵K_theta
    objective(iteration_count) = -trace(H' * K_theta * H) + trace(K_theta);
    %计算目标函数的当前值，其中包括聚类指示矩阵H与组合核矩阵K_theta的迹乘积。目标函数值存储在数组objective中，索引为当前迭代次数
    

    %改10-4为10-3
    if iteration_count>2 && (abs((objective(iteration_count-1)-objective(iteration_count))...
            /(objective(iteration_count-1)))<1e-4|| iteration_count>50)
        flag =0;
    end
    
    %条件判断:
    %iteration_count > 2: 确保至少运行了几次迭代（这里是3次）。
    %abs((objective(iteration_count-1) - objective(iteration_count)) / (objective(iteration_count-1))) < 1e-4: 
    % 检查目标函数的相对变化是否小于阈值1e-4，如果是，认为目标函数已收敛。
    %iteration_count > 50: 如果迭代次数超过50，无论目标函数是否收敛，都停止迭代。
    %flag = 0;: 如果满足上述任一条件，设置flag为0，结束循环。    
    
    %     if iteration_count>=maxIter
    %         flag =0;
    %     end
    %     if iteration_count>100
    %         flag =0;
    %     end
end
% H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1, cluster_count);
H_normalized = H;

%整个循环结构通过不断优化核权重和聚类指示矩阵，目标是最小化聚类指示矩阵与组合核矩阵之间的Frobenius范数，从而达到优化聚类效果的目的。



