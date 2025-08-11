%离散数据或者图像数据可以直接应用标准核函数
function [Hstar,Sigma,obj] = GBsimpleMKKM(KH,numclass,option)
%输入是核矩阵堆叠KH，聚类数量numclass和配置选项option。
% 返回聚类指示矩阵Hstar，核权重Sigma，和目标函数值obj。
numker = size(KH,3); %确定核矩阵的数量，即KH的第三维大小。
Sigma = ones(numker,1)/numker; %初始化核权重Sigma为均匀分布

%--------------------------------------------------------------------------------
% Options used in subroutines子程序中使用的选项
%--------------------------------------------------------------------------------
%这部分代码检查option中是否设置了特定字段，如果没有，则使用默认值。这些设置用于算法的细粒度控制，例如黄金分割搜索的精度
if ~isfield(option,'goldensearch_deltmax')
    option.goldensearch_deltmax=5e-2;
end
if ~isfield(option,'goldensearchmax')
    optiongoldensearchmax=1e-8;
end
if ~isfield(option,'firstbasevariable')
    option.firstbasevariable='first';
end
%--------------------------------------------------------------------------------

nloop = 1;  %初始化迭代计数器
loop = 1;  %控制主循环的变量
goldensearch_deltmaxinit = option.goldensearch_deltmax;  %保存初始的黄金搜索精度参数，用于调整搜索精度

%-----------------------------------------
% Initializing Kernel K-means
%------------------------------------------
Kmatrix = sumKbeta(KH,Sigma.^2); %计算所有核矩阵加权求和后的结果，作为核K均值的输入
[Hstar,obj1]= mykernelkmeans(Kmatrix,numclass); %执行核K均值聚类算法，得到聚类结果Hstar和目标函数初始值obj1
%[Hstar,obj1]= newmykernelkmeans(Kmatrix,numclass);%new
obj(nloop) = obj1; %将初始目标函数值存入obj数组
% [res_mean(:,nloop),res_std(:,nloop)] = myNMIACCV2(Hstar,Y,numclass);
[grad] = simpleMKKMGrad(KH,Hstar,Sigma);  %计算核矩阵权重的梯度

Sigmaold  = Sigma;  %保存当前的核矩阵权重，用于下一次迭代比较
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop
    nloop = nloop+1;  %增加迭代计数器
    %-----------------------------------------
    % Update weigths Sigma
    %-----------------------------------------
    [Sigma,Hstar,obj(nloop)] = simpleMKKMupdate(KH,Sigmaold,grad,obj(nloop-1),numclass,option); % 更新核矩阵权重和聚类结果
    
    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------
    if max(abs(Sigma-Sigmaold))<option.numericalprecision &&...
            option.goldensearch_deltmax > optiongoldensearchmax  %% 如果核矩阵权重变化很小且当前搜索精度超过最小值
        option.goldensearch_deltmax=option.goldensearch_deltmax/10;  %% 减小搜索步长，增加精度
    elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit  %% 如果当前搜索精度已调整
        option.goldensearch_deltmax*10;  %% 恢复初始搜索步长
    end
    
    [grad] = simpleMKKMGrad(KH,Hstar,Sigma);  % % 重新计算梯度
    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
        if  max(abs(Sigma-Sigmaold))<option.seuildiffsigma   %% 如果核矩阵权重变化小于阈值
            loop = 0;  %% 结束循环
            fprintf(1,'variation convergence criteria reached \n');  %% 打印收敛信息
        end
    
    
    %-----------------------------------------------------
    % Updating Variables
    %----------------------------------------------------
    Sigmaold  = Sigma;  %% 更新旧的核矩阵权重为当前值，准备下一次迭代
end


