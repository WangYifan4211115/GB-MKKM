function [res_mean,res_std,res1,res2,res3,res4]= myNMIACCV2(U,Y,numclass)
% 输出[res_mean, res_std, res1, res2, res3, res4]
% res_mean：每个指标在多次运行后的平均值。
% res_std：每个指标在多次运行后的标准差。
% res1：每次迭代计算的分类准确率。
% res2：每次迭代计算的互信息。
% res3：每次迭代计算的纯度。
% res4：每次迭代计算的调整兰德指数。
% 输入(U, Y, numclass)：函数参数：
% U：待聚类的输入数据矩阵。
% Y：数据的真实标签，用于评估聚类性能。
% numclass：聚类目标类别数。

stream = RandStream.getGlobalStream;
reset(stream);  %确保实验可重复
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,numclass);
maxIter = 20;
res1 = zeros(maxIter,1);
res2 = zeros(maxIter,1);
res3 = zeros(maxIter,1);
res4 = zeros(maxIter,1);
for it = 1 : maxIter
    indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',10);
    %% indx = kmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',maxIter);
    indx = indx(:);
    disp(['Y size: ', num2str(size(Y)), ', unique values: ', num2str(unique(Y)')]);
    disp(['indx size: ', num2str(size(indx)), ', unique values: ', num2str(unique(indx)')]);
% try
%     newIndx = bestMap(Y, indx);
% catch ME
%     disp('Error in bestMap:');
%     disp(ME.message);
%     keyboard; % Debug mode
% end
    
    [newIndx] = bestMap(Y,indx);
    res1(it) = mean(Y==newIndx);
    res2(it) = MutualInfo(Y,newIndx);
    res3(it) = purFuc(Y,newIndx);
    res4(it) = adjrandindex(Y,newIndx);
end
res_mean(1) = mean(res1);
res_mean(2) = mean(res2);
res_mean(3) = mean(res3);
res_mean(4) = mean(res4);
res_std(1) = std(res1);
res_std(2) = std(res2);
res_std(3) = std(res3);
res_std(4) = std(res4);