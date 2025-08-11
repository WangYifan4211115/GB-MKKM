function [H_normalized,obj]= newmykernelkmeans(K,cluster_count)

K = (K+K')/2;
opt.disp = 0;
[H,~] = eigs(K,cluster_count,'LA',opt);
H_normalized = H;
% lambda = 0.1; % 正则化强度
% D = diag(sum(K, 2)); % 构造度矩阵 (n x n)
% I_projected = lambda * H' * D * H; % 将 D 映射到 k x k
% obj = trace(H' * K * H - I_projected); % 确保维度一致
lambda = 0.1; % 正则化强度
I = lambda * eye(size(H' * K * H, 1)); % 构造 k x k 单位矩阵
obj = trace(H' * K * H - I); % 确保维度一致
