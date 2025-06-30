clear;clc;
rng(42); 
dataName = 'jaffe_213n_676d_10c';
load([dataName,'_uni'],'X','y');

%% 1.initialization 
numclass = length(unique(y)); 
y(y<1) = numclass;
[num_samples, num_features] = size(X); 
sample_indices = (1:num_samples)'; 
find_sample = @(idx) X(idx, :); 
find_label = @(idx) y(idx); 

%粗化粒球
clusters = initializeGB(X, y);
%自适应分裂
current_clusters = adaptively_split_clusters(X, y, clusters);
%提取代表样本
[cluster_representatives, cluster_labels] = extract_representatives(X, y, current_clusters);
%粒球核矩阵计算
KH = build_kernel_matrices(cluster_representatives);

numclass1 = length(unique(cluster_labels));
cluster_labels(cluster_labels < 1) = numclass1; % 将小于1的标签替换为numclass
numker = size(KH, 3); % 获取核矩阵堆栈中的核数量
num = size(KH, 1); % 获取样本数量
KH = kcenter(KH); % 对核矩阵进行中心化处理
KH = knorm(KH); % 对核矩阵进行归一化处理


%初始化存储变量
num_runs = 1;
nmi_results = zeros(num_runs, 1);
acc_results = zeros(num_runs, 1);
ari_results = zeros(num_runs, 1);
acc_res_std = zeros(num_runs, 1);
nmi_res_std = zeros(num_runs, 1);
ari_res_std = zeros(num_runs, 1);
timecosts = zeros(num_runs, 1);

%设置优化参数
options.seuildiffsigma = 1e-5;
options.goldensearch_deltmax = 1e-3;
options.numericalprecision = 1e-16;
options.firstbasevariable = 'first';
options.nbitermax = 500;
options.seuil = 0;
options.seuilitermax = 10;
options.miniter = 0;
options.threshold = 1e-4;
qnorm = 2; % 范数类型

for i = 1:num_runs
    % 预处理：核矩阵中心化 & 归一化
    KH = kcenter(KH);
    KH = knorm(KH);
  
    % 运行 SimpleMKKM 聚类
    tic;
    [H_normalized, Sigma, obj] = GBsimpleMKKM(KH, numclass1, options);;
    timecosts2(i) = toc;
    
    % 计算 
    [res_mean, res_std] = myNMIACCV2(H_normalized, cluster_labels, numclass1);
    
    % 存储结果
    acc_results(i) = res_mean(1); % ACC
    nmi_results(i) = res_mean(2); % NMI
    ari_results(i) = res_mean(3); % ARI
    acc_res_std(i)=res_std(1);%std_acc
    nmi_res_std(i)=res_std(2);%std_nmi
    ari_res_std(i)=res_std(3);%std_ari
end

% 计算均值和标准差
mean_acc = mean(acc_results);
std_acc = mean(acc_res_std);

mean_nmi = mean(nmi_results);
std_nmi = mean(nmi_res_std);

mean_ari = mean(ari_results);
std_ari = mean(ari_res_std);


mean_time = mean(timecosts2);
% 显示最终结果
fprintf('\nClustering completed after %d runs.\n', num_runs);
fprintf('Mean NMI: %.4f, Std NMI: %.4f\n', mean_nmi, std_nmi);
fprintf('Mean ACC: %.4f, Std ACC: %.4f\n', mean_acc, std_acc);
fprintf('Mean Time Cost: %.4f sec, Std Time Cost: %.4f sec\n', mean_time); 

























