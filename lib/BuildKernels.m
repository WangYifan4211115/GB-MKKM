function [f] = BuildKernels(dataset)  %定义了一个函数，它接收一个参数dataset,并返回一个变量f（f被赋值为1但是未进一步使用）
global data_dir
global kernel_dir
global KernelTypes KernelPostProcessTypes PolynomialDegrees PolyPlusDegrees GaussianDegrees 
%声明了一系列全局变量，这些变量将在函数中用于存储路径，和类型，后处理类型以及多项式和高斯核的参数
global kernel_path_part_1 %声明了一个全局变量，用于存储核文件路径的一部分
%create a folder named by the name of dataset创建一个以数据集名称命名而定文件夹中

data_dir = fullfile(pwd, '..', 'data'); %设置data_dir为包含数据集文件的目录路径，这里使用fullfile函数来构造跨平台的路径

kernel_dir = fullfile(data_dir, [dataset, '_kernel']);%设置kernel_dir为存储核文件的 目录路径，该路径以数据集名称加上_kernel后缀命名
if exist(kernel_dir, 'dir') == 0  %检查kernel_dir是否存在
    mkdir(kernel_dir);  %如果kernel_dir不存在，则创建它
end

load(fullfile(data_dir, dataset));  %加载数据集文件，这里假设数据集文件与数据集名称同名，并位于data_dir目录下
if exist('X', 'var')  %检查x是否存在
    % error([dataset, ' do not have variable X!']);
    is_mv = 0;%如果x存在，则设置is_mv（多视图标志）为0，表示数据集不是多视图数据集
elseif exist('views', 'var')  %如果x不存在但views存在，则进入此分支
        % error([dataset, ' do not have variable views!']);
        is_mv = 1;  %设置is_mv为1，表示数据集是多视图数据集
        for iView = 1:length(views)  %遍历所有视图
            if eval(sprintf('exist(''%s'', ''var'') == 0', ['X', num2str(iView)]))%使用eval检查每个视图对应的x变量是否存在
                eval(sprintf('error([dataset, '' do not have variable %s!'']);', ['X', num2str(iView)])); %如果某个视图的x不存在，则报错
            end
        end
else %如果既不是单视图数据集也不是多视图数据集，(x和view都没有)则报错
    error([dataset, ' do not have variable X or views!']);
end

KernelTypes = {'Linear', 'PolyPlus', 'Polynomial', 'Gaussian'}; %定义核类型
KernelPostProcessTypes = {'Sample-Scale'};  %定义核后处理类型
PolynomialDegrees = [2, 4];%定义多项式核的度数
PolyPlusDegrees = [2, 4]; %定义polyplus核的度数
GaussianDegrees = [0.01, 0.05, 0.1, 1, 10, 50, 100];  %定义高斯核的带宽参数


if is_mv  %如果是多视图数据集
    for iView = 1:length(views)  %遍历所有视图
        kernel_path_part_1 = [dataset, '_kernel_view', num2str(iView) '_'];%设置当前视图的和文件路径前缀
        clear X;  %清除之前的x变量
        eval(sprintf('X = %s;', ['X', num2str(iView)]));  %使用eval加载当前视图的x变量
        BuildSingleKernels(X);  %调用BuildSingleKernels函数为当前视图构建核矩阵
    end
else  %如果是但是图数据集
    kernel_path_part_1 = [dataset, '_kernel_'];  %设置核文件路径前缀
    BuildSingleKernels(X); %调用BuildSingleKernels函数为但是图数据集构造核矩阵
end

f = 1;%将f设置为1
end



function f2 = BuildSingleKernels(X) 
%定义了一个名为BuildSingleKernels的函数，他接受一个参数x,(通常是一个数据矩阵)，并返回一个值f2,（f2被设置为1，可能表示函数成功执行）
%还是声明接下来要使用的全局变量，这些变量在其他函数或者工作空间中也可以访问和修改
global kernel_dir
global KernelTypes KernelPostProcessTypes PolynomialDegrees PolyPlusDegrees GaussianDegrees 
%分别存储和类型，后处理类型，多项式核的度数，polyplus核的度数
global kernel_path_part_1 % 构建核文件路径的第一部分

for kernel_type = KernelTypes  %遍历所有核类型
    kernel_option = [];  %初始化一个空的结构体kernel_option，用于存储当前和类型的选项
    %根据核类型构造核
    switch lower(kernel_type{1}) %使用aswitch语句根据kernel_type的值选择相应的代码块执行
        case lower('Linear')%每个case对应一种核类型
            
            %对于每种核类型，代码会遍历所有后处理类型，为每个组合构建核矩阵
            %使用fullfile和stract函数构建核文件的完整路径
%检查文件是否存在(使用exit函数)，如果不存在，则使用constructkernel构建核矩阵，然后使用KernelNormalize进行后处理，然后save保存在磁盘上。
            kernel_option.KernelType = 'Linear'; 
            kernel_path_part_2 = 'linear_';
            for iPost = KernelPostProcessTypes
                kernel_path_part_3 = ['post_', iPost{1}];
                kernel_file = fullfile(kernel_dir, strcat(kernel_path_part_1, kernel_path_part_2, kernel_path_part_3, '.mat'));
                if ~exist(kernel_file, 'file')
                    K0 = constructKernel(X, [], kernel_option);
                    K = KernelNormalize(K0, iPost{1});%#ok
                    save(kernel_file, 'K');
                end
            end
        case lower('Polynomial')
            kernel_option.KernelType = 'Polynomial';
            for iKernelParam = PolynomialDegrees
                kernel_option.d = iKernelParam;
                kernel_path_part_2 = ['polynomial_', num2str(iKernelParam), '_'];
                for iPost = KernelPostProcessTypes
                    kernel_path_part_3 = ['post_', iPost{1}];
                    kernel_file = fullfile(kernel_dir, strcat(kernel_path_part_1, kernel_path_part_2, kernel_path_part_3, '.mat'));
                    if ~exist(kernel_file, 'file')
                        K0 = constructKernel(X, [], kernel_option);
                        K = KernelNormalize(K0, iPost{1});%#ok
                        save(kernel_file, 'K');
                    end
                end
            end
        case lower('PolyPlus')
            kernel_option.KernelType = 'PolyPlus';
            for iKernelParam = PolyPlusDegrees
                kernel_option.d = iKernelParam;
                kernel_path_part_2 = ['polyplus_', num2str(iKernelParam), '_'];
                for iPost = KernelPostProcessTypes
                    kernel_path_part_3 = ['post_', iPost{1}];
                    kernel_file = fullfile(kernel_dir, strcat(kernel_path_part_1, kernel_path_part_2, kernel_path_part_3, '.mat'));
                    if ~exist(kernel_file, 'file')
                        K0 = constructKernel(X, [], kernel_option);
                        K = KernelNormalize(K0, iPost{1});%#ok
                        save(kernel_file, 'K');
                    end
                end
            end
        
        %特殊处理，对于高斯核，在计算核矩阵之前，会根据数据的最大欧氏距离调整尺度参数t
        case lower('Gaussian')
            kernel_option.KernelType = 'Gaussian';
            for iKernelParam = GaussianDegrees
                kernel_option.t = iKernelParam;
                kernel_path_part_2 = ['gaussian_', num2str(iKernelParam), '_'];
                for iPost = KernelPostProcessTypes
                    kernel_path_part_3 = ['post_', iPost{1}];
                    kernel_file = fullfile(kernel_dir, strcat(kernel_path_part_1, kernel_path_part_2, kernel_path_part_3, '.mat'));
                    if ~exist(kernel_file, 'file')
                        D = EuDist2(X, [], 0);
                        max_D = max(D(:));
                        max_D = sqrt(max_D);
                        kernel_option.t = kernel_option.t * max_D;
                        K0 = constructKernel(X, [], kernel_option);
                        % K0 = exp(- D / (2 * kernel_option.t^2) );
                        K = KernelNormalize(K0, iPost{1});%#ok
                        save(kernel_file, 'K');
                    end
                end
            end
        
        %对于文本核，要检查数据是否为全整数(可能是为了区分文本数据和数值数据)，然后调用build_kernel_text函数
        case lower('text')
            kernel_path_part_2 = ['text'];
            t1 = sum(X(:));
            if fix(t1) == t1
                isTF = 1;
            else
                isTF = 0;
            end
            kernel_file = fullfile(kernel_dir, strcat(kernel_path_part_1, kernel_path_part_2));
            %生成文本数据的核矩阵
            build_kernels_text(X, isTF, kernel_file);
        otherwise
            error('KernelType does not exist!');%如果遇到未知的核类型，则使用error函数抛出错误
    end
end
f2 = 1;
end