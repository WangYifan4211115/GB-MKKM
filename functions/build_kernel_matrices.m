function KH = build_kernel_matrices(reps)
    cluster_kernel_dir = fullfile(pwd, 'cluster_kernels');
    if ~exist(cluster_kernel_dir, 'dir'), mkdir(cluster_kernel_dir); end
    KernelTypes = {'Linear', 'Polynomial', 'Gaussian'};
    PolynomialDegrees = [2, 4];
    GaussianDegrees = [0.1, 1, 10];
    KH = [];
    for kernel_type = KernelTypes
        switch lower(kernel_type{1})
            case 'linear'
                K0 = reps * reps';
                KH = cat(3, KH, K0);
            case 'polynomial'
                for d = PolynomialDegrees
                    K0 = (reps * reps' + 1).^d;
                    KH = cat(3, KH, K0);
                end
            case 'gaussian'
                D = EuDist2(reps, [], 0);
                for sigma = GaussianDegrees
                    K0 = exp(-D / (2 * sigma^2));
                    KH = cat(3, KH, K0);
                end
        end
    end
end