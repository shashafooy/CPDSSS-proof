
N=2;
sigma=diag(ones(1,N));
true_H=gaussian_entropy(sigma);
k_range=100:1:100;
iter_range = [5e4 1e5 5e5];
H=zeros(length(iter_range),length(k_range));

iter_idx=1;
for iter=iter_range
k_idx=1;
for k=k_range

    g_term = randn(N,iter);
    H(iter_idx,k_idx)=HShannon_gkNN_estimation(g_term,k);
    k_idx=k_idx+1;
end
iter_idx=iter_idx+1;
end
figure,
plot(k_range,H)
yline(true_H,'-','H(G)')
legend("iter="+iter_range)

ylabel('H($$\hat{\bf G}$$)','Interpreter','Latex')
xlabel('K neighbors')
title("Gaussian estimated entropy N="+N)


