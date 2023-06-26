close all
sigma2=1.5,N=100;


for N=[100 500 1000]


Sigma_cov = sigma2*eye(N);
H_analytical = N/2*(log(2*pi)+1)+1/2*logdet(Sigma_cov);

idx=0;
sample_range=floor(logspace(log10(N+1),6,200));
for k=sample_range
    idx=idx+1;
    
    x=normrnd(0,sqrt(sigma2),N,k);
    S=(k-1)*cov(x');
    H=0.5*N*(1+log(pi)) + 0.5*logdet(S);
    
    for i=1:N
        H = H - 0.5*psi(0.5*(k-i));
    end
    H_unbiased(idx)=H;
    H_bias(idx)=0.5*N*(1+log(2*pi)) + 0.5*logdet(S/(k-1));
end

MSE_unbiased=(H_analytical-H_unbiased).^2;
MSE_bias=(H_analytical-H_bias).^2;

figure(1)
loglog(sample_range,MSE_unbiased),hold on

figure(2)
loglog(sample_range,MSE_bias),hold on
end

figure(1)
hold off
title('MSE for unbiased estimator for different dimensions')
legend('N=100','N=500','N=1000')
ylabel('MSE')
xlabel('samples')

figure(2)
hold off
title('MSE for biased estimator for different dimensions')
legend('N=100','N=500','N=1000')
ylabel('MSE')
xlabel('samples')



figure(2)
loglog(sample_range,(H_analytical-H_unbiased).^2),hold on
loglog(sample_range,(H_analytical-H_bias).^2),hold off
legend('bias correction','no correction')
title('MSE of biased vs unbiased Entropy Estimator')
ylabel('MSE')
xlabel('samples')

