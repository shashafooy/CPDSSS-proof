sigma2=1.5,N=100;
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
figure(1)
semilogx(sample_range,H_unbiased),hold on
semilogx(sample_range,H_bias)
yline(H_analytical,'-')
legend('bias correction','no correction')
ylabel('Nats')
xlabel('samples')
title('Entropy of biased vs unbiased estimators')
hold off

figure(2)
loglog(sample_range,(H_analytical-H_unbiased).^2),hold on
loglog(sample_range,(H_analytical-H_bias).^2),hold off
legend('bias correction','no correction')
title('MSE of biased vs unbiased Entropy Estimator')
ylabel('MSE')
xlabel('samples')

