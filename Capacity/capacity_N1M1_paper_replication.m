%% M=1,N=1
M=1;
N=1;
rho=0.5;
T_max=40;
C_L=zeros(T_max,1);
% H=(normrnd(0,1)+1i*normrnd(0,sqrt(sigma2),N/L,iter))/sqrt(2);


for T=N+1:T_max
    S=ones(N,T);
    [Phi, V, ]=svd(S);
    cap=-T*log2(exp(1)) - log2(1+rho*T);
%     MI=-T*log2(exp(1)) - log2(1+rho*V(1));
    % Matlab incomplete gamma(x,a) normalized by Gamma(a), undo this here
    % Referencing the lower capacity limit eq(12)
    fun=@(lam) ((T-1)*exp(-lam/(1+rho*T)).*gamma(T-1).*gammainc(rho*T*lam/(1+rho*T),T-1))./...
        (gamma(T)*(1+rho*T)*(rho*T/(1+rho*T))^(T-1))...
        .*log2(((T-1)*exp(-lam/(1+rho*T)).*gamma(T-1).*gammainc(rho*T*lam/(1+rho*T),T-1))./...
        ((1+rho*T)*(rho*T*lam/(1+rho*T)).^(T-1)));
    C_L(T)=(cap - integral(fun,0,400))/T;
end



figure(1)
hold on
plot(C_L)
hold off
ylim([0 0.9])
xlim([1 40])
xlabel('T peroids')
ylabel('bits/T')
title('Lower bound for M=1,N=1')
shg

%% N=2, M=1
N=2;
rho=1;
T_max=20;
C_L=zeros(T_max,1);
for T=N+1:T_max
    S=ones(N,T);
    [Phi, V, ]=svd(S);
    cap=-2*T*log2(exp(1)) - log2(1+rho*T);
%     MI=-T*log2(exp(1)) - log2(1+rho*V(1));
    % Matlab incomplete gamma(x,a) normalized by Gamma(a), undo this here
    % Referencing the lower capacity limit eq(12)
    fun=@(lam2,lam1) (exp(-lam1-lam2).*(lam1.*lam2).^(T-2) .*(lam1-lam2).^2)./... %p
        (gamma(T)*gamma(2)*gamma(T-1)*gamma(1))... %p
        .*f_N2(lam1,lam2,T,rho)...      
        .*(log2(f_N2(lam1,lam2,T,rho)) - log2(exp(1))*(lam1+lam2));
    C_L(T)=(cap - integral2(fun,0,500,@(x) x,500))/T;
end



figure(2)
plot(C_L)
ylim([0 1.5])
xlim([1 T_max])
xlabel('T peroids')
ylabel('bits/T')
title('Lower bound for M=1,N=2')
shg

function [output] = f_N2(L1,L2,T,rho)
    alpha1=L1*rho*T/(1+rho*T);
    alpha2=L2*rho*T/(1+rho*T);
    first=(1+rho*T)^-2 * gamma(T) ./ (gamma(T-2).*(alpha2-alpha1));
%     Matlab incomplete_gamma(x,a) is normalized by Gamma(a), undo this here
    second=exp(alpha2).*gamma(T-2).*gammainc(alpha2,T-2)./alpha2.^(T-2) - exp(alpha1).*gamma(T-2).*gammainc(alpha1,T-2)./alpha1.^(T-2);
    output=first.*second;

end

% function [output]= f_N(Lam, N, T)
%     alpha=Lam*T/(1+T);
%     L=length(Lam);
%     first=(1+T)^-(N) * gamma(T)/gamma(T-L);
%     const_1=exp(alpha(1)).*gammainc(alpha(1),T-L)./(alpha(1).^(T-L));
%     sum_val=0;
%     for i=1:L        
%         temp = 1./prod(alpha(i)-[alpha(1:i-1) alpha(i+1:end)]);
%         sum_val = sum_val + temp*(exp(alpha(i)).*gammainc(alpha(i),T-L)./(alpha(i).^(T-L)) - const_1);
%     end
%     output=first*sum_val;
% end
