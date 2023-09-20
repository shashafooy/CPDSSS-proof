clear all
N=4;M=4;
rho_db=6;
rho=10^(rho_db/10);

RUNS=1*10^3;
MAX_T = 15;
T_range = [M:20];
N_range = [N];
MI=zeros(1,RUNS);



%% Gram Schmidt orthogonalization experiment

% Phi=(randn(T,M)+1i*randn(T,M))/sqrt(2);
% Q=zeros(T,M);R=zeros(M,M);
% %A=QR
% for j=1:M
%     v=Phi(:,j);
%     for i=1:j-1
%         R(i,j)=Q(:,i)'*Phi(:,j);
%         v=v-R(i,j)*Q(:,i);
%     end
%     R(j,j)=norm(v);
%     Q(:,j)=v/R(j,j);
% end
MI_F_monteCarlo=zeros(T_range(end),1);
MI_G_monteCarlo = zeros(T_range(end),1);

for T=T_range
    % MI_F_monteCarlo=zeros(15,1);
    % MI_G_monteCarlo = zeros(15,1);
    for N=N_range
        MI_F=zeros(RUNS,1);
        MI_G=zeros(RUNS,1);
        %% Monte Carlo
        for run=[1:RUNS]
            %Generate sample of X and Phi
            H=(randn(M,N)+1i*randn(M,N))/sqrt(2);
            W=(randn(T,N)+1i*randn(T,N))/sqrt(2);
            % S=(randn(T,M)+1i*randn(T,M))/sqrt(2);            %T x M isotroppically distributed
            Phi=[eye(M) zeros(M,T-M)]'; %T time M
            mu=zeros(N*T,1);
            Sigma=kron(eye(N),eye(T)+rho*T/M*Phi*Phi');
            X=(mvnrnd(mu,Sigma)+1i*mvnrnd(mu,Sigma)).'/sqrt(2);
            X=reshape(X,T,N);



            V=eye(M);
            S=sqrt(T)*Phi;
            % [Phi,V,]=orthogonalize(S);
            % V=eye(M);
            B=V*V';
            % X=sqrt(rho/M)*S*H+W;

            %Various decompositions/constants
            K=min(T,N);
            lambda=eig(X*X');
            lambda=lambda(end:-1:T-K+1);
            beta=rho*(T/M)/(1+rho*T/M);
            alpha=max(lambda)*1.1; %alpha is 10% larger than the biggest eigenvalue


            % F_mn=generate_F_G('F',beta,lambda,K,T,M);
            G_mn=generate_F_G('G',beta,lambda,K,T,M);

            %% MI using F
            % gam_F = 1;
            % for i=1:M
            %     gam_F=gam_F*gamma(i);
            % end
            % for i=T+1-M:T
            %     gam_F=gam_F/gamma(i);
            % end
            % exp_term = exp(beta*trace(X'*Phi*Phi'*X)) / (-1)^(M*(M-1)/2);
            % MI_F(run) = log2(gam_F * exp_term / det(F_mn));
            % if(real(MI_F(run))==Inf) %kind of get rid of the Inf due to too large F_mn det()
            %     MI_F(run)=mean(MI_F(1:run-1));
            % end
            %% MI using G
            gam_G = 1;
            for i=1:T-M
                gam_G=gam_G * gamma(i);
            end
            % if(M==T)
            %     gam_G = gam_G / gamma(M+1);
            % else
            for i=M+1:T
                gam_G=gam_G / gamma(i);
            end
            % end
            exp_term = exp(beta*trace(X*X'*(Phi*Phi' - eye(T)))) / (-1)^((T-M)*(T-M-1)/2);
            MI_G(run) = real(log2(gam_G * exp_term / det(G_mn)));
            % if(MI_G(run)==Inf)
            %     MI_G(run)=mean(MI_G(1:run-1));
            % end

        end
        %remove inf objects
        MI_F(abs(MI_F)==Inf)=[];
        MI_G(abs(MI_G)==Inf)=[];
        MI_G = MI_G(~(isnan(MI_G) | isinf(MI_G))); %remove any inf or NaN values


        MI_F_monteCarlo(T)=mean(MI_F)/T; %scale MI my number of symbols
        MI_G_monteCarlo(T)=mean(MI_G)/T; %scale MI my number of symbols

    end
    % max_cap=log2(det(eye(N)+(rho/M)* M*eye(N)));
    figure(1)
    % plot([M:20],MI_F_monteCarlo(M:20)); hold on

    % plot(T_range,MI_G_monteCarlo(T_range)); hold on

end

iter=1e5;
E_cap=zeros(iter,1);
for i=1:iter
    H=(randn(M,N)+1i*randn(M,N))/sqrt(2);
    E_cap(i)=log2(det(eye(N)+rho/M * H'*H));
end
max_cap = real(squeeze(mean(E_cap,1)));

% legend('F matrix','G matrix')
% legend("N = " + string([1:5]))
% % yline(max_cap,'--','max capacity')
% title("Mutual Information M=" + num2str(M) + ", rho(dB) = " + num2str(rho_db))
% xlabel('N')
% ylabel('bits/T')
%
figure(2)
plot(T_range,MI_G_monteCarlo(T_range));
hold on, yline(max_cap,'--','max capacity'),hold off
title("Mutual Information M=N=" + num2str(M) + ", rho(dB) = " + num2str(rho_db))
xlabel('T')
ylabel('bits/T')








function [H_mn] = generate_F_G(matrix_type,beta,lambda,K,T,M)
%Generates the matricies F or G
%  matrix_type = 'F', 'f', 'G', 'g'

%F hankel matrix
if(matrix_type == 'F' | matrix_type == 'f')
    dim=M;
elseif(matrix_type == 'G' | matrix_type == 'g')
    dim=T-M;
    beta=-beta;
else
    error("Invalid Matrix Type, require argument 'F' or 'G'")
end

H_mn=zeros(dim,dim);

for m=1:dim
    for n=1:dim
        Q=T-K-m-n+2;
        for k=1:K
            gamma_H=1; %default Q<=0
            if(Q>=1)
                %no function for negative argument in gammainc
                gamma_H = 0;
                for q=0:Q-1
                    gamma_H = gamma_H + (beta*lambda(k))^q / factorial(q);
                end
                gamma_H = 1-exp(-beta*lambda(k))*gamma_H;
            end
            lambda_diff_noK=1;
            if(K>1)
                lambda_noK = [lambda(1:k-1); lambda(k+1:end)]; %remove lambda(k)
                lambda_diff_noK = beta*lambda(k) - beta*lambda_noK;
            end

            H_mn(m,n)=H_mn(m,n) + exp(beta*lambda(k)) * gamma_H / ( (beta*lambda(k))^Q * prod(lambda_diff_noK) );

        end
    end
end
end