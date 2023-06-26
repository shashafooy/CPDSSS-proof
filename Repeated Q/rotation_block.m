% load ITE_dir
% addpath(genpath(ITE_code_dir));
REAL = false;
sigma2_s=1;
sigma2_v=20*sigma2_s;


iter=1e4;
N=8;L=4;NNL=N-N/L;
RUNS=1000;
% Q_hold = floor(iter/10);
Q_hold = iter/2;
MI_manual = zeros(1,RUNS);
MI_equation = zeros(1,RUNS);
MI_cov = zeros(1,RUNS);
% all_EG = zeros(


for run=1:RUNS
    
    
    
    
    
    % Symbols and noise
    %     if(REAL)
    %         s=normrnd(0,sqrt(sigma2_s),N/L,iter);
    %         v=normrnd(0,sqrt(sigma2_v),NNL,iter);
    %
    %     else
    s=(normrnd(0,sqrt(sigma2_s),N/L,iter)+1i*normrnd(0,sqrt(sigma2_s),N/L,iter))/sqrt(2);
    v=(normrnd(0,sqrt(sigma2_v),NNL,iter)+1i*normrnd(0,sqrt(sigma2_v),NNL,iter))/sqrt(2);
    %     end
    x_tx=zeros(N,iter);
    E_G = zeros(N,N/L);
    E_g = zeros(N,1);
    E_R = zeros(N,N);
    E_Rinv = zeros(N,N);
    E_p = zeros(N,1);
    E_QAQ = zeros(N,N);
    E_lambda = zeros(N,N);
    Q_generated = 0;
    % ZC sequence
    z=exp(1j*pi*[0:N-1]'.*[0:N-1]'/N)/sqrt(N);
    Z=toeplitz(z,[z(1); z(end:-1:2)]);
    
    E=eye(N/L);E=upsample(E,L);
    
    
    
    for i=1:iter
        % Generate matrices G Q
        if(mod(i-1,Q_hold)==0) %only generate Q matrix for every Q_hold
            Q_generated = Q_generated + 1;
            h=(randn(N,1)+1i*randn(N,1)).*exp(-[0:N-1]'/3);
            %         if(REAL) h=real(h); end
            H=toeplitz(h,[h(1); h(end:-1:2)]);
            [F,Hf]=eig(H);
            
            % Precoder design
            
            A=E'*H;
            R=A'*A+0.0001*eye(N);
            a=[1; zeros(N/L-1,1)];
            p=A'*a;
            Rinv=R^-1;
            g=Rinv*p;
%             g=R\p;
            G=toeplitz(g,[g(1); g(end:-1:2)]);
            GZ=G*Z;
            
            E_R=E_R+R;
            E_Rinv=E_Rinv + Rinv;
            E_g=E_g+g;
            E_p=E_p+p;
            E_G = E_G + GZ*E;
            
            
            % AN subspace
            [V D]=eig(R);
            
            Q=V(:,1:NNL);
            
            
        end
        
        
        x_tx(:,i)=GZ*E*s(:,i)+Q*v(:,i);
        % sigma_g=alpha*sigma_g + (1-alpha)*mean(G*E);
        
        %         lambda = [s(:,i)*s(:,i)' zeros(N/L,NNL);zeros(NNL,N/L) v(:,i)*v(:,i)']; %assume s and v are independent
        lambda = diag([s(:,i).*conj(s(:,i)); v(:,i).*conj(v(:,i))]);
        %         lambda=[s(:,end);v(:,end)]*[s(:,end)' v(:,end)'];
        E_QAQ=E_QAQ + [GZ*E Q]*lambda*[GZ*E Q]';
        E_lambda = E_lambda + lambda;
        
        
    end
    E_G = E_G/Q_generated;
    E_g=E_g/Q_generated;
    E_R = E_R/Q_generated;
    E_Rinv = E_Rinv/Q_generated;
    E_p = E_p/Q_generated;
    E_QAQ = E_QAQ/Q_generated;
    E_lambda = E_lambda/iter;
    
    if(mod(run,10000)==1)
        save CPDSSS_samples E_G E_g E_R E_Rinv E_p E_QAQ s v x_tx
    end
    %% MI with toolbox
    % co_knn = IShannon_HShannon_initialization(1);
    % co_edge = IShannon_HShannon_initialization(1);
    % co_edge.member_name = 'Shannon_Edgeworth';
    % co_edge.member_co = H_initialization(co_edge.member_name,1);
    %
    % MI_knn = IShannon_HShannon_estimation([x_tx;s],[N;N/L],co_knn);
    % MI_edge = IShannon_HShannon_estimation([x_tx;s],[N;N/L],co_edge);
    
    
    
    %% Manual expectations
    %Mutual information using expectation equations CPDSSS Proof (83)-(87)
    cov_s=cov(s');
    cov_x=cov(x_tx');
    cov_xs=cov([x_tx;s]');
    
    %expectations
    co_s=cov_s;
    co_x=E_QAQ;
    co_xs=[E_QAQ E_G*co_s;(E_G*co_s)' co_s];
    Hs=1/2*log(det(co_s))+1/2*(log(2*pi)+1);
    Hx=1/2*log(det(co_x))+2/2*(log(2*pi)+1);
    Hxs=1/2*log(det(co_xs))+3/2*(log(2*pi)+1);
    MI_manual(run)=Hs+Hx-Hxs;
    MI_equation(run) = 1/2*(log(det(E_QAQ)) - log(det(E_QAQ - E_G*co_s*E_G')));
    
    
    
    
    %% covariance equations
    
    % histogram(x(1,:))
    H_s=1/2*log(det(cov_s))+(N/L)/2*(log(2*pi)+1);
    H_x=1/2*log(det(cov_x))+N/2*(log(2*pi)+1);
    H_xs=1/2*log(det(cov_xs))+(N+N/L)/2*(log(2*pi)+1);
    
    MI_cov(run)=H_s+H_x-H_xs;
    
    
end

figure(2), histogram(real(MI_equation(1:run)),200), hold on, 
% histogram(imag(MI_equation(1:run)),100),legend('real','imag'),
title("MI using equation (93) in proof, N="+run),hold off
figure(1), histogram(real(MI_cov(1:run)),200), hold on, 
% histogram(imag(MI_cov(1:run)),100),legend('real','imag'),
title("MI using matlab covariance, N="+run),hold off

MI_eq_avg = mean(MI_equation);
MI_cov_avg = mean(MI_cov);