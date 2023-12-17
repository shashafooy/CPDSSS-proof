num_samples=1e3;
k=30;
% co.NSmethod='kdtree';

N=2;L=1;T=10;
M=N/L;P=N-N/L;


sigma_s=1;sigma_v=1;

g1_vector = zeros(N,num_samples);
x_vectors = zeros(N,T,num_samples);
G_vector = zeros(N,M,num_samples);

%%%%%%%%%%%%%%%%%%%%
% Generate samples %
%%%%%%%%%%%%%%%%%%%%

for iter=1:num_samples
    [G,Q]=generate_G_Q(N,L);

    g1_vector(:,iter)=G(:,1);
    G_vector(:,:,iter)=G;

    % s=sqrt(sigma_s)*(randn(M,T)+1i*randn(M,T))/sqrt(2);
    % v=sqrt(sigma_v)*(randn(P,T)+1i*randn(P,T))/sqrt(2);
    s=sqrt(sigma_s)*randn(M,T);
    v=sqrt(sigma_v)*randn(P,T);
    X=G*s+Q*v;

    x_vectors(:,:,iter)=X;
end

%%%%%%%%%%%%%%%%%%%%%%
% Mutual Information %
%%%%%%%%%%%%%%%%%%%%%%


max_k_scale=1;
MI=zeros(T,max_k_scale);

%Initial values from first transmission
xT_term=C2R_vector(squeeze(x_vectors(:,1,:)));
g_term=C2R_vector(g1_vector);

%compute "previous" iteration's conditional and joint term
H_xxc=HShannon_gkNN_estimation(xT_term,k);
H_joint=HShannon_gkNN_estimation([xT_term;g_term],k);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TOTAL X ENTROPY TEST%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MI(1,1)=0;



for k_scale = [1]
    for t=2:T
        % MI(X,Y|Z)
        k=100;
        % .k=40;

        xT_term=C2R_vector(squeeze(x_vectors(:,t,:)));
        xcond_term=C2R_vector(reshape(x_vectors(:,1:t-1,:),N*(t-1),num_samples));
        % xT_term=squeeze(x_vectors(:,t,:));
        % g_term=g1_vector;
        % xcond_term=reshape(x_vectors(:,1:t-1,:),N*(t-1),num_samples);

        %Reuse previous iteration calculations
        H_gxc = H_joint;
        H_cond = H_xxc;

        % Cross terms betwen X,Z and Y,Z
        H_xxc = HShannon_gkNN_estimation([xT_term;xcond_term],k);
        H_gknn(t)=H_xxc;
        % co=HShannon_kNN_k_initialization(1);
        % H_knn(t)=HShannon_kNN_k_estimation([xT_term;xcond_term],co);

        % H_gxc = HShannon_gkNN_estimation([g_term;xcond_term],k);

        % Joint entropy
        H_joint = HShannon_gkNN_estimation([xT_term;g_term;xcond_term],k);

        % conditional entropy value
        % H_cond = HShannon_gkNN_estimation(xcond_term,k);

        MI(t,k_scale)=H_gxc + H_xxc - H_joint - H_cond;
        % MI(t,k_scale)=H_xxc;

    end
end
figure(1)
plot([1:10]*2,H_gknn,'-r*'),hold on
plot([1:10]*2,H_knn,'-b*'),hold off
legend("geometric knn","knn")

figure(1)
plot(1:T,MI)
title("Individual conditional MI")
xlabel("T")
ylabel("I(g,x_T | x_1 ...)"),shg


figure(2)
cum_MI = cumsum(MI);
plot(1:T,cum_MI)
yline(gaussian_entropy(diag(ones(N,1))),'-','H(G)')
title("Cumulative MI")
xlabel("T")
ylabel("I(g,x_T)"),shg











function [X_R] = C2R_vector(X_C)
if(isreal(X_C))
    X_R=X_C;
    return
end
[N,sam]=size(X_C);
X_R=zeros(2*N,sam);
X_R(1:2:end)=real(X_C);
X_R(2:2:end)=imag(X_C);
end



function [G,Q] = generate_G_Q(N,L)
epsilon=1e-4;

h=(randn(N,1)+1i*randn(N,1)).*exp(-[0:N-1]'/3);
h=randn(N,1).*exp(-[0:N-1]'/3); % REAL EXPERIMENT
H=toeplitz(h,[h(1); h(end:-1:2)]);
E=eye(N/L);E=upsample(E,L);

A=E'*H;
R=A'*A+epsilon*eye(N);
a=[1; zeros(N/L-1,1)];
p=A'*a;
g=R\p;

G=toeplitz(g,[g(1); g(end:-1:2)]);
G=G*E;

[V D]=eig(R);
NNL=N-N/L;
P=NNL;
Q=V(:,1:NNL);
% Q=Q';       %take hermitian to match X=SG + VQ
% v=(randn(T,P)+1i*randn(T,P))/sqrt(2);


M=N/L;P=N-M;
G=randn(N,M);
Q=randn(N,P);
end


