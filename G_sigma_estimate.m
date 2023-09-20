 N=6;L=3;
M=N/L; P=N-N/L;

 T=8;
 iterations=1e5;
 epsilon = 1e-4;

noise_power = 0.5 + rand(P,1); %power between 0.5-1.5


 sigG=zeros(iterations,N,N);
 % sig_g=zeros(iterations,N,N,M,M); %g*g' is MxM, total of NxN different columns g
 % sig_noise=zeros(iterations,N,N,T,T);
 sig_qq=zeros(iterations,N,N);
 E_V=zeros(iterations,N,N);
 E_V_rot=zeros(iterations,N,N);
 E_Lambda = zeros(iterations,N,N);
 E_AN = zeros(iterations,N,N);
 E_noise = zeros(iterations,P,P);
 E_Q = zeros(iterations,N,N);
 
 
 % g1_g1=zeros(iterations,M,M);
 % g1_g2=zeros(iterations,M,M);
 % g1_g3=zeros(iterations,M,M);
 % g2_g1=zeros(iterations,M,M);
 % g2_g2=zeros(iterations,M,M);
 % g3_g3=zeros(iterations,M,M);
 % q1_q1=zeros(iterations,T,T);
 % q1_q2=zeros(iterations,T,T);
 % q2_q1=zeros(iterations,T,T);
 % q2_q2=zeros(iterations,T,T);
 sigH=zeros(iterations,N,N);

orth = eye(N);
% orth(:,randperm(N)) = orth(:,:); %random rotation matrix using identity matrix
 % sig_noise=zeros(iterations,T,T);

 X=zeros(N,iterations);
for iter=1:iterations
    % h=randn(N,1);
    % h=(randn(N,1)+1i*randn(N,1))/sqrt(2);
    h=(randn(N,1)+1i*randn(N,1)).*exp(-[0:N-1]'/3);
    H=toeplitz(h,[h(1); h(end:-1:2)]);
    % sigH(iter,:,:)=inv(H*H'+epsilon*eye(N));
    % [F,Hf]=eig(H);
    % %%
    % % Precoder design
    E=eye(N/L);E=upsample(E,L);

    A=E'*H;
    R=A'*A+epsilon*eye(N);
    a=[1; zeros(N/L-1,1)];
    p=A'*a;
    g=R\p; %g=g/sqrt(g'*g);
    % % sig_p(i,:,:)=p*p';
    G=toeplitz(g,[g(1); g(end:-1:2)]);
    G=G*E;
    G=G;
    % [V, Lambda] = eig(G'*G);
    % E_V(iter,:,:)=V;
    % E_V_rot(iter,:,:)=orth*V;
    % E_Lambda(iter,:,:)=Lambda;
    % sigG(iter,:,:)=G*G';
    % X(:,iter)=G*ones(M,1); %sample of G and symbols

    % mean_G(i,:,:)=G;
    % form s*G + v*Q, so take hermitian of G
    % G=G';
    % %%% Test with orthogonal G %%%
    % H=(randn(N,N) + 1i*randn(N,N))/sqrt(2);
    % H_orth=orthogonalize(H);
    % G=H_orth(:,1:M);
    % Q=H_orth(:,M-+1:end);
    % 
    % for i=1:N
    %     for j=1:N
    %         % sig_g(iter,i,j,:,:)=G(:,i)*G(:,j)';
    %         sig_g(iter,i,j,:,:)=G_orth(:,i)*G_orth(:,j)';
    %     end
    % end

    % 
    % g1_g1(iter,:,:)=G(1,:)' * G(1,:); %g1 * g1'
    % g1_g2(iter,:,:)=G(1,:)' * G(2,:); %g1 * g2'
    % g1_g3(iter,:,:)=G(1,:)' * G(3,:);
    % g2_g1(iter,:,:)=G(2,:)' * G(1,:); %g2 * g1'
    % g2_g2(iter,:,:)=G(2,:)' * G(2,:); %g2 * g1'
    % g3_g3(iter,:,:)=G(3,:)' * G(3,:);

% 
[V D]=eig(R);
    NNL=N-N/L;
    P=NNL;
    Q=V(:,1:NNL);
    Q=Q;       %take hermitian to match X=SG + VQ
      v=(randn(T,P)+1i*randn(T,P))/sqrt(2);

    
    % Sigma_v=diag([1:P]);
    % sig_qq(iter,:,:)=Q'*Sigma_v*Q;
    

    % for i=1:P
    %     for k=1:N
    %         for l=1:N
    %         sig_qq(iter,:,k,l)=Q(:,k).*conj(Q(:,l));
    %         end
    %     end
    % end

  % q1_q1(iter,:,:)=v*Q(:,1)*Q(:,1)'*v';
  % q1_q2(iter,:,:)=v*Q(:,1)*Q(:,2)'*v';
  % q2_q1(iter,:,:)=v*Q(:,2)*Q(:,1)'*v';
  % q2_q2(iter,:,:)=v*Q(:,2)*Q(:,2)'*v';

  noise = (normrnd(0,sqrt(noise_power)) + 1i*normrnd(0,sqrt(noise_power)))/sqrt(2);
  E_AN(iter,:,:) = Q*noise*noise'*Q';
  E_noise(iter,:,:)=noise*noise';
  E_Q(iter,:,:)=Q*Q';
  
end
%remove nan samples
sig_g_no_nan = sig_g(~any(isnan(sig_g),[2 3 4 5]),:,:,:,:);
sig_qq_no_nan = sig_qq(~any(isnan(sig_qq),[2 3 4]),:,:);

sig_qq_mean=squeeze(mean(sig_qq_no_nan,1));
sig_qq_threshold = sig_qq_mean;
sig_qq_threshold(abs(sig_qq_threshold)<0.01)=0;
abs(sig_qq_threshold)


sig_g_mean=squeeze(mean(sig_g,1));
sig_g_thresh = sig_g_mean;
sig_g_thresh(abs(sig_g_mean)<0.001)=0;

sig_noise_mean=squeeze(mean(sig_noise,1));
sig_noise_thresh = sig_noise_mean;
sig_noise_thresh(abs(sig_noise_mean)<0.01)=0;

full_g=zeros(M*N,M*N);
full_noise=zeros(N*T,N*T);
for i=1:N
    for j=1:N
        full_noise([(i-1)*T+1:(i-1)*T + T],[(j-1)*T+1:(j-1)*T+T])=real(squeeze(sig_noise_thresh(i,j,:,:)));
        full_g([(i-1)*M+1:(i-1)*M+M],[(j-1)*M+1:(j-1)*M+M])=real(squeeze(sig_g_thresh(i,j,:,:)));
    end
end










% sig_g11=squeeze(mean(g1_g1,1));
% % sig_g11(abs(sig_g11)<0.001)=0;
% sig_g12=squeeze(mean(g1_g2,1));
% % sig_g12(abs(sig_g12)<0.001)=0;
% sig_g13=squeeze(mean(g1_g3,1));
% % sig_g13(abs(sig_g13)<0.001)=0;
% 
% sig_g21 = squeeze(mean(g2_g1,1));
% % sig_g21(abs(sig_g21)<0.001)=0;
% sig_g22 = squeeze(mean(g2_g2,1));
% % sig_g22(abs(sig_g22)<0.001)=0;
% 
% sig_g33 = squeeze(mean(g3_g3,1));
% % sig_g33(abs(sig_g33)<0.001)=0;
% 
% 
% sig_q11=squeeze(mean(q1_q1,1));
% sig_q12=squeeze(mean(q1_q2,1));
% sig_q21=squeeze(mean(q2_q1,1));
% sig_q22=squeeze(mean(q2_q2,1));



sigG=squeeze(mean(sigG,1));
imagesc(abs(sigG));
sigH=squeeze(mean(sigH,1));
imagesc(abs(sigH));

sig_noise=squeeze(mean(sig_noise,1));
imagesc(abs(sig_noise))
