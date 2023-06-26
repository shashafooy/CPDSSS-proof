load ITE_dir
addpath(genpath(ITE_code_dir));


N=1e6;
sigma2_s=1;
sigma2_v=100*sigma2_s;

% K=128;h_releigh=(randn(K,1)+1i*randn(K,1)).*exp(-[0:K-1]'/3);
% h_releigh=real(h_releigh);
% H=toeplitz(h_releigh,[h_releigh(1); h_releigh(end:-1:2)]);
s=normrnd(0,sqrt(sigma2_s),1,N)*10;
v=normrnd(0,sqrt(sigma2_v),1,N);
theta=2*pi*rand(1,N);
% Q=[cos(theta) -sin(theta);sin(theta) cos(theta)];
% Qr=reshape(permute(Q,[1 3 2]),N*2,2);
% Q=permute(reshape(Qr,2,N,2),[1 3 2]); %to go back to Q
% x1=Qr*[s;v];
% x1=permute(reshape(x1,2,N,2),[1 3 2]); %to go back to Q
x=[cos(theta); sin(theta)].*s+ [-sin(theta); cos(theta)].*v;

%% MI with toolbox
co_knn = IShannon_HShannon_initialization(1);
co_edge = IShannon_HShannon_initialization(1);
co_edge.member_name = 'Shannon_Edgeworth';
co_edge.member_co = H_initialization(co_edge.member_name,1);

MI_knn = IShannon_HShannon_estimation([x;s],[2;1],co_knn);
MI_edge = IShannon_HShannon_estimation([x;s],[2;1],co_edge);



%% Manual expectations
QAQ=zeros([2 2]);
s2=s.^2;
v2=v.^2;
for i=1:N
    Qi=[cos(theta(i)) -sin(theta(i));sin(theta(i)) cos(theta(i))];
    Lambda=[s2(i) 0;0 v2(i)];
    QAQ=QAQ+Qi*Lambda*Qi';    
end
QAQ=QAQ/N;
Qk=mean([cos(theta);sin(theta)],2);

%Mutual information using expectation equations CPDSSS Proof (83)-(87)
co_s=cov(s);
co_x=QAQ;
co_xs=[QAQ Qk*co_s;Qk'*co_s co_s];
Hs=1/2*log(det(co_s))+1/2*(log(2*pi)+1);
Hx=1/2*log(det(co_x))+2/2*(log(2*pi)+1);
Hxs=1/2*log(det(co_xs))+3/2*(log(2*pi)+1);
MI_manual=Hs+Hx-Hxs;
MI_equation = 1/2*(log(det(QAQ)) - log(det(QAQ - Qk*co_s*Qk')));


%% covariance equations
cov_s=cov(s')
cov_x=cov(x')
cov_xs=cov([x;s]')
% histogram(x(1,:))
H_s=1/2*log(det(cov_s))+1/2*(log(2*pi)+1)
H_x=1/2*log(det(cov_x))+2/2*(log(2*pi)+1)
H_xs=1/2*log(det(cov_xs))+3/2*(log(2*pi)+1)

MI_cov=H_s+H_x-H_xs