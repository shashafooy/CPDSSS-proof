if ispc
    load ITE_dir
elseif isunix
    load ITE_dir_linux
end
addpath(genpath(ITE_code_dir));
% REAL = false;
% sigma2_s=1;
% sigma2_v=20*sigma2_s;
%
%
% iter=1e4;
% N=8;L=4;NNL=N-N/L;
% RUNS=1000;
% % Q_hold = floor(iter/10);
% Q_hold = iter/2;
% MI_manual = zeros(1,RUNS);
% MI_equation = zeros(1,RUNS);
% MI_cov = zeros(1,RUNS);
% % all_EG = zeros(

N=1;M=N;
max_T=5;
% T_range=M+1:max_T;
rho_db=6;
rho=10^(rho_db/10);

RUNS=5*10^4;

MI_estimation = zeros(max_T,1);
MI_estimation_nocond = zeros(max_T,1);
MI_estimation_multiDim = zeros(max_T,1);

for T=M:max_T
    % T=M+1;
    
    %% Generate data
    
    S=zeros(T,M,RUNS);
    X=zeros(T,N,RUNS);
    
    for run=1:RUNS
        
        
        
        H=(randn(M,N)+1i*randn(M,N))/sqrt(2);
        W=(randn(T,N)+1i*randn(T,N))/sqrt(2);
        %     S=(randn(T,M)+1i*randn(T,M))/sqrt(2);            %T x M isotroppically distributed
        %     Phi=[eye(M) zeros(M,T-M)]'; %T time M
        n_rand=(randn(T,T)+1i*randn(T,T))/sqrt(2);
        [Phi,R]=qr(n_rand); % isotropic unitary
        Phi=Phi(:,1:M); %truncate to T x M

        S(:,:,run)=sqrt(T)*Phi;
        X(:,:,run)=sqrt(rho/M)*sqrt(T)*Phi*H+W;
        
        %     mu=zeros(N*T,1);
        %     Sigma=kron(eye(N),eye(T)+rho*T/M*Phi*Phi');
        %     X=(mvnrnd(mu,Sigma)+1i*mvnrnd(mu,Sigma)).'/sqrt(2);
        %     X=reshape(X,T,N);
    end
    
    %% MI with toolbox
    % co_knn = IShannon_HShannon_initialization(1);
    % co_edge = IShannon_HShannon_initialization(1);
    % co_edge.member_name = 'Shannon_Edgeworth';
    % co_edge.member_co = H_initialization(co_edge.member_name,1);
    
    % MI_knn = IShannon_HShannon_estimation([x_tx;s],[N;N/L],co_knn);
    % MI_edge = IShannon_HShannon_estimation([x_tx;s],[N;N/L],co_edge);
    
    
    %%%%%%%%%%%%%%%%%%%%
    % conditional MI
    %%%%%%%%%%%%%%%%%%%%
    
    % s=squeeze(S(2,:,:));
    temp=squeeze(S);
    s=temp(2,:);
%     s=[s zeros(1,(T-2)*RUNS)]; %Zero pad s to match given_s
    s=repmat(s,1,T-1); %repeat s to match given_s
    % given_s=squeeze(S(1:T-1,:,:));
    given_s=temp(1:T-1,:);
    given_s=given_s(:).';
    % x=squeeze(X(2,:,:));
    x=squeeze(X);x=x(2,:);
%     x=[x zeros(1,(T-2)*RUNS)];
    x=repmat(x,1,T-1); 
    Y=[x;s;given_s];
    ds=[N;M;M];
    
    % co = condIShannon_HShannon_initialization(1);
%     MI_estimation(T)=condIShannon_HShannon_estimation(Y,ds,co)/T;
    
    %%%%%%%%%%%%%%%%%%%
    % vectorized MI
    %%%%%%%%%%%%%%%%%%%
    %Vectorize X,S
    x=squeeze(X); 
    x=x(:).';
    s=squeeze(S);
    s=s(:).';
    
    Y=[x;s];ds=[N;M];

    %Need to treat Y as real
    Y_expanded=C2R_vector(Y);
    % Y_expanded=zeros(sum(ds)*2,RUNS);
    % Y_expanded(1:2:end,:)=real(Y);
    % Y_expanded(2:2:end,:)=imag(Y);
    ds=2*ds;
    
    co_edge = IShannon_HShannon_initialization(1);
%     co_edge.member_name = 'Shannon_Edgeworth';
%     co_edge.member_co = H_initialization(co_edge.member_name,1);
    MI_estimation_nocond(T)=IShannon_HShannon_estimation(Y_expanded,ds,co_edge)/T;


    %%%%%%%%%%%%%%%%%%%%%
    % Multidimensional MI
    %%%%%%%%%%%%%%%%%%%%%
    Y=[squeeze(X);squeeze(S)];ds=[N*T;M*T];
    Y_expanded=C2R_vector(Y);ds=2*ds;
    % Y_expanded=zeros(sum(ds),RUNS*2);
    % Y_expanded(:,1:2:end)=real(Y);
    % Y_expanded(:,2:2:end)=imag(Y);

    MI_estimation_multiDim(T)=IShannon_HShannon_estimation(Y_expanded,ds,co_edge)/T;
end



%% Plots

figure(3), plot(1:max_T,MI_estimation),title("conditional dist I(x_T;s_T|s_{T-1} ...)");
figure(4), plot(1:max_T,MI_estimation_nocond), title("vectorized dist I(vec(X);vec(S))");

