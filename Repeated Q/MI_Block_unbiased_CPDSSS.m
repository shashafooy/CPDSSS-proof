sigma2=1;
sigma2_v=1;
BLOCKS=5;
RUNS=1000;

% iter=1e4;
N=12;L=3;NNL=N-N/L;
%Dynamic iterations to have at least 15 times the number of variables in a
%block (have at most N*BLOCKS variables in x). This preventss numerical
%instability/inaccuracy when at high block size with not enough samples
% iter=N*BLOCKS*15;
observations=N*BLOCKS*10;
observations=10000;

% MI_manual = zeros(1,RUNS);
% MI_equation = zeros(1,RUNS);
MI_cov = zeros(1,BLOCKS);
MI_cov_no_noise = zeros(1,BLOCKS);
MI_cov_MSD = zeros(RUNS,BLOCKS);
MI_cov_MSD_no_noise = zeros(RUNS,BLOCKS);
% all_EG = zeros(

%% Calculate the mutual information for the given number of blocks
run=0;
% for run=1:RUNS
for observations=[500 5000 50000]
    run=run+1;
for num_block=1:BLOCKS
    iter=observations*num_block;
    %increase number of samples based on block size to maintain similar
    %number of iterations for MI covariance calculation
    s=(normrnd(0,sqrt(sigma2),N/L,iter)+1i*normrnd(0,sqrt(sigma2),N/L,iter))/sqrt(2);
    v=(normrnd(0,sqrt(sigma2_v),NNL,iter)+1i*normrnd(0,sqrt(sigma2_v),NNL,iter))/sqrt(2);

    x_tx=zeros(N,iter);
    x_no_noise = zeros(N,iter);
    
    % ZC sequence
    z=exp(1j*pi*[0:N-1]'.*[0:N-1]'/N)/sqrt(N);
    Z=toeplitz(z,[z(1); z(end:-1:2)]);
    
    E=eye(N/L);E=upsample(E,L);
    
    for i=1:num_block:iter-num_block
        %% Generate matrices G Q every num_block samples of s,v
        
        h=(randn(N,1)+1i*randn(N,1)).*exp(-[0:N-1]'/3);
        %         if(REAL) h=real(h); end
        H=toeplitz(h,[h(1); h(end:-1:2)]);
        [F,Hf]=eig(H);
        
        % Precoder design        
        A=E'*H;
        R=A'*A+0.0001*eye(N);
        a=[1; zeros(N/L-1,1)];
        p=A'*a;
        g=R\p;
        G=toeplitz(g,[g(1); g(end:-1:2)]);
        GZ=G*Z;
        
        % AN subspace
        [V D]=eig(R);        
        Q=V(:,1:NNL);
        
        
        %% Compute a block of samples using the same channel measurement
        x_tx(:,i:i+num_block-1)=GZ*E*s(:,i:i+num_block-1)+Q*v(:,i:i+num_block-1);
        x_no_noise(:,i:i+num_block-1)=GZ*E*s(:,i:i+num_block-1);
%         x_tx(:,i)=GZ*E*s(:,i)+Q*v(:,i);
        
    end

    %% covariance equations
    %Reshape data from (N x iterations*block_size) to (N*block_size x
    %iterations)
%     sN=s(:,1:floor(iter/num_block)*num_block); %trim to used samples
    sN=reshape(s,N/L*num_block,[]); 
    xN=reshape(x_tx,N*num_block,[]);
    xN_no_noise=reshape(x_no_noise,N*num_block,[]);
    
    
    %get covariance of block matrices
%     cov_s=cov(sN');
%     cov_x=cov(xN');
%     cov_x_no_noise=cov(xN_no_noise');
%     cov_xs=cov([xN;sN]');
%     cov_xs_no_noise=cov([xN_no_noise;sN]');
%     
%     
%     %Compute entropy via (eq 85) in proof
%     H_s=1/2*log(det(cov_s))+length(cov_s)/2*(log(2*pi)+1);
%     H_x=1/2*log(det(cov_x))+length(cov_x)/2*(log(2*pi)+1);
%     H_xs=1/2*log(det(cov_xs))+length(cov_xs)/2*(log(2*pi)+1);
% 
% %     H_x_no_noise=1/2*log(det(cov_x_no_noise))+length(cov_x_no_noise)/2*(log(2*pi)+1);
% %     H_xs_no_noise=1/2*log(det(cov_xs_no_noise))+length(cov_xs_no_noise)/2*(log(2*pi)+1);
%     
%     MI_cov(num_block)=H_s+H_x-H_xs;
    MI_cov_MSD(run,num_block)=H_MSD(sN) + H_MSD(xN) - H_MSD([xN;sN]);
    MI_cov_MSD_no_noise(run,num_block)=H_MSD(sN) + H_MSD(xN_no_noise) - H_MSD([xN_no_noise;sN]);
%     MI_cov_no_noise(num_block)=H_s+H_x_no_noise-H_xs_no_noise;
    
    
end
end
MSD_mean=mean(MI_cov_MSD,1);
MSD_no_noise_mean=mean(MI_cov_MSD_no_noise,1);
% MI_cov_MSD_no_noise = MI_cov_MSD_no_noise/RUNS;
figure(1)
% plot([1:BLOCKS],MI_cov./[1:BLOCKS])
plot([1:BLOCKS],MSD_mean),hold on
plot([1:BLOCKS],MSD_no_noise_mean),hold off
% hold on
% plot([1:BLOCKS],MI_cov_no_noise./[1:BLOCKS]),hold off
xlabel('channel block size')
ylabel('MI (nats)')
title('Mutual Information vs fixed channel period')
legend('MSD with noise','MSD without additive noise')


%Same plot, but with log scaling
% figure(2)
% semilogy([1:BLOCKS],(MI_cov)./[1:BLOCKS])
% % hold on
% % semilogy([1:BLOCKS],(MI_cov_no_noise)./[1:BLOCKS]),hold off
% xlabel('channel block size')
% ylabel('MI (nats)')
% title('Mutual Information vs fixed channel period')
% % legend('Using AN','without AN')


% save MI_CPDSSS_values MI_cov MI_cov_no_noise
% MI_eq_avg = mean(MI_equation);
% MI_cov_avg = mean(MI_cov);