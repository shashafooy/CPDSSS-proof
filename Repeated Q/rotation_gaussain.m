N=1e6;
% num_block=2;
s=normrnd(0,1,1,N);
v=normrnd(0,1,1,N);
theta=2*pi*rand(1,N);

% Histograms to show gaussian combination is gaussian
% x=[cos(theta); sin(theta)].*s+ [-sin(theta); cos(theta)].*v;
% y=[cos(theta); sin(theta)].*s;
% figure(1),histogram(x(1,:)),title('Linear comb of gaussian is gaussian');
% figure(2),histogram(y(1,:)),title('Single gaussian with another distribution');

for num_block = [1:100]
    x=zeros(2*num_block,floor(N/num_block));
    cnt=1;
    for i=[1:num_block:N-num_block]
        %create diagonal matrix with replicas of Q1
        Q1=[cos(theta(i)) -sin(theta(i));sin(theta(i)) cos(theta(i))];
        Q_cell = repmat({Q1},1,num_block);
        Q=blkdiag(Q_cell{:});
        
        %interleave s and v for a block of the channel
        symbols = [s(i:i+num_block-1); v(i:i+num_block-1)];
        symbols = symbols(:);
        x(:,cnt)= Q*symbols;
        cnt=cnt+1;
        
        
    end
    sN=s(1:length(x(1,:))*num_block);
    sN=reshape(sN,num_block,[]);
    cov_s=cov(sN');
    cov_x=cov(x');
    cov_xs=cov([x;sN]');
    H_s=1/2*log(det(cov_s))+1/2*(log(2*pi)+1);
    H_x=1/2*log(det(cov_x))+2*num_block/2*(log(2*pi)+1);
    H_xs=1/2*log(det(cov_xs))+(2*num_block + 1)/2*(log(2*pi)+1);
    
    MI_cov(num_block)=H_s+H_x-H_xs;
end
figure(1)
plot([1:100],MI_cov)
xlabel('channel block size')
ylabel('MI')
title('Mutual Information vs fixed channel period')



figure(2)
plot([1:100],20*log10(MI_cov))
xlabel('channel block size')
ylabel('MI (dB)')
title('Mutual Information (dB) vs fixed channel period')


% save rotation MI_cov N s v theta