N= 1000;
runs = 999;
co = HShannon_Edgeworth_initialization(1);
d=1;
sigma = 0.1^2*eye(d);
sigma = 1;
b_range = logspace(-1,2);
Nc=[2 5 10];

% MI = zeros(length([0:0.01:0.5]),1);
H = zeros(length(b_range),3);

for i=1:3
    n=1;
    for beta = b_range
        %     mu=[beta/2 beta/2;-beta/2 beta/2;beta/2 -beta/2];
        %     gm = gmdistribution(mu,[1 1]);
        mu=[0:beta:(Nc(i)-1)*beta]';
        gm = gmdistribution(mu,1);
        for iter = 1:runs
            %         x1=mvnrnd([beta/2;beta/2],sigma,N)';
            %         x2=mvnrnd([-beta/2;beta/2],sigma,N)';
            %         x3=mvnrnd([beta/2;-beta/2],sigma,N)';
            xi=mvnrnd(mu,sigma*eye(length(mu)),N);
            x = random(gm,N)';
            %         plot(x1(1,:),x1(2,:),'r*');
            %         hold on
            %         plot(x2(1,:),x2(2,:),'b*');
            %         plot(x3(1,:),x3(2,:),'g*');
            %         hold off
            H_max = 0;
            for j=1:Nc(i)
                H_max = H_max + 0.5*log(det(cov(xi))) + d/2*log(2*pi)+d/2;
            end
            H_max = H_max/Nc(i) + log(Nc(i));
            
            H(n,i) = H(n,i) + min(HShannon_Edgeworth_estimation(x,co),H_max);
            
            %         MI(n) = MI(n) - HShannon_Edgeworth_estimation([x1;x2;x3],co); %joint
            %         MI(n) = MI(n) + HShannon_Edgeworth_estimation(x1,co);
            %         MI(n) = MI(n) + HShannon_Edgeworth_estimation(x2,co);
            %         MI(n) = MI(n) + HShannon_Edgeworth_estimation(x3,co);
        end
        %     MI(n)=MI(n)/runs;
        H(n,i)=H(n,i)/runs;
        n=n+1;
        
    end
end
% figure(2),plot([0:0.01:0.5],MI);
figure(3),semilogx(b_range,H),shg;
ylim([0 5]);
legend("Nc: " + Nc)






