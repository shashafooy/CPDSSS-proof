co.k=20;
co.NSmethod='kdtree';

num_samples=1e4;
alpha=1e-4;
num_alpha=6;
alpha_range=logspace(-6, -1,num_alpha);
MI=zeros(num_alpha,1);
idx=1;

family = 3;


for alpha=alpha_range
    switch family
        case 1
            X=rand(1,num_samples);
            V=rand(1,num_samples);
            Y=X+alpha*V;
        case 2
            X=rand(1,num_samples);
            V=randn(1,num_samples);
            Y=X+alpha*V;
        case 3
            Sigma=[7 -5 -1 -3;...
                -5 5 -1 3;...
                -1 -1 3 -1;...
                -3 3 -1 2+alpha];
            temp=mvnrnd(zeros(1,4),Sigma,num_samples).';
            X=temp(1:2,:);Y=temp(3:4,:);
    end
    
    H_joint=HShannon_gkNN_estimation([X; Y],co);
    H_X=HShannon_gkNN_estimation(X,co);
    H_Y=HShannon_gkNN_estimation(Y,co);
    
    MI(idx)=H_X+H_Y-H_joint;
    idx=idx+1;
end

semilogx(alpha_range,MI);
shg




function MI = MI_gkNN(Y,co,ds)
    cum_ds=cumsum([1;ds(1:end-1)]);
    M=length(ds)-1;
    H_joint=HShannon_gkNN_estimation(Y,co);

    H_cross = 0;
    for m=1:M+1
        indm = [cum_ds(m):cum_ds(m)+ds(m)-1];
        H_cross = H_cross + HShannon_gkNN_estimation(Y(indm,:),co);
    end

    MI=H_cross - H_joint;


end
