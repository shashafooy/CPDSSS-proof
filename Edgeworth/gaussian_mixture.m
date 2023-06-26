function [H] = gaussian_mixture(beta,d,Nc,samples)
%v(dim,samples)
% Nc=3;
gm = gmdistribution([0:beta:(Nc-1)*beta]',1);
x=random(gm,samples);


H=0;
for i=1:Nc
    H = H + 0.5*log(det(cov(v((i-1)*d+1:i*d,:)'))) + d/2*log(2*pi)+d/2;
end
H = H/Nc + log(Nc);

temp = 0;
for num=1:Nc
    v_k=v((num-1)*d+1:num*d,:);
    [t1 t2 t3] = my_Edgeworth_t1_t2_t3(v_k);
    
    temp = temp + t1 + t2 + t3;
end

H = H - 1/(12*Nc)*temp;

end