function [output] = p_Phi(T,Phi)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
output=1;
cnt=T;
for i=1:T
    output=output*gamma(cnt)/pi^cnt * dirac(Phi(:,i)'*Phi(:,i)-1);
    for j=1:i-1
        output=output*dirac(Phi(:,j)'*Phi(:,i));
    end
    cnt=cnt-1;
end

end

