function [output] = p_lambda(lambda,T,N)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
num1=exp(-(sum(lambda))) * prod(lambda)^abs(T-N);
num2=1;
for i=1:min(T,N)-1
    for j=i+1:min(T,N)
        num2=num2*(lambda(i)-lambda(j))^2;
    end
end

den=1;
for i=1:min(T,N)
   den=den*(gamma(T-i+1)*gamma(N-i+1)); 
end
output=num1*num2/den;
end

