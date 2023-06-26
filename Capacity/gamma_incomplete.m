function [gamma] = gamma_incomplete(T,z)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
fun=@(q) q.^(T-1).*exp(-q);
gamma=integral(fun,0,z);
end

