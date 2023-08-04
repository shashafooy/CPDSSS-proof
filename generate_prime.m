function [num] = generate_prime(n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

num = 1;
for i=1:2^n
    temp = 0;
    for j=1:i
        temp = temp + floor(cos(pi*(factorial(j-1)+1)/j)^2);
    end
    num=num + floor((n/temp)^(1/n));
end