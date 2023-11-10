function [H] = gaussian_entropy(sigma)
%GAUSSIAN_ENTROPY Summary of this function goes here
%   Detailed explanation goes here
[N,~]=size(sigma);

H = N/2 + N/2*log(2*pi) + 1/2*log(det(sigma));


end

