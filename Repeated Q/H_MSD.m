function [H] = H_MSD(X)
%Computes the Entropy for a given variable X where the rows are indivitual
%variables and the columns are samples of that variable.
%Computed using Misra, Singh, and Demchuk technique
N=size(X,2); %number of samples
d=size(X,1); %number of dimensions

S=(N-1)*cov(X.'); %covariance normalized by (N-1), undo this here
H=0.5*d*(1+log(pi)) + 0.5*logdet(S,'chol');

for i=1:d
   H = H - 0.5*psi(0.5*(N-i));    
end
end

