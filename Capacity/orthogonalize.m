function [Q,R] = orthogonalize(A)
%ORTHOGONALIZE Summary of this function goes here
% Gram Schmidt orthogonalization. Transforms A such that all columns are
% independent of each other
% A=QR
%
[T,M]=size(A);
Phi=(randn(T,M)+1i*randn(T,M))/sqrt(2);
Q=zeros(T,M);R=zeros(M,M);
%A=QR
for j=1:M
    v=A(:,j);
    for i=1:j-1
        R(i,j)=(Q(:,i)'*A(:,j));%/(Q(:,i)'*Q(:,i));
        v=v-R(i,j)*Q(:,i);
    end
    R(j,j)=norm(v);
    Q(:,j)=v/R(j,j);
end
end

