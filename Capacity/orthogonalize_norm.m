function [Q,R] = orthogonalize_norm(A)
%ORTHOGONALIZE Summary of this function goes here
% Gram Schmidt orthogonalization. Transforms A such that all columns are
% independent of each other
% A=QR
% alt='alt' to run alternative numerically stable model

% if nargin < 2
%     use_alt = 0;
%     normalize = 0;
% else
%     use_alt=0;
% end


[T,M]=size(A);
Q=zeros(T,M);R=zeros(M,M);
%A=QR
%Q_i is the normalized orthogonal vector
%R_ij is the projection operator between i-th and j-th vector


for j=1:M
    v=A(:,j);
    for i=1:j-1
        %alternative, use updated v in projection
            % R(i,j)=(Q(:,i)'*v)/(Q(:,i)'*Q(:,i)); %Q_i is already normalized so <Q_i,Q_i>=1
        % use original vector in projection
            R(i,j)=(Q(:,i)'*A(:,j))/(Q(:,i)'*Q(:,i)); %Q_i is already normalized so <Q_i,Q_i>=1
 
        v=v-R(i,j)*Q(:,i);
    end
    R(j,j)=norm(v);
    Q(:,j)=v/R(j,j);
end
end