function [H] = HShannon_gkNN_estimation(Y,co)
%function [H] = HShannon_kNN_k_estimation(Y,co)
%Estimates the Shannon differential entropy (H) of Y using the geometric kNN method (and neighbors S={k}).
% This method uses elipses instead of balls for the local region. This
% metric should perform better for more localized data resulting in a more
% accurate estimation while also needing less samples
%
%Create ellipse centered on the i-th sample where the length of the major axis of the
%ellipse is the distance to the k-th neighbor.
%INPUT:
%   Y: Y(:,t) is the t^th sample.
%  co: entropy estimator object.
%
%REFERENCE:
%   Warren M. Lord, Jie Sun, Erik M. Bollt; Geometric k-nearest neighbor estimation of entropy and mutual information. Chaos 1 March 2018; 28 (3): 033114. https://doi.org/10.1063/1.5011683


[d,num_of_samples] = size(Y);
% squared_distances = kNN_squared_distances(Y,Y,co,1);
%%%%%%%%%%%%%%%%%%%%%
%%% DELETE LATER %%%%
%%%%%%%%%%%%%%%%%%%%%
% Y=Y(1:2:end,:);
%%%%%%%%%%%%%%%%%%%%%

[indices,distances] = knnsearch(Y.',Y.','K',co.k+1,'NSMethod',co.NSmethod); %[double,...
indices = int32(indices(:,2:end).');%.': to be compatible with 'ANN'
            % squared_distances = (distances(:,2:end).').^2;%distances -> squared distances; .': to be compatible with 'ANN'
% d=d/2;

%co.mult:OK. The information theoretical quantity of interest can be (and is!) estimated exactly [co.mult=1]; the computational complexity of the estimation is essentially the same as that of the 'up to multiplicative constant' case [co.mult=0]. In other words, the estimation is carried out 'exactly' (instead of up to 'proportionality').

%H estimation:
    % V = volume_of_the_unit_ball(d);
    [sigmas]=ellipse_sigma_axis(Y.',indices,distances,d);
    H = log(num_of_samples-1) - psi(co.k) + log(V) + d / num_of_samples * sum(log(distances(co.k,:))); %sqrt <= squared_distances,
 
end

function [sigma] = ellipse_sigma_axis(samples,indicies,distances,dim)
    [k,num_samples]=size(indicies);
    k=k+1;
    sigma=zeros(dim,k)
    for i=1:num_samples
        x_ij=samples([i;indicies(:,i)],:); %(k+1)xd k nearest samples to the i-th sample. d is the dimension of x_i
        z=mean(x_ij,1); %centroid
        z=x_ij(1,:);
        y=x_ij-z; %Center data

        [U,Sigma,V]=svd(y);

        sigma(i,:)=diag(Sigma);
    end

end

function [k_i] = points_in_ellipse(samples,sigma,v)
    cen=samples(1,:);
    [K,~]=size(samples);
    k_i=1;
    eigs=1;
    for i=2:K
        point=(samples(i,:)-cen)';
        val=point'*A*point
    end
end