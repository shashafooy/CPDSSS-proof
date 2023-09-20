iterations=1e3;
N=4;L=2;
M=N/L;P=N-N/L;
T=8;
epsilon = 1e-4;

all_V=zeros(iterations,N,N);
all_lam = zeros(iterations,N);
thetas = zeros(iterations*(iterations-1)/2,N);

for iter=1:iterations
    % H=(randn(N,N) + 1i*randn(N,N));
    % [V,Lambda]=eig(H'*H);
    h=(randn(N,1)+1i*randn(N,1)).*exp(-[0:N-1]'/3);
    H=toeplitz(h,[h(1); h(end:-1:2)]);
    E=eye(N/L);E=upsample(E,L);

    A=E'*H;
    R=A'*A+epsilon*eye(N);
    a=[1; zeros(N/L-1,1)];
    p=A'*a;
    g=R\p; 

    G=toeplitz(g,[g(1); g(end:-1:2)]);
    G=G*E;
    G=G';
    [V, Lambda] = eig(G'*G);
    all_V(iter,:,:)=V;
    all_lam(iter,:)=diag(Lambda);
end
for i=1:ceil(N/6):N %at most 6 figures
    %get dot product between every vector
    % currently a full matrix, but don't want duplicates
    % upper triangular elements are all we want
    % put all upper triangular elements into a vector

    % full_dot_matrix=real(acos(real(all_V(:,:,i)*all_V(:,:,i)')));  %for
    % complex V, only find angle of real part?

    %to take into account complex vectors, split N complex values into 2*N
    %real and image vector
    expanded_V = zeros(iterations,2*N,N);
    rot_scale=exp(1i*0.6); %some rotation so final V element isn't real
    rot_scale=1;
    expanded_V(:,1:N,:)=real(rot_scale*all_V);expanded_V(:,N+1:2*N,:)=imag(rot_scale*all_V);
    full_dot_matrix=acos(real(expanded_V(:,:,i)*expanded_V(:,:,i)')); 
    % full_dot_matrix = angle(all_V(:,:,i)*all_V(:,:,i)');
    tri_idx=triu(true(size(full_dot_matrix)),1); %get indices for upper triangular, do not include diagonal-
    thetas(:,i)=full_dot_matrix(tri_idx); %index only upper triangular, place into vector
    figure(i)
    histogram(thetas(:,i),'Normalization','pdf');
    xlabel("\theta_{ij}")
    ylabel("PDF")
    title("\theta Distribution for eigenvector " + i)
    xlim([0 pi])

end
figure(N+1)
x=[0:pi/1000:pi];
N=N*2;
h_theta = 1/sqrt(pi) * gamma(N/2)/gamma((N-1)/2) * (sin(x).^(N-2));
plot(x,h_theta)
xlabel("\theta_{ij}")
ylabel("PDF")
title("h(\theta)")
xlim([0 pi])

