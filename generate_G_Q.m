function [GZ,Q] = generate_G_Q(N,L)

    z=exp(1j*pi*[0:N-1]'.*[0:N-1]'/N)/sqrt(N);
    Z=toeplitz(z,[z(1); z(end:-1:2)]);
    
    h=(randn(N,1)+1i*randn(N,1)).*exp(-[0:N-1]'/3);
    H=toeplitz(h,[h(1); h(end:-1:2)]);
    % [F,Hf]=eig(H);
    %%
    % Precoder design
    E=eye(N/L);E=upsample(E,L);
    A=E'*H;
    R=A'*A+0.0001*eye(N);
    a=[1; zeros(N/L-1,1)];
    p=A'*a;
    g=R\p; %g=g/sqrt(g'*g);
    G=toeplitz(g,[g(1); g(end:-1:2)]);
    GZ=G*Z; gz=GZ(:,1);
    % s=2*randi([0,1],N/L,1)-1;
    % [s E'*H*G*E*s]
    %%
    % AN subspace
    [V D]=eig(R);
    NNL=N-N/L;
    Q=V(:,1:NNL);

end