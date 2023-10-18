co.k=20;
co.NSmethod='kdtree';

num_samples=1e5;
alpha=1e-4;
X=rand(1,num_samples);
V=rand(1,num_samples);
Y=X+alpha*V;

H=HShannon_gkNN_estimation([X; Y],co);
