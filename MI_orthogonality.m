
%% 2D case
runs=2000;
covar = zeros(runs,3,3);
for i=1:runs
N=100000;
theta = 2*pi*rand(1,N);
A=[cos(theta);sin(theta)];
B=[sin(theta);-cos(theta)];
s=normrnd(0,sqrt(1),1,N);
v=normrnd(0,sqrt(0.9),1,N);

x=A.*s+B.*v;
% x=s+v;
covar(i,:,:)=cov([x;s]');
end
avg_cov = squeeze(sum(covar,1)/runs)


co_knn = IShannon_HShannon_initialization(1);
co_edge = IShannon_HShannon_initialization(1);
co_edge.member_name = 'Shannon_Edgeworth';
co_edge.member_co = H_initialization(co_edge.member_name,1);
co_kcca = IKCCA_initialization(1);
co_kgv = IKGV_initialization(1);

MI_knn = IShannon_HShannon_estimation([x;s],[2;1],co_knn);
MI_edge = IShannon_HShannon_estimation([x;s],[2;1],co_edge);
MI_kcca = IKCCA_estimation([x;s],[2;1],co_kcca);
MI_kgv = IKGV_estimation([x;s],[2;1],co_kgv);
test_knn = IShannon_HShannon_estimation([s;theta.*s/(2*pi)],[1;1],co_knn);
test_edge = IShannon_HShannon_estimation([s;theta.*s/(2*pi)],[1;1],co_edge);

fprintf("knn MI: %.4f\nedgeworth MI: %.4f\n",MI_knn,MI_edge);
