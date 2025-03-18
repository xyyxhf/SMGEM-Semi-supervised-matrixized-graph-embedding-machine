function X=SGXCX(XE,nk,nr)
%%%% XE 输入向量   20<nk<50   nr<nk
X1=phasespacega(XE,nk,1); %重构相空间矩阵
 X1=X1;%相空间重构
[md,d]=size(X1);
A=X1'*X1;
M = blkdiag(A', -A);
N=M^2;
[m1,d1]=size(A);
[U,S,V]=svd(N(1:m1,1:m1),'econ');%奇异值分解
U1=U(:,1:nr);V1=V(:,1:nr);
X=U1*V1';