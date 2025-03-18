% Haifeng Xu, Anhui University of Technology, January 2023. 
% Contact information: see readme.txt.
%
% Reference: 
% Pan H, Xu H, Zheng J, et al. A semi-supervised matrixized graph embedding machine for roller bearing 
% fault diagnosis under few-labeled samples. IEEE Transactions on Industrial Informatics.
% 
% First written by Haifeng Xu, Anhui Universiy of Technology, October 2021.

function [L,W] = semi_adjacency(knn,X)  

options = struct( 'NN', knn, ...
                 'GraphDistanceFunction', 'euclidean', ... 
                 'GraphWeights', 'heat', ...
                 'GraphWeightParam', 0, ...
                 'LaplacianNormalize', 1, ...
                 'LaplacianDegree', 1);
             
% options = struct( 'NN', 16, ...
%                  'GraphDistanceFunction', 'euclidean');
sz=size(X);
X=reshape(X,sz(1)*sz(2),[])';

W = adjacency(options,X);
D = sum(W,2);
L = spdiags(D,0,speye(size(W,1)))-W; 
end

function A = adjacency(options,X)
% {adjacency} computes the graph adjacency matrix.

n=size(X,1);
p=2:(options.NN+1);

if size(X,1)<500 % block size: 500
    step=n;
else
	step=500;
end

idy=zeros(n*options.NN,1);
DI=zeros(n*options.NN,1);
t=0;
s=1;

for i1=1:step:n    
    t=t+1;
    i2=i1+step-1;
    if (i2>n) 
        i2=n;
    end

    Xblock=X(i1:i2,:);  
    dt=feval(options.GraphDistanceFunction,Xblock,X);
    [Z,I]=sort(dt,2);
	 	    
    Z=Z(:,p)'; I=I(:,p)'; [g1,g2]=size(I);
    idy(s:s+g1*g2-1)=I(:);DI(s:s+g1*g2-1)=Z(:);
    s=s+g1*g2;
end 

I=repmat((1:n),[options.NN 1]); I=I(:);

t=mean(DI(DI~=0)); 
A=sparse(I,idy,exp(-DI.^2/(2*t*t)),n,n);
A=A+((A~=A').*A'); 
end


function D = euclidean(A,B)
% {euclidean} computes the Euclidean distance.
if (size(A,2) ~= size(B,2))
    error('A and B must be of same dimensionality.');
end

if (size(A,2) == 1) 
    A = [A, zeros(size(A,1),1)]; B = [B, zeros(size(B,1),1)];
end
a = dot(A,A,2);b = dot(B,B,2);ab=A*B';
D = real(sqrt(bsxfun(@plus,a,b')-2*ab));
end