% Haifeng Xu, Anhui University of Technology, January 2023. 
% Contact information: see readme.txt.
%
% Reference: 
% Pan H, Xu H, Zheng J, et al. A semi-supervised matrixized graph embedding machine for roller bearing 
% fault diagnosis under few-labeled samples. IEEE Transactions on Industrial Informatics.
% 
% First written by Haifeng Xu, Anhui Universiy of Technology, October 2021.

function [W , b, obj_recent] = ADMM_solver(Xl, Xu, yl, L, tau, lambda)

if (~exist('max_iter', 'var'))
    max_iter = 50;
end

if (~exist('eps', 'var'))
    eps = 1e-8;
end

if (~exist('rho', 'var'))
    rho = 0.01; 
end

if (~exist('eta', 'var'))
    eta = 0.999; 
end

X = cat(3,Xl,Xu);
[~,~,l] = size(Xl);
[m,n,lu] = size(X);

X = reshape(X,m*n,lu)';
Xl = reshape(Xl,m*n,l)';
L = reshape(L,lu*lu,1);

XI = kron(X,ones(lu,1));
XJ = repmat(X, lu, 1);  
XminusX = (XI - XJ) .* kron(L,ones(1,m*n));
clear XI XJ

%
c = 1/(1 + rho + lambda * sum(sum(XminusX .* XminusX),2));
sqrtc = c^2;
coeff1 = (rho + 1) * sqrtc - 2*c;
coeff2 = lambda * sqrtc;
%
H = coeff1 * ((Xl * Xl.') .* (yl * yl.')) ...
    + coeff2 * (( (Xl * XminusX.')  * (Xl * XminusX.').') .* (yl * yl.'));

% 
s_km1 = zeros(m*n, 1);
s_hatk = s_km1;
lambda_km1 = ones(m*n, 1);
lambda_hatk = lambda_km1;
t_k = 1;
c_km1 = 0;

recent_number = max_iter;
recent_idx = 0;
obj_recent = zeros(recent_number, 1);

for k=1: max_iter

    h = 1 + coeff1 * ((Xl * (lambda_hatk + rho * s_hatk)) .* yl) ...
        + coeff2 * (((XminusX * Xl.').' * (XminusX * (lambda_hatk + rho * s_hatk)) ) .* yl );

    LB = zeros(l,1);
    UB = (1/l) * ones(l,1);
    Aeq=[];
    for i=1:l
        Aeq=[Aeq;yl'];
    end
    beq=zeros(l,1);
    alpha=quadprog(-H,-h,[],[],Aeq,beq,LB,UB);

    w_k = c * ( lambda_hatk + rho * s_hatk + Xl'*(alpha.*yl));
    sel = (alpha > 0) & (alpha < (1/l));
    b = sel' * (yl - Xl * w_k) / sum(sel);

    W_k = reshape(w_k, m, n);
    Lambda_k = reshape(lambda_hatk, m, n);
    [S, nuc_sumvalue] = semi_wnnm_shrinkage(rho*W_k - Lambda_k, tau);
    s_k = reshape(S/rho, [m*n, 1]);

    lambda_k = lambda_hatk - rho * (w_k - s_k);
    
    c_k = (lambda_k - lambda_hatk)' * (lambda_k - lambda_hatk) / rho ...
        + rho * (s_k - s_hatk)' * (s_k - s_hatk);
    
    W = w_k;
    
    if (c_k < eta * c_km1)
        t_kp1 = 0.5 * (1 + sqrt(1 + 4*t_k*t_k));
        s_hatkp1 = s_k + (t_k-1) / t_kp1 * (s_k - s_km1);
        lambda_hatkp1 = lambda_k + (t_k-1) / t_kp1 * (lambda_k - lambda_km1);
        restart = false;
    else
        t_kp1 = 1;
        s_hatkp1 = s_km1;
        lambda_hatkp1 = lambda_km1;
        c_k = c_km1 / eta;
        restart = true;
    end
    
    s_hatk = s_hatkp1;
    lambda_hatk = lambda_hatkp1;
    c_km1 = c_k;
    s_km1 = s_k;
    lambda_km1 = lambda_k;
    t_k = t_kp1;
    
    obj_k = semi_obj_value(w_k, b, Xl, XminusX, yl, nuc_sumvalue, tau, lambda);
    recent_idx = recent_idx + 1;
    obj_recent(recent_idx) = obj_k;
    if (recent_idx == recent_number)
        recent_idx = 0;
    end
    if mod(k, 1000) == 0
        rk = sum(svd(reshape(w_k, m, n))>1e-6);
        fprintf('k=%d, obj=%f, restart=%d, rank=%d\n', k, obj_k, restart, rk);
    end
    
    if (abs(obj_k - mean(obj_recent)) / abs(mean(obj_recent)) < eps && k > recent_number)
        break;
    end
end
fprintf('stop_iter %.f',k);
end

