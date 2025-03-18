% Haifeng Xu, Anhui University of Technology, January 2023. 
% Contact information: see readme.txt.
%
% Reference: 
% Pan H, Xu H, Zheng J, et al. A semi-supervised matrixized graph embedding machine for roller bearing 
% fault diagnosis under few-labeled samples. IEEE Transactions on Industrial Informatics.
% 
% First written by Haifeng Xu, Anhui Universiy of Technology, October 2021.

function obj = semi_obj_value(w, b, Xl, XminusX, y,  nuc_sumvalue, tau, lambda)
[~,~,l] = size(Xl);
obj = 0.5 * (w') * w + (1/l) * sum(max(0, 1 - y .* (Xl * w + b))) + tau * nuc_sumvalue ...
    + 0.5 * lambda * (sum(XminusX * w))^2;
end