function obj = semi_obj_value(w, b, Xl, XminusX, y,  nuc_sumvalue, tau, lambda)
[~,~,l] = size(Xl);
obj = 0.5 * (w') * w + (1/l) * sum(max(0, 1 - y .* (Xl * w + b))) + tau * nuc_sumvalue ...
    + 0.5 * lambda * (sum(XminusX * w))^2;
end