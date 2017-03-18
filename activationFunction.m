function [y] = activationFunction(x, deriv)
%activationFunction: applies sigmoid function to x
% deriv boolean, to know forward or back pass
    if (deriv == 1)
        y = (x.*(1-x));
    elseif (deriv == 0)
        y = (1./(1+exp(-x)));
    end
end

