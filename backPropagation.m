function [ l_delta, l_error ] = backPropagation( l,y,synMatrix )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

% Back propagation of errors using the chain rule.
    l3_error = y - l{4};
    
    l3_delta = l3_error .* activationFunction(l{4}, 1);
    
    l2_error = l3_delta * synMatrix{3}.';
    
    l2_delta = l2_error .* activationFunction(l{3}, 1);
    
    l1_error = l2_delta * synMatrix{2}.';
    
    l1_delta = l1_error .* activationFunction(l{2}, 1);
    
    l_delta = {l1_delta l2_delta l3_delta};
    l_error = {l1_error l2_error l3_error};
end

