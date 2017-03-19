function [ l_delta, l_error ] = backPropagation( l,y,synMatrix,numHiddenLayers )
%Backpropagation

% Back propagation of errors using the chain rule.
    l_error{numHiddenLayers} = y - l{numHiddenLayers+1};
    
    for k = numHiddenLayers:-1:2
       l_delta{k} = l_error{k} .* activationFunction(l{k+1}, 1);
       l_error{k-1} = l_delta{k} * synMatrix{k}.';
    end
    l_delta{1} = l_error{1} .* activationFunction(l{2}, 1);
end

