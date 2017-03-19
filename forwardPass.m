function [ l ] = forwardPass( trainSet, synMatrix, ...
    numHiddenLayers )
%forwardPass : Initial forward pass calculations for the backpropogation
%algorithm

% TrainSet = cat(3,[],[],[],[] ... [])
% Where there are X num of [] (based on input)
% Calculate forward through the network.
    % l = layer ?
    l{1} = trainSet;
    for i = 1:numHiddenLayers
       l{i+1} = activationFunction(l{i} * synMatrix{i},0);
    end
%     l1 = activationFunction(l0 * syn0, 0);
%     l2 = activationFunction(l1 * syn1, 0);
%     l3 = activationFunction(l2 * syn2, 0);   
end

