% The following code creates the input matrix. Although not mentioned in the video, the third column is for accommodating the bias term and is not part of the input.
% In[25]:
clear all;
clc;
close all;

%input data
X = [[0,0,1] ; [0,1,1] ; [1,0,1] ;[1,1,1] ;[0,0,1] ; [0,1,1] ; [1,0,1] ;[1,1,1] ;[0,0,1] ; [0,1,1] ; [1,0,1] ;[1,1,1]];

%output data
y = [0 ; 1 ; 1 ; 0 ;0 ; 1 ; 1 ; 0 ;0 ; 1 ; 1 ; 0];

rng('default');
rng(1);

% Now we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer.  It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value. Note that neither of the neural networks shown in the video describe the example.
inputBuffer = 3;
outputNeurons = 1;

hiddenNeurons = 4;
numHiddenLayers = 3;

synMatrix{1} = 2*rand(inputBuffer,hiddenNeurons) - 1;
for k = 2:(numHiddenLayers-1)
    synMatrix{k} = 2*rand(hiddenNeurons,hiddenNeurons) - 1;
end
synMatrix{numHiddenLayers} = 2*rand(hiddenNeurons,outputNeurons) - 1;

% This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases.

% training step
counter = 1;
epochs=1000;
numErrorPrints = 4;
for j = 1:epochs
    % Calculate forward through the network.
    l = forwardPass(X,synMatrix,numHiddenLayers);
    [l_delta,l_error]= backPropagation(l,y,synMatrix,numHiddenLayers);
    % Only print the error every 10000 steps, to save time and limit the amount of output.
    if(mod(j,epochs/numErrorPrints) == 0)
        fprintf('Error: %f \n', mean(abs(l_error{numHiddenLayers})));
%         plot(counter,mean(abs(l_error{numHiddenLayers})),'r*-');
%         xlabel('Error every 10,000 iters');
%         ylabel('Error values');
%         pause(0.05)
%         hold on;
%         counter= counter + 1;
    end
    % update weights (no learning rate term)
    for k = 1:numHiddenLayers
        synMatrix{k} = synMatrix{k} + l{k}.'*(l_delta{k});
    end
end

fprintf('Output after training\n')
disp(l{numHiddenLayers+1})
