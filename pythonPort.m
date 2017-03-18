% The following code creates the input matrix. Although not mentioned in the video, the third column is for accommodating the bias term and is not part of the input.
% In[25]:
clear all
clc

%input data
X = [[0,0,1] ; [0,1,1] ; [1,0,1] ;[1,1,1]];

% The output of the exclusive OR function follows.

% In[26]:

%output data
y = [0 ; 1 ; 1 ; 0];


% The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

% In[27]:

rng('default');
rng(1);


% Now we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer.  It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value. Note that neither of the neural networks shown in the video describe the example.

% In[28]:
inputBuffer = 3;
hiddenNeurons = 4;
outputNeurons = 1;

% synapses
syn0 = 2*rand(inputBuffer,hiddenNeurons) - 1;
syn1 = 2*rand(hiddenNeurons,hiddenNeurons) - 1;
syn2 = 2*rand(hiddenNeurons,outputNeurons) - 1;

% This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases.

% In[29]:

% training step
% Python2 Note: In the follow command, you may improve
%   performance by replacing 'range' with 'xrange'.
counter = 1;
for j = 1:60000

    % Calculate forward through the network.
    l0 = X;
    l1 = activationFunction(l0 * syn0, 0);
    l2 = activationFunction(l1 * syn1, 0);
    l3 = activationFunction(l2 * syn2, 0);

    % Back propagation of errors using the chain rule.
    l3_error = y - l3;
    if(mod(j,10000) == 0)   % Only print the error every 10000 steps, to save time and limit the amount of output.
        fprintf('Error: %f \n', mean(abs(l3_error)));
        plot(counter,mean(abs(l3_error)),'r*-');
        xlabel('Error every 10,000 iters');
        ylabel('Error values');
        pause(0.05)
        hold on;
        counter= counter + 1;
    end
        
    l3_delta = l3_error .* activationFunction(l3, 1);

    l2_error = l3_delta * syn2.';

    l2_delta = l2_error .* activationFunction(l2, 1);

    l1_error = l2_delta * syn1.';

    l1_delta = l1_error .* activationFunction(l1, 1);

    % update weights (no learning rate term)
    syn2 = syn2 + l2.'*(l3_delta);
    syn1 = syn1 + l1.'*(l2_delta);
    syn0 = syn0 + l0.'*(l1_delta);
    
    
end
       
fprintf('Output after training\n')
disp(l3)
