% Program to find Network parameter of a Radial-basis neural network model
% MATLAB function 'newrb' is used to train a neural network for a given
% data set.

% The function has form net = newrb(Y, X, goal,spread). 
% spread (a network parameter) will affect the model performance.

% The program is used to find the best spread for a given data set
% using a batch K-fold cross-validation method.

%==========================================================================
% Required Inputs:
%==========================================================================
% 'Y' is the output vector or reponse (N-by-1)
% 'X' is the matrix of input vectors ((N-by-D)
% 'goal' is the mean squared-error goal, and is fixed
% 'spread constant' a netwrok parameter to be determined
%==========================================================================
% Output
%==========================================================================
% Box-plot showing cross-validation vs. spread constant
%==========================================================================
% Version 1.0
% Date: 20th September,2019
%==========================================================================
% Developed by : Arshad Afzal, India, Email: arshad.afzal@gmail.com 
%==========================================================================
% Copyright (c) 2019, Arshad Afzal
%==========================================================================

fprintf('\n         ==================================================================================');
fprintf('\n                      Learning network parameter of a Radial-basis Neural Network model ');
fprintf('\n         ==================================================================================\n');

load ("Training_input.mat")
load ("Training_output.mat")

X = table2array(Traininginput)';
Y = table2array(Trainingoutput)';

N = size(X,1); % N = size(Y,2)
D = size(X,2);
x = X';
y = Y';
B = input('\nEnter the number of batches of cross-validation :');
sc = [0.01, 0.1, 1, 1.5]%input('\nEnter the values for spread constant as a row-vector:');
M = size(sc,2);

% Initilization
CVerror = zeros(B,M);
goal = 0.000001; % sum-squared error goal

%==========================================================================
% Main Program
%==========================================================================
for j = 1:M
   for i = 1:B
    sse = 0;
% 10 Fold cross-validation is used
    Indices = crossvalind('Kfold', N, 10);
       for k = 1:10
          test = (Indices == k); train = ~test;
          net = newrb(x(1:D,train),y(train)',goal,sc(j));
          yhat = sim(net,x(:,test));
          % squared-error
          sse = sse + sum((yhat' - Y(test)).^2);
       end
    % Mean-squared error
    CVerror (i,j) = sse / 10;
   end
end
% Making the box-plot for analysis
boxplot(CVerror)