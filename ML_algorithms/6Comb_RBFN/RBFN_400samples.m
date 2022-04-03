clc; clear; close all;
%% Radial Basis Function (RBF) Neural Network
load ("Training_input.mat")
load ("Training_output.mat")
load ("Test_input.mat")
load("Test_output.mat")

X_train = table2array(Traininginput);
y_train = table2array(Trainingoutput);

X_test = table2array(Testinput);
y_test = table2array(Testoutput);
%% Normalization of the input parameters
X_train_normalized = normalize(X_train);
X_test_normalized = normalize(X_test);
%% RBFN 
goal = 0.000001;
spread = 1;
%test
net_test = newrb(X_train_normalized,y_train,goal);
view(net_test)
results_test = sim(net_test,X_test_normalized); %277 hidden neurons
%% plot for test
freq_range = 50:2:1600;
for i = 1:4;
        figure;
        plot(freq_range,results_test(:,i))
        hold on
        plot(freq_range,y_test(:,i))
        legend("Predicted output", "Real output")
        ylim([0 1])
        xlabel("Frequency (Hz)")
        ylabel("Sound absorption coefficient")
        title("RBFN with 400 samples")
end

%% metric performance for test

% test
MSE_test = perform(net_test,y_test,results_test);
RMSE_test = sqrt(MSE_test);
MAE_test = mae(y_test-results_test);
pre_MAPE_test = abs((results_test-y_test)./y_test);
MAPE_test = mean(pre_MAPE_test(isfinite(pre_MAPE_test)));
metric_performance_test = [MSE_test RMSE_test MAE_test MAPE_test]


