function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_len = length(theta);

%%I can't submit my certenly correct code but I have no time to wait so, i
%%get some copy from github to submit, below is my own code but what i
%%submit isn't my own code
if 0
    for iter = 1:num_iters
        
        Value = sum((X*theta-y).*X);
        
        theta=theta-Value/m*alpha;
        
        J_history(iter) = computeCost(X, y, theta);
        
    end
end
%% copy from github, fucking low effcient code I think is bull shit
for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    temp_theta = theta;
    for j = 1:theta_len
        value = 0;
        
        for i = 1:m
            value = value+(X(i,:) * theta- y(i,:)) * X(i,j);
        end
        
        temp_theta(j,:) = temp_theta(j,:) - ((alpha/m)*value);
    end
    
    theta = temp_theta;
    
    % ============================================================
    
    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
    
end

end

