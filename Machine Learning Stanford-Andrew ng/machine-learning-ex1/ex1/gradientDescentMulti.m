function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
Multi=~(1==min(size(X)));

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
if Multi
    n=length(theta);
end
for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    if ~Multi
        Value = (X.*theta-y)*X';
        theta=theta-Value/m*alpha;
        J_history(iter) = computeCost(X, y, theta);
    else
        for i=1:n
            value=sum((X*theta-y).*X(:,i));
            theta(i)=theta(i)-value/m*alpha;
        end
         J_history(iter) = computeCostMulti(X, y, theta);
    end
   
end

