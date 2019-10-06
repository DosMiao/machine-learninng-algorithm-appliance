function [J,grad] = nnCostFunction_multi(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    hidden_layer_size2, ...
    hidden_layer_size3, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
over=hidden_layer_size * (input_layer_size + 1);
Theta1 = reshape(nn_params(1:over), ...
    hidden_layer_size, (input_layer_size + 1));
start=over+1;   add=hidden_layer_size2 * (hidden_layer_size + 1);   over=start+add-1;
Theta2 = reshape(nn_params(start:over), ...
    hidden_layer_size2, (hidden_layer_size + 1));
start=over+1;   add=hidden_layer_size3 * (hidden_layer_size2 + 1);  over=start+add-1;
Theta3 = reshape(nn_params(start:over), ...
    hidden_layer_size3, (hidden_layer_size2 + 1));
start=over+1;   add=num_labels * (hidden_layer_size3 + 1);  over=start+add-1;
Theta4 = reshape(nn_params(start:over), ...
    num_labels, (hidden_layer_size3 + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


a1 = [ones(m,1) X];
z2 = (a1*Theta1');
a2 = [ones(size(z2,1),1) sigmoid(z2)];
z3 = (a2*Theta2');
a3 = [ones(size(z3,1),1) sigmoid(z3)];
z4 = (a3*Theta3');
a4 = [ones(size(z4,1),1) sigmoid(z4)];
h_theta = sigmoid(a4*Theta4');
a5 = h_theta;
y_matrix = eye(num_labels);
y_matrix=y_matrix(y,:);
J = (-sum(sum(y_matrix.*log(h_theta))) - sum(sum((1-y_matrix).*(log(1-h_theta)))))/m;

% REGULARIZATION
regularization_term = (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2))...
    +sum(sum(Theta3(:,2:end).^2))+sum(sum(Theta4(:,2:end).^2)));
J = J + regularization_term;

% BACK PROPOGATION
d5 = a5 - y_matrix;                                             % has same dimensions as a3
d4 = (d5*Theta4).*[ones(size(z4,1),1) sigmoidGradient(z4)];     % has same dimensions as a2
d3 = (d4(:,2:end)*Theta3).*[ones(size(z3,1),1) sigmoidGradient(z3)]; 
d2 = (d3(:,2:end)*Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)]; 
D1 = d2(:,2:end)' * a1;    % has same dimensions as Theta1
D2 = d3(:,2:end)' * a2;
D3 = d4(:,2:end)' * a3;
D4 = d5' * a4;    % has same dimensions as Theta2

Theta1_grad = Theta1_grad + (1/m) * D1;
Theta2_grad = Theta2_grad + (1/m) * D2;
Theta3_grad = Theta3_grad + (1/m) * D3;
Theta4_grad = Theta4_grad + (1/m) * D4;


% REGULARIZATION OF THE GRADIENT

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + (lambda/m)*(Theta3(:,2:end));
Theta4_grad(:,2:end) = Theta4_grad(:,2:end) + (lambda/m)*(Theta4(:,2:end));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];

end
