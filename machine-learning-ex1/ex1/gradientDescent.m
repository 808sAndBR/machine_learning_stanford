function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % temp0 = theta(1) - sum( alpha * (1/m) * ((X * theta) - y));
    % temp1 = theta - sum( alpha * (1/m) * (((X * theta) - y)) .* X);

    % theta(1) = temp0
    % theta(2) = temp1
    hypo = X * theta;
    errors = hypo - y;
    theta_change = (X' * errors) * alpha * (1/m);
    theta = theta - theta_change;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
