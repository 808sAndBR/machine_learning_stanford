function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypo = sigmoid(X * theta);


first_term = -y' * log(hypo);
second_term = (1 - y') * log(1-hypo);

% set theta(1) to 0 to exclude bias	feature
theta(1) = 0;

% square theta
theta_sq = sum(theta' * theta);

J = 1/m * (first_term - second_term) + ((lambda/(2*m)) * theta_sq);

hypo2 = sigmoid(X * theta);

grad = (((X' * (hypo-y)) * 1/m) + (lambda/m)*theta);

% =============================================================

end
