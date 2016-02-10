function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%


% solve for hypothesis
hypo = sigmoid(X * theta);

% solve each half of the sigma equation then subtract and scale
first_term = -y' * log(hypo);
second_term = (1 - y') * log(1-hypo);
J = 1/m * (first_term - second_term);

grad = (X' * (hypo-y)) * 1/m;

% =============================================================

end
