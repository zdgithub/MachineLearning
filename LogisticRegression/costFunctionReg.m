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

col = size(theta,1);

theta2 = theta(2:col);
    
h = sigmoid(X * theta);
y1 = y' * log(h);
y2 = (y' * (-1) + 1) * log(h * (-1) + 1);
J = (y1 + y2) * (-1) / m + theta2' * theta2 * lambda / (2*m);

% «Û»°Ã›∂»
grad(1) = sum((h - y)) / m;

for j=2:col
    grad(j) = (h - y)' * X(:,j) / m + lambda / m * theta(j);
end




% =============================================================

end
