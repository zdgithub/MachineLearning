function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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
% 计算输出h(x)=a3
X = [ones(m,1) X]; %加上bias unit
z2 = Theta1 * X';
a2 = sigmoid(z2);
a2 = [ones(1,m); a2]; %加上bias unit
z3 = Theta2 * a2;
a3 = sigmoid(z3);  % 10 x 5000

% 计算代价函数
num = 0;  %注意不能写成sum，它可是内置函数啊
for i=1:m
    for k=1:num_labels
        xi = a3(:,i);
        yy = log(xi(k)) * (-1) * (y(i)==k) - (1- (y(i)==k)) * log(1-xi(k));
        num = num + yy;
    end
end
J = num / m;

% 正则化代价函数
th1 = Theta1(:,2:end);
th1 = sum(th1.^2);
th2 = Theta2(:,2:end);
th2 = sum(th2.^2);
J = J + lambda / (2*m) * (sum(th1) + sum(th2));

% 反向传播
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for t=1:m
    xxi = X(t,:)'; %每一个样例输入
    zz2 = Theta1 * xxi;
    aa2 = sigmoid(zz2);
    aa2 = [1; aa2];
    zz3 = Theta2 * aa2;
    aa3 = sigmoid(zz3);
    yi = zeros(num_labels,1);
    yi(y(t)) = 1;
    dert3 = aa3 - yi;
    dert2 = Theta2' * dert3 .*  [1;sigmoidGradient(zz2)]; %此处注意对bias unit的处理，最后要扔掉
    Delta2 = Delta2 + dert3 * aa2';
    dert2 = dert2(2:end); % 扔掉bias unit
    Delta1 = Delta1 + dert2 * xxi';
end

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end) .* lambda ./m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end) .* lambda ./m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
