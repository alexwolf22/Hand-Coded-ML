function pred_Y = linearReg_predict(theta, X, mode)
% Predicts the output values of the input examples X, given the learned parameter vector theta.
% 
% INPUT
%  theta: [n x 1] vector, containing the learned model parameters
%  X: [m x d] matrix, where each row is a d-dimensional input example
%  mode: type of features, either 'linear' or 'quadratic'
%
% OUTPUT
%  pred_Y: [m x 1] vector, containing the predicted output values.
%

X=linearReg_features(X, mode);

pred_Y= X*theta;

end