function [beta_hat, se_hat, t_vals, p_vals] = ols_estimate(X, y)
% ols_estimate computes the slope coefficients, standard errors, t-values, and p-values of the OLS model
% Input X,y: Matrix, Array.
% Output beta_hat, se_hat, t_vals, p_vals: Arrays.

    % Find slope coefficients
    beta_hat = (X' * X)^(-1) * X' * y; 

    % Residuals and MSE estimate
    residuals  = y - X * beta_hat;     
    MSE = (residuals' * residuals) / (size(X, 1) - size(X, 2));  

    % Compuete Standard errors
    se_hat = sqrt(diag(MSE * (X' * X)^(-1)));     

    % Estimate t-values
    t_vals = beta_hat ./ se_hat; 

    % Compute p-values using t-values and Student's t distribution
    p_vals = 2 * (1 - tcdf(abs(t_vals), size(X, 1) - size(X, 2)));

end