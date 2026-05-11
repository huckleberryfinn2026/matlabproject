classdef OLS
    % OLS regression: beta = inv(X'*X) * X'*y.
    % Value class in the style of CDF.m from Lecture 8.

    properties
        X       % Design matrix.
        y       % Dependent variable.
        beta    % OLS coefficient vector.
    end

    methods
        function obj = OLS(X, y)
            % Constructor: store the data and compute beta.
            obj.X = X;
            obj.y = y;
            obj.beta = inv(X' * X) * X' * y;
        end

        function y_hat = predict(obj, X_new)
            % Predicted values for a new design matrix.
            y_hat = X_new * obj.beta;
        end

        function r = residuals(obj)
            % In-sample residuals.
            r = obj.y - obj.X * obj.beta;
        end
    end
end