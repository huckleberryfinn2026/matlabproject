classdef StatsUtils
    % StatsUtils  Static utility class for common statistical computations.
    %
    % All methods are declared Static (Lecture 8: static classes), so they
    % are called without instantiating the class, e.g.
    %     se = StatsUtils.olsSE(X, residuals);
    % Grouping the OLS post-estimation formulas in one class keeps the
    % main script readable and avoids copy-pasting identical code in
    % both the bootstrap loop and the Monte Carlo loop.

    methods (Static)

        function se = olsSE(X, residuals)
            % OLS standard errors from the design matrix and residuals.
            n   = size(X, 1);
            k   = size(X, 2);
            mse = (residuals' * residuals) / (n - k);
            se  = sqrt(diag(mse * (X' * X)^(-1)));
        end

        function t = tStat(beta, se)
            % t-statistics: coefficient divided by its standard error.
            t = beta ./ se;
        end

        function p = pVal(t, df)
            % Two-sided p-values from a t-distribution with df degrees.
            p = 2 * (1 - tcdf(abs(t), df));
        end

        function s = bootSE(theta_boot)
            % Bootstrap standard error, equation (4.6) of the project.
            B = numel(theta_boot);
            s = sqrt(sum((theta_boot - mean(theta_boot)).^2) / (B - 1));
        end

    end
end
