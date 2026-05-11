%% Project Work - Part 1: Analysis of the Employment Programme

clear
clc
close all


%% 1. Importing the data as tables
% Vertical concatenation of the two text files into a single table.
% nswre74_treated.txt: 185 treated individuals (NSW experiment).
% psid_controls.txt:   2490 non-treated controls (PSID).

T = [readtable('nswre74_treated.txt'); readtable('psid_controls.txt')];

T.Properties.VariableNames = {'treatment' 'age' 'education' 'black' ...
    'hispanic' 'married' 'nodegree' 'RE74' 'RE75' 'RE78'};

size(T)         % Dimension of the dataset (2675 x 10).
head(T, 2)      % First two rows.


%% 2. Data preparation (Section 4.1)
% Restrict the sample to African Americans (black = 1) and drop the
% variables black, hispanic and RE74. Seven variables remain.

T(T.black == 0, :) = [];                            % Logical row deletion.
T = removevars(T, {'black', 'hispanic', 'RE74'});   % Drop columns.

size(T)         % 780 x 7.


%% 3. Summary statistics by treatment status (Section 4)
% Means, medians and standard deviations of age, education, married,
% nodegree, RE75 and RE78 separately for participants and non-participants.
% grpstats automatically uses all numeric variables except the grouping one.

summary_stats = grpstats(T, 'treatment', {'mean', 'median', 'std'})


%% 4. OLS estimator and treatment effect (Section 4.2.1)
% theta_hat = (1/N1) * sum_{treated} (mu(1) - mu(0)),
% with mu(j) = X * b_j and b_j = inv(X_j'*X_j) * X_j'*y_j for j = 0, 1.

% Construct the 780 x 6 design matrix X and the dependent variable y.
N = size(T, 1);
X = [ones(N, 1), T.age, T.education, T.married, T.nodegree, T.RE75];
y = T.RE78;

% Treatment indicator and subsample sizes.
treated = (T.treatment == 1);
N1 = sum(treated)
N0 = N - N1

% Two OLS estimators (one per group), implemented via the user-defined
% value class OLS.m (Lecture 8).
ols1 = OLS(X(treated,  :), y(treated));
ols0 = OLS(X(~treated, :), y(~treated));

% Predicted earnings under each regime for all 780 observations.
mu1 = X * ols1.beta;
mu0 = X * ols0.beta;

% Average treatment effect on the treated.
theta_hat = mean(mu1(treated) - mu0(treated))


%% 5. Bootstrap standard error - serial loop (Section 4.2.2)
% B = 299 nonparametric bootstrap replications. See equation (4.6).

B = 299;
rng(123)
theta_b = zeros(B, 1);

tic
for b = 1:B
    idx = randi(N, 1, N);                       % Bootstrap indices.
    Xb = X(idx, :);
    yb = y(idx);
    tb = treated(idx);

    ols1_b = OLS(Xb(tb,  :), yb(tb));           % OLS on resampled treated.
    ols0_b = OLS(Xb(~tb, :), yb(~tb));          % OLS on resampled controls.

    theta_b(b) = mean(Xb(tb, :) * (ols1_b.beta - ols0_b.beta));
end
elapsed_time_serial = toc

% Bootstrap standard error, equation (4.6).
se_theta = sqrt(sum((theta_b - mean(theta_b)).^2) / (B - 1))


%% 6. Bootstrap standard error - parallel loop (Lecture 7)
% Same computation written as a parfor loop. M is the maximum number of
% workers; M = 0 forces serial execution.

delete(gcp('nocreate'))                         % Close any existing pool.
try
    M = 3;                                      % Number of workers.
    parpool(M);
catch
    M = 0;                                      % Fall back to serial parfor.
    warning('Parallel pool unavailable. Running parfor serially.')
end

rng(123)
theta_b_par = zeros(B, 1);

tic
parfor (b = 1:B, M)
    idx = randi(N, 1, N);
    Xb = X(idx, :);
    yb = y(idx);
    tb = treated(idx);

    ols1_b = OLS(Xb(tb,  :), yb(tb));
    ols0_b = OLS(Xb(~tb, :), yb(~tb));

    theta_b_par(b) = mean(Xb(tb, :) * (ols1_b.beta - ols0_b.beta));
end
elapsed_time_parallel = toc

elapsed_time_serial / elapsed_time_parallel     % Ratio serial vs. parallel.

delete(gcp('nocreate'))                         % Shut down the parallel pool.


%% 7. Visualization of the bootstrap distribution (Lecture 4)

figure
ax = gca;
histogram(theta_b, 25, 'FaceColor', 'red')
hold on
xline(theta_hat,     'b-',  'LineWidth', 2)
xline(mean(theta_b), 'k--', 'LineWidth', 2)
ax.XLabel.String = 'Bootstrap replicates of theta hat';
ax.YLabel.String = 'Frequency';
title('Bootstrap distribution of the treatment effect (B = 299)')
hold off
