%% PROJECT WORK: PART 1

clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TASK 3.1: Load datasets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

treated = readmatrix('nswre74_treated.txt');
control = readmatrix('psid_controls.txt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TASK 3.2 / 4.1: Data preparation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Combine treated and control observations into one dataset
data = [treated; control];

%   data(:,4) - Take all rows from column 4
%   data(:,4) == 1 - Is column 4 equal to 1?"; Result is a logical vector,
%   where TRUE = 1 and FALSE = 0
%   data(logical_vector,:) - Keep rows where logical vector = TRUE
%   data = data(data(:,4)==1,:) - (1) Look at column 4 (black), (2) Find rows where
%   value = 1, (3) Keep only those rows
data = data(data(:,4)==1,:);

% Extract relevant variables from the filtered dataset
treatment = data(:,1);
age = data(:,2);
education = data(:,3);
married = data(:,6);
nodegree = data(:,7);
RE75 = data(:,9);
RE78 = data(:,10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TASK 4.1: Summary statistics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create logical indices for treated and control groups
% treated_group contains TRUE for participants in the program
% control_group contains TRUE for non-participants
treated_group = treatment == 1;
control_group = treatment == 0;

%   variables matrix
X_summary = [age education married nodegree RE75 RE78];

%   variables names
varNames = {'Age','Education','Married','Nodegree','RE75','RE78'};

% Compute summary statistics for treated individuals
mean_treated = mean(X_summary(treated_group,:));
median_treated = median(X_summary(treated_group,:));
std_treated = std(X_summary(treated_group,:));

% Compute summary statistics for control individuals
mean_control = mean(X_summary(control_group,:));
median_control = median(X_summary(control_group,:));
std_control = std(X_summary(control_group,:));

% Create a table with summary statistics
% Rows correspond to variables
% Columns correspond to different statistics
SummaryTable = table( ...
    mean_treated', median_treated', std_treated', ...
    mean_control', median_control', std_control', ...
    'RowNames', varNames, ...
    'VariableNames', { ...
    'Mean_Treated', ...
    'Median_Treated', ...
    'Std_Treated', ...
    'Mean_Control', ...
    'Median_Control', ...
    'Std_Control'} ...
);

disp(SummaryTable)





%% Task 4.2.1: Estimation procedure
% theta_hat = (1/N1) * sum_{treated} (mu(1) - mu(0)),
% with mu(j) = X * b_j and b_j = inv(X_j'*X_j) * X_j'*y_j for j = 0, 1.
% OLS is computed via the user-defined value class OLS.m (Lecture 8).

% Design matrix (780 x 6) and dependent variable.
X = [ones(length(age), 1), age, education, married, nodegree, RE75];
y = RE78;
N = size(X, 1);

% Subsample sizes.
N1 = sum(treated_group)
N0 = sum(control_group)

% Two OLS estimators (one per group).
ols1 = OLS(X(treated_group, :), y(treated_group));
ols0 = OLS(X(control_group, :), y(control_group));

% Predicted earnings under each regime for all 780 observations.
mu1 = X * ols1.beta;
mu0 = X * ols0.beta;

% Average treatment effect on the treated.
theta = mean(mu1(treated_group) - mu0(treated_group))


%% Task 4.2.2: Bootstrap standard error - serial loop
% B = 299 nonparametric bootstrap replications. See equation (4.6).

B = 299;
rng(123)
theta_boot = zeros(B, 1);

tic
for b = 1:B

    % Bootstrap sample indices.
    boot_index = randi(N, N, 1);

    % Bootstrap sample.
    X_boot         = X(boot_index, :);
    y_boot         = y(boot_index);
    treated_boot   = treated_group(boot_index);     % logical for this draw

    % OLS on resampled treated and control groups.
    ols1_boot = OLS(X_boot(treated_boot,  :), y_boot(treated_boot));
    ols0_boot = OLS(X_boot(~treated_boot, :), y_boot(~treated_boot));

    % Bootstrap treatment effect.
    theta_boot(b) = mean(X_boot(treated_boot, :) * ...
                         (ols1_boot.beta - ols0_boot.beta));

end
elapsed_time_serial = toc

% Bootstrap standard error, equation (4.6).
se_theta = sqrt(sum((theta_boot - mean(theta_boot)).^2) / (B - 1))


%% Task 4.2.2: Bootstrap standard error - parallel loop (Lecture 7)
% Same computation written as a parfor loop. M is the maximum number of
% workers; M = 0 forces serial execution. If the Parallel Computing
% Toolbox is not available, the code falls back to M = 0 so that the
% parfor syntax is still demonstrated.

delete(gcp('nocreate'))                         % Close any existing pool.
try
    M = 3;                                      % Number of workers.
    parpool(M);
catch
    M = 0;                                      % Fall back to serial parfor.
    warning('Parallel pool unavailable. Running parfor serially.')
end

rng(123)
theta_boot_par = zeros(B, 1);

tic
parfor (b = 1:B, M)

    boot_index   = randi(N, N, 1);
    X_boot       = X(boot_index, :);
    y_boot       = y(boot_index);
    treated_boot = treated_group(boot_index);

    ols1_boot = OLS(X_boot(treated_boot,  :), y_boot(treated_boot));
    ols0_boot = OLS(X_boot(~treated_boot, :), y_boot(~treated_boot));

    theta_boot_par(b) = mean(X_boot(treated_boot, :) * ...
                             (ols1_boot.beta - ols0_boot.beta));

end
elapsed_time_parallel = toc

elapsed_time_serial / elapsed_time_parallel     % Ratio serial vs. parallel.

% Parallel bootstrap standard error (same formula, parallel results).
se_theta_par = sqrt(sum((theta_boot_par - mean(theta_boot_par)).^2) / (B - 1))

delete(gcp('nocreate'))                         % Shut down the parallel pool.


%% Visualization of the bootstrap distribution (Lecture 4)

figure
ax = gca;
histogram(theta_boot, 25, 'FaceColor', 'red')
hold on
xline(theta,            'b-',  'LineWidth', 2)
xline(mean(theta_boot), 'k--', 'LineWidth', 2)
ax.XLabel.String = 'Bootstrap replicates of theta hat';
ax.YLabel.String = 'Frequency';
title('Bootstrap distribution of the treatment effect (B = 299)')
hold off






% ── Section 5.1: Data Simulation ────────────────────────────────────────
% ── One-time setup (outside Monte Carlo loop) ───────────────────────────

% Step 1: Logit regression — computed once from empirical data
% glmfit adds intercept automatically; b_logit is 6x1
X_covariates = [age, education, married, nodegree, RE75];   % 780x5
b_logit = glmfit(X_covariates, treatment, 'binomial', 'link', 'logit');
X_full  = [ones(780, 1), age, education, married, nodegree, RE75];  % 780x6

fprintf('Logit coefficients (b_logit):\n');
disp(b_logit);

% Step 2: New artificial treatment for the 624 control obs — fixed across all simulations
% Apply indicator function 1{X*b_logit > 0} and keep only control group rows
linear_pred   = X_full * b_logit;                        % 780x1
treatment_sim = linear_pred(control_group) > 0;          % 624x1 logical
married_ctrl  = married(control_group);                   % 624x1, unchanged
nodegree_ctrl = nodegree(control_group);                  % 624x1, unchanged

fprintf('Simulated treated: %d  |  Simulated control: %d\n', ...
    sum(treatment_sim), sum(~treatment_sim));

% OLS coefficients and residual SDs from empirical data — fixed across simulations
X1 = X_full(treated_group, :);   X0 = X_full(control_group, :);
y1 = RE78(treated_group);         y0 = RE78(control_group);
b1 = (X1' * X1) \ (X1' * y1);
b0 = (X0' * X0) \ (X0' * y0);
N1 = sum(treated_group);   N0 = sum(control_group);   K = size(X_full, 2) - 1;
sigma1 = sqrt((1 / (N1 - K)) * sum((X1 * b1 - y1).^2));
sigma0 = sqrt((1 / (N0 - K)) * sum((X0 * b0 - y0).^2));

% Precompute MVN group parameters once — reused in every simulation iteration
binary_combos = [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1];
mu_groups   = cell(8, 1);
cov_groups  = cell(8, 1);
grp_sim_idx = cell(8, 1);
min_vals = [min(age), min(education), min(RE75)];
max_vals = [max(age), max(education), max(RE75)];

for k = 1:8
    t_val = binary_combos(k,1); m_val = binary_combos(k,2); n_val = binary_combos(k,3);
    emp_idx      = (treatment == t_val) & (married == m_val) & (nodegree == n_val);
    grp_sim_idx{k} = (treatment_sim == t_val) & (married_ctrl == m_val) & (nodegree_ctrl == n_val);
    emp_cont = [age(emp_idx), education(emp_idx), RE75(emp_idx)];
    if size(emp_cont, 1) >= 2
        mu_groups{k}  = mean(emp_cont);
        cov_groups{k} = cov(emp_cont);
    end
end

% ── Monte Carlo loop: 1000 simulations ──────────────────────────────────
n_mc = 1000;

for s = 1:n_mc

    % Step 3: Simulate age, education, RE75 using MVN within each binary group
    cont_sim = zeros(624, 3);
    for k = 1:8
        idx = grp_sim_idx{k};
        n_k = sum(idx);
        if n_k == 0 || isempty(mu_groups{k}); continue; end
        samples = mvnrnd(mu_groups{k}, cov_groups{k}, n_k);   % n_k x 3
        samples = max(samples, min_vals);   % clip to empirical min
        samples = min(samples, max_vals);   % clip to empirical max
        samples(:, 1) = round(samples(:, 1));   % age -> integer
        samples(:, 2) = round(samples(:, 2));   % education -> integer
        cont_sim(idx, :) = samples;
    end

    % Step 4: Build X_sim (624x6) — same structure as X_full
    X_sim = [ones(624, 1), cont_sim(:,1), cont_sim(:,2), ...
             married_ctrl, nodegree_ctrl, cont_sim(:,3)];

    % Step 5: Generate RE78_sim — u and v drawn fresh each iteration
    u = randn(624, 1);
    v = randn(624, 1);
    mu_sim1  = X_sim * b1 + u * sigma1;   % potential outcome under treatment
    mu_sim0  = X_sim * b0 + v * sigma0;   % potential outcome without treatment
    RE78_sim = mu_sim1 .* double(treatment_sim) + mu_sim0 .* double(~treatment_sim);

    % Section 5.2 (OLS estimator) will go here

end




