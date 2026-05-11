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




