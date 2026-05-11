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

% Step 1: Logit regression
% Regress the original treatment variable on the 5 covariates.
% glmfit automatically adds an intercept, so we pass 5 columns (no ones column).
% Result b_logit is 6x1: [intercept; b_age; b_educ; b_married; b_nodegree; b_RE75]
X_covariates = [age, education, married, nodegree, RE75];   % 780x5
b_logit = glmfit(X_covariates, treatment, 'binomial', 'link', 'logit');

% Build the full design matrix with intercept (780x6) for computing X*b_logit
X_full = [ones(780, 1), age, education, married, nodegree, RE75];

fprintf('Logit coefficients (b_logit):\n');
disp(b_logit);

% Step 2: Create new artificial treatment variable for the 624 control obs
% Compute linear predictor for all 780 obs, apply indicator function 1{X*b_logit > 0}
linear_pred      = X_full * b_logit;                   % 780x1
treatment_new    = linear_pred > 0;                    % 780x1 logical: 1 if > 0, else 0
treatment_sim    = treatment_new(control_group);       % 624x1: drop the 156 actual participants

fprintf('New treated (in sim):   %d\n', sum(treatment_sim));
fprintf('New control (in sim):   %d\n', sum(~treatment_sim));

% Step 3: Simulate age, education, RE75 by binary group
% Binary variables for the 624 control obs (kept unchanged from empirical data)
married_ctrl  = married(control_group);    % 624x1
nodegree_ctrl = nodegree(control_group);   % 624x1

% Overall empirical min/max for clipping simulated values
min_vals = [min(age), min(education), min(RE75)];
max_vals = [max(age), max(education), max(RE75)];

% Preallocate simulated continuous variables (624 rows x 3 cols: age, educ, RE75)
cont_sim = zeros(624, 3);

% All 8 combinations of (treatment, married, nodegree) — each can be 0 or 1
binary_combos = [0 0 0; 0 0 1; 0 1 0; 0 1 1;
                 1 0 0; 1 0 1; 1 1 0; 1 1 1];

for k = 1:8
    t_val = binary_combos(k, 1);
    m_val = binary_combos(k, 2);
    n_val = binary_combos(k, 3);

    % Rows in empirical data (780 obs) matching this group
    emp_idx = (treatment == t_val) & (married == m_val) & (nodegree == n_val);

    % Rows in simulated data (624 obs) matching this group
    sim_idx = (treatment_sim == t_val) & (married_ctrl == m_val) & (nodegree_ctrl == n_val);

    n_sim = sum(sim_idx);
    if n_sim == 0; continue; end   % skip empty groups

    % Compute group mean and covariance from empirical data
    emp_cont  = [age(emp_idx), education(emp_idx), RE75(emp_idx)];
    mu_grp    = mean(emp_cont);
    cov_grp   = cov(emp_cont);

    % Draw multivariate normal samples for this group
    samples = mvnrnd(mu_grp, cov_grp, n_sim);   % n_sim x 3

    % Clip to overall empirical min/max
    samples = max(samples, min_vals);
    samples = min(samples, max_vals);

    % Round age and education to integers (columns 1 and 2)
    samples(:, 1) = round(samples(:, 1));
    samples(:, 2) = round(samples(:, 2));

    cont_sim(sim_idx, :) = samples;
end

age_sim  = cont_sim(:, 1);   % 624x1
educ_sim = cont_sim(:, 2);   % 624x1
RE75_sim = cont_sim(:, 3);   % 624x1

% Step 4: Build simulated design matrix X_sim (624x6)
% Same column structure as X_full: [ones, age, education, married, nodegree, RE75]
% Binary variables (married, nodegree) are taken unchanged from the control group
X_sim = [ones(624, 1), age_sim, educ_sim, married_ctrl, nodegree_ctrl, RE75_sim];

% Step 5: Generate RE78_sim
% OLS coefficients from empirical data (section 4.2.1)
X1 = X_full(treated_group, :);         % 156x6 treated submatrix
X0 = X_full(control_group, :);         % 624x6 control submatrix
y1 = RE78(treated_group);              % RE78 for treated (156x1)
y0 = RE78(control_group);             % RE78 for control (624x1)

b1 = (X1' * X1) \ (X1' * y1);        % OLS coefs for treated group
b0 = (X0' * X0) \ (X0' * y0);        % OLS coefs for control group

N1 = sum(treated_group);              % 156 treated
N0 = sum(control_group);              % 624 control
K  = size(X_full, 2) - 1;            % 5 (cols of X minus intercept)

% Residual standard deviations from empirical regressions (eq. 5.1 and 5.2)
sigma1 = sqrt((1 / (N1 - K)) * sum((X1 * b1 - y1).^2));
sigma0 = sqrt((1 / (N0 - K)) * sum((X0 * b0 - y0).^2));

% Standard normal noise vectors, drawn fresh each simulation run
u = randn(624, 1);
v = randn(624, 1);

% Simulated potential outcomes for all 624 obs (eq. 5.1 and 5.2)
mu_sim1 = X_sim * b1 + u * sigma1;   % what each obs would earn under treatment
mu_sim0 = X_sim * b0 + v * sigma0;   % what each obs would earn without treatment

% Assign RE78_sim: use mu_sim1 if treatment_sim=1, else mu_sim0
RE78_sim = mu_sim1 .* double(treatment_sim) + mu_sim0 .* double(~treatment_sim);




