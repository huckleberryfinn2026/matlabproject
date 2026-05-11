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

% Display output
disp(SummaryTable);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TASK 4.2.1: Estimation procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create X matrix
% Columns: constant, age, education, married, nodegree, RE75
X = [ones(length(age),1), ...
     age, ...
     education, ...
     married, ...
     nodegree, ...
     RE75];

% Define dependent variable
y = RE78;

% Number of observations
N = size(X,1);

% Number of treated and control observations
N1 = sum(treated_group);
N0 = sum(control_group);

% Split groups
X1 = X(treated_group,:);
X0 = X(control_group,:);

y1 = y(treated_group);
y0 = y(control_group);

% OLS coefficients
b1 = (X1' * X1) \ (X1' * y1);
b0 = (X0' * X0) \ (X0' * y0);

% Predicted outcomes
mu1 = X * b1;
mu0 = X * b0;

% Treatment effect
theta = mean(mu1(treated_group) - mu0(treated_group));

% Display
disp(theta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TASK 4.2.2: Bootstrap standard error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of bootstrap replications
B = 299;

% Set seed for reproducibility
rng(123)

% Preallocate vector for bootstrap treatment effects
theta_boot = zeros(B,1);

% Start timer
tic

for b = 1:B

    % Draw bootstrap sample indices with replacement
    boot_index = randi(N, N, 1);

    % Create bootstrap sample
    X_boot = X(boot_index,:);
    y_boot = y(boot_index);
    treatment_boot = treatment(boot_index);

    % Create treated/control indicators in bootstrap sample
    treated_boot_group = treatment_boot == 1;
    control_boot_group = treatment_boot == 0;

    % Split bootstrap sample into treated and control groups
    X1_boot = X_boot(treated_boot_group,:);
    X0_boot = X_boot(control_boot_group,:);

    y1_boot = y_boot(treated_boot_group);
    y0_boot = y_boot(control_boot_group);

    % Estimate OLS coefficients in bootstrap sample
    b1_boot = (X1_boot' * X1_boot) \ (X1_boot' * y1_boot);
    b0_boot = (X0_boot' * X0_boot) \ (X0_boot' * y0_boot);

    % Compute bootstrap treatment effect
    theta_boot(b) = mean(X_boot(treated_boot_group,:) * (b1_boot - b0_boot));

end

% Stop timer
elapsed_time_serial = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TASK 4.2.2: Parallel bootstrap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Same computation written as a parfor loop. 
% M is the maximum number of workers; M = 0 forces serial execution.

% Close any existing parallel pool
try

    % Number of workers
    M = 3;

    % Start parallel pool
    parpool(M);

catch

    M = 0;

    warning('Parallel pool unavailable. Running parfor serially.')

end

% Set seed for reproducibility
rng(123)

% Preallocate vector for bootstrap treatment effects
theta_boot_par = zeros(B,1);

% Start timer
tic

parfor (b = 1:B, M)

    % Draw bootstrap sample indices with replacement
    boot_index = randi(N, N, 1);

    % Create bootstrap sample
    X_boot = X(boot_index,:);
    y_boot = y(boot_index);
    treatment_boot = treatment(boot_index);

    % Create treated/control indicators in bootstrap sample
    treated_boot_group = treatment_boot == 1;
    control_boot_group = treatment_boot == 0;

    % Split bootstrap sample into treated and control groups
    X1_boot = X_boot(treated_boot_group,:);
    X0_boot = X_boot(control_boot_group,:);

    y1_boot = y_boot(treated_boot_group);
    y0_boot = y_boot(control_boot_group);

    % Estimate OLS coefficients in bootstrap sample
    b1_boot = (X1_boot' * X1_boot) \ (X1_boot' * y1_boot);
    b0_boot = (X0_boot' * X0_boot) \ (X0_boot' * y0_boot);

    % Compute bootstrap treatment effect
    theta_boot_par(b) = mean(X_boot(treated_boot_group,:) * (b1_boot - b0_boot));

end

% Stop timer
elapsed_time_parallel = toc;

% Ratio serial vs. parallel
if exist('elapsed_time_serial','var')
    speed_ratio = elapsed_time_serial / elapsed_time_parallel;
    disp(speed_ratio)
else
    disp('elapsed_time_serial not available because the serial bootstrap section was not run.')
end


% Bootstrap standard error
se_theta = sqrt(sum((theta_boot - mean(theta_boot)).^2) / (B - 1));

% Parallel bootstrap standard error
se_theta_par = sqrt(sum((theta_boot_par - mean(theta_boot_par)).^2) / (B - 1));

% Display bootstrap standard errors
disp(se_theta)
disp(se_theta_par)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization of the bootstrap distribution (Lecture 4)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
histogram(theta_boot,25)
hold on
xline(theta,'LineWidth',2)
xline(mean(theta_boot),'--','LineWidth',2)
xlabel('Bootstrap estimates of treatment effect')
ylabel('Frequency')
title('Bootstrap distribution of treatment effect')
hold off
