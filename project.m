% TASK 3.1
treated = readmatrix('nswre74_treated.txt');
control = readmatrix('psid_controls.txt');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TASK 3.2
data = [treated; control];

%   data(:,4) - "Take all rows from column 4"
%   data(:,4) == 1 - "Is column 4 equal to 1?"; Result is a logical vector,
%   where TRUE = 1 and FALSE = 0
%   data(logical_vector,:) - "Keep rows where logical vector = TRUE"
%   data = data(data(:,4)==1,:) - "(1) Look at column 4 (black), (2) Find rows where
%   value = 1, (3) Keep only those rows"
data = data(data(:,4)==1,:);

treatment = data(:,1);
age = data(:,2);
education = data(:,3);
married = data(:,6);
nodegree = data(:,7);
RE75 = data(:,9);
RE78 = data(:,10);

% Create logical indices for treated and control groups
% treated_group contains TRUE for participants in the program
% control_group contains TRUE for non-participants
treated_group = treatment == 1;
control_group = treatment == 0;

%   variables matrix
X = [age education married nodegree RE75 RE78];
%   variables names
varNames = {'Age','Education','Married','Nodegree','RE75','RE78'};


% Compute summary statistics for treated individuals
mean_treated = mean(X(treated_group,:));
median_treated = median(X(treated_group,:));
std_treated = std(X(treated_group,:));

% Compute summary statistics for control individuals
mean_control = mean(X(control_group,:));
median_control = median(X(control_group,:));
std_control = std(X(control_group,:));

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

% TASK 4.2.1

% Create X matrix
X = [ones(length(age),1), ...
     age, ...
     education, ...
     married, ...
     nodegree, ...
     RE75];

% Split groups
X1 = X(treatment==1,:);
X0 = X(treatment==0,:);

y1 = RE78(treatment==1);
y0 = RE78(treatment==0);

% OLS coefficients
b1 = (X1' * X1) \ (X1' * y1);
b0 = (X0' * X0) \ (X0' * y0);

% Predicted outcomes
mu1 = X * b1;
mu0 = X * b0;

% Treatment effect
theta = mean(mu1(treatment==1) - mu0(treatment==1));

% Display
disp(theta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TASK 4.2.2

% Define variables after filtering black == 1
treatment = data(:,1);
age       = data(:,2);
education = data(:,3);
married   = data(:,6);
nodegree  = data(:,7);
RE75      = data(:,9);
RE78      = data(:,10);

% Construct X matrix
% Columns: constant, age, education, married, nodegree, RE75
X = [ones(length(treatment),1), age, education, married, nodegree, RE75];

% Define outcome variable
y = RE78;

% Split treated and untreated observations
X1 = X(treatment == 1,:);
X0 = X(treatment == 0,:);

y1 = y(treatment == 1);
y0 = y(treatment == 0);

% Estimate OLS coefficients separately
b1 = (X1' * X1) \ (X1' * y1);
b0 = (X0' * X0) \ (X0' * y0);

% Predicted outcomes for everyone
mu1 = X * b1;
mu0 = X * b0;

% Treatment effect for program participants only
theta_hat = mean(mu1(treatment == 1) - mu0(treatment == 1));

% Display result
disp(theta_hat)
