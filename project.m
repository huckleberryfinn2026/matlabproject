treated = readmatrix('nswre74_treated.txt');
control = readmatrix('psid_controls.txt');

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

treated_group = treatment == 1;
control_group = treatment == 0;

%   variables matrix
X = [age education married nodegree RE75 RE78];
%   variables names
varNames = {'Age','Education','Married','Nodegree','RE75','RE78'};

mean_treated = mean(X(treated_group,:));
median_treated = median(X(treated_group,:));
std_treated = std(X(treated_group,:));

mean_control = mean(X(control_group,:));
median_control = median(X(control_group,:));
std_control = std(X(control_group,:));

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