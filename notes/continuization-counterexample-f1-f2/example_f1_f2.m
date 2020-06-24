% Examples F1 and F2 from M. Gaukler: "Analysis of Real-Time Control Systems using First-Order Continuization" (2020, under review)

% This file needs MATLAB/Simulink R2018b (should also work with newer versions. Will definitely not work with Octave.)

% Two examples where my interpretation of the paper "Periodically-Scheduled
% Controler Analysis Using Hybrid Systems Reachability and Continuization"
% by Bak and Johnson, RTSS 2015, doesn't work.

example = 1 % CHANGEME
% example = 2
if example == 1
    T = 1
    % d x_p / dt = A x_p + B x_c
    A = 0
    B = 1
    % x_c[k] = controller_update(x_p[k], x_c[k-1]) = K x_p[k] + L x_c[k-1]
    K = 0
    L = 2
    x_p_0 = 1
    x_c_0 = 2
else
    T = 1
    % d x_p / dt = A x_p + B x_c
    A = 0
    B = 3
    % x_c(kT) = controller_update(x_p(kT)) = K x_p + L x_c
    K = -1
    L = 0
    x_p_0 = 1
    x_c_0 = -1
end
% continuized initial state (per definition)
x_c_0_cont = K * x_p_0 + L * x_c_0

% error input - choose manually
omega_dot = 0

open('continuization_model')
sim('continuization_model')

% save Simulink logs as CSV
decimation = 100; % plot every n-th point
values=[];
names={'time'};
for i=1:logsout.numElements
    el = logsout.getElement(i);
    names{i+1} = el.Name;
    values(1,:) = el.Values.Time(1:decimation:end);
    values(i+1,:) = el.Values.Data(1:decimation:end);
end
T = array2table(values', 'VariableNames', names);
writetable(T, "plotdata-example-" + string(example) + ".csv")