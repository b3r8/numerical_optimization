% Conjugate Gradient algorithm

clear all
close all
clc

% Dimension of the problem
n = 1000

% Create a pair of random, symmetric, positive definite matrices A1, A2 with specific eigenvalues

[Q,R] = qr(rand(n,n));

% Option 1 (uniformly distributed eigenvalues)
eigenValues = linspace(10,n,n);

% Option 2 (clustered eigenvalues)
% 65 percent of eigenvalues clustered in the [9, 11] segment
aux1 = linspace(9,11,6.5*n/10);
% 35 percent of eigenvalues clustered in the [999, 1001] segment
aux2 = linspace(999,1001,3.5*n/10);
eigenValuesClust = [aux1, aux2];

D1 = diag(eigenValues);
D2 = diag(eigenValuesClust);
% Matrix A1 with uniformly distributed eigenvalues
A1 = Q'*D1*Q;
% Matrix A2 with clustered eigenvalues
A2 = Q'*D2*Q;

% Sanity check
disp('Uniformly distributed eigenvalues:')
disp(sort(eig(A1)));
disp('Clustered eigenvalues:')
disp(sort(eig(A2)));

% Create random vector b
b = rand(n,1);

% Tolerance
tol = 0.0001

% Optimum point for A1
xmin = A1\b;
% Value of quadratic function at optimum x (for A1)
minf = -.5*(b'*xmin);

% Optimum point for A2
xminClust = A2\b;
% Value of quadratic function at optimum x (for A2)
minfClust = -.5*(b'*xminClust);

% Initial guess for A1
x = 2*rand(n,1);             
                 
% Record of errors in log scale (error = ||x - xmin||^2_A)                 
record = [log10(2*((.5*(x'*A1*x)) - (b'*x) - minf))];
recordClust = [log10(2*((.5*(x'*A2*x)) - (b'*x) - minfClust))];

% CG (Algorithm 5.2 in Nocedal)
                     
% CG for A1 (from here)
prev_r = zeros(n,1);
r = A1*x - b;
p = -r;

% Stopping criteria: residual                   
while norm(r) > tol
	alpha = (r'*r)/(p'*A1*p);
	x = x + (alpha*p);
	prev_r = r;
	r = r + (alpha*A1*p);
	beta = (r'*r)/(prev_r'*prev_r);
	p = -r + (beta*p);
	record = [record ; log10(2*((.5*(x'*A1*x)) - (b'*x) - minf))];
end

% Print final solution
%disp('final solution x_sol for A1:')
%disp(x)

% Print value of f at final solution
disp('f(x_sol) for A1:')
disp((.5*(x'*A1*x)) - (b'*x))
% CG for A1 (to here)


% CG for A2 (from here)
     
% Initial guess for A2
x = 2*rand(n,1);                  
     
prev_r = zeros(n,1);
r = A2*x - b;
p = -r;

% Stopping criteria: residual                        
while norm(r) > tol
	alpha = (r'*r)/(p'*A2*p);
	x = x + (alpha*p);
	prev_r = r;
	r = r + (alpha*A2*p);
	beta = (r'*r)/(prev_r'*prev_r);
	p = -r + (beta*p);
	recordClust = [recordClust ; log10(2*((.5*(x'*A2*x)) - (b'*x) - minfClust))];
end

% Print final solution
%disp('final solution x_sol for A2:')
%disp(x)

% Print value of f at final solution
disp('f(x_sol) for A2:')
disp((.5*(x'*A2*x)) - (b'*x))
% CG for A2 (to here)

% Error upper bound (Equation 5.36 in Nocedal)
     
% Approximate kappa
kappa = 1000/10;
slope = 2*log10(((kappa^.5)-1)/((kappa^.5)+1));
k = 1:length(record);

% Upper bound for A1
upperBound1 = log10(4) + record(1) + (slope*k);

% Upper bound for A2
upperBound2 = log10(4) + recordClust(1) + (slope*k);

%Plot convergence and save in PNG file
axis1 = 0:length(record) - 1;
axis2 = 0:length(recordClust) - 1;
figure;
title('CG, Ax=b, n=1000');
plot(axis1, record, 'b', axis2, recordClust, 'r', k-1, upperBound1, 'c--', k-1, upperBound2, 'm--');
xlabel('iteration');
ylabel('log(\bf{||x-x^*||}^2_A)');
legend('uniformly distributed eigenvalues (A1)','clustered eigenvalues  (A2)', 'upper bound (A1)', 'upper bound (A2)');
grid on;
grid minor;  
saveas(gcf,'CG.png');
