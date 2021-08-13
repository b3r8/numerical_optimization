% Trust region methods

clear all
close all
clc

% Dimension of the problem
n = 200

% Create a random, symmetric, positive definite matrix A with specific eigenvalues

[Q,R] = qr(rand(n,n));

eigenValues = rand(n,1)*n;

D = diag(eigenValues);
A = Q'*D*Q;

% Initial guess
x0 = 1*rand(n,1);
x = x0;

% Tolerance
tol = 0.00000001

% Record of errors in log scale

aux = 1 + (x'*A*x);
error = log(aux);
errorLog = log10(error);
recordC = [errorLog];

% Overall bound on the step lengths           
maxDelta = 10

% Criteria parameter to update solution x           
eta = 1/8

% TR with Cauchy point (Algorithm 4.1 in Nocedal with Cauchy point for p_k)

delta = maxDelta/2

% Stopping criteria: error           
while error > tol
  	% Cauchy point calculation (Equations 4.11 and 4.12 in Nocedal)
	grad = (2/aux)*A*x;
	discr = grad'*A*grad;
	if discr > 0
		alpha = min(delta/norm(grad),(grad'*grad)/discr);
	else
		alpha = delta/norm(grad);
	end
	p = -alpha*grad;

  	% Ratio ro (Equation 4.4 in Nocedal)                                  
	ro = (log(aux) - log(1 + ((x+p)'*A*(x+p))))/(-(grad'*p) - (.5*p'*A*p));

	if ro < .25
		delta = delta*.25;
	elseif (ro > .75) && (norm(p) == delta)
		delta = min(2*delta, maxDelta);
	else
		delta = delta;
	end

	if ro > eta
		x = x + p;
	end

	aux = 1 + (x'*A*x);
	error = log(aux);
	errorLog = log10(error);
	recordC = [recordC ; errorLog];
end

% Print final solution
%disp('final solution x_sol for TR with Cauchy point:')
%disp(x)

% Print value of f at final solution
disp('f(x_sol) for TR with Cauchy point:')
disp(log(aux))
% TR Algorithm with Cauchy point ends here

% TR with dogleg method (Algorithm 4.1 in Nocedal with dogleg method for p_k)

% Initial guess
x = x0;
             
delta = maxDelta/2

% Record of errors in log scale

aux = 1 + (x'*A*x);
error = log(aux);           
errorLog = log10(error);
recordD = [errorLog];

% Stopping criteria: error                      
while error > tol
  	% Dogleg method (Equations 4.15 and 4.16 in Nocedal)
	grad = (2/aux)*A*x;
	discr = grad'*A*grad;
	pU = -grad*(grad'*grad)/discr;
	pB = A\(-grad);
	if norm(pB) <= delta
		p = pB;
	elseif norm(pU) >= delta
		p = -grad*delta/norm(grad);
	else
		a = norm(pB - pU)^2;
		b = 2*(pB - pU)'*pU;
		c = (norm(pU)^2) - (delta^2);
		tau = 1 + ((((b^2) - (4*a*c))^.5) - b)/(2*a);
		p = pU + (tau - 1)*(pB - pU);
	end
	
  	% Ratio ro (Equation 4.4 in Nocedal) 
	ro = (log(aux) - log(1 + ((x+p)'*A*(x+p))))/(-(grad'*p) - (.5*p'*A*p));

	if ro < .25
		delta = delta*.25;
	elseif (ro > .75) && (norm(p) == delta)
		delta = min(2*delta, maxDelta);
	else
		delta = delta;
	end

	if ro > eta
		x = x + p;
	end

  	aux = 1 + (x'*A*x);
	error = log(aux);
	errorLog = log10(error);
	recordD = [recordD ; errorLog];
end

% Print final solution
%disp('final solution x_sol for TR with dogleg method:')
%disp(x)

% Print value of f at final solution
disp('f(x_sol) for TR with dogleg method:')
disp(log(aux))
% TR Algorithm with dogleg method ends here

% Plot convergence and save in PNG file
axis1 = 0:length(recordC) - 1;
axis2 = 0:length(recordD) - 1;
figure;
plot(axis1, recordC, axis2, recordD, '--');
xlabel('iteration');
ylabel('log(f(x_k) - f(x^*))');
legend('TR with Cauchy Point Method','TR with Dogleg Method','Location','southwest');
grid on;
grid minor;
saveas(gcf,'TR.png');
