% L-BFGS

clear all
close all
clc

% Dimension of the problem
n = 10000

% Constant used in function f to minimize
alphaf = 10

% Tolerance
tol = 0.0000001

% Array with different memory sizes
M = [1, 3, 10, 15]

% Start plot
figure;

% Run L-BFGS algorithm with different memory sizes
for j = 1:length(M)
	% Memory size
	m = M(j)

	% Initial guess
	x = -1*ones(n,1);

	% Record of errors in log scale

	error = fOne(x, alphaf);
	errorLog = log10(error);
	record = [errorLog];

	% L-BFGS algorithm (Algorithm 7.5 in Nocedal)

	% Initialize memory (sk and yk in Nocedal)
	Sk = zeros(n,m);
	Yk = zeros(n,m);

	% Iteration
	k = 0

	% Stopping criteria: error
	while error > tol
		% To know which memory address to replace
		index = mod(k,m);
		% Compute Hk using equation 7.20 from Nocedal
		if k == 0
			H = eye(n);
		else
			if index == 0
				curvCond = Sk(:,m)'*Yk(:,m);
				% Check if curvature condition is positive
				if curvCond < .05
					disp('WARNING: Curvature condition close to zero, Sk(:,m)*Yk(:,m) < .05')
					disp(curvCond)
				end
				H = eye(n)*(curvCond/(Yk(:,m)'*Yk(:,m)));
			else
				curvCond = Sk(:,index)'*Yk(:,index);
				% Check if curvature condition is positive
				if curvCond < .05
					disp('WARNING: Curvature condition close to zero, Sk(:,index)*Yk(:,index) < .05')
					disp(curvCond)
				end
				H = eye(n)*(curvCond/(Yk(:,index)'*Yk(:,index)));
			end
		end
		% Compute pk using algorithm 7.4 from Nocedal
		P = BFGSR(x, alphaf, k, m, H, Sk, Yk);
		alpha = lineSearch(x, P, alphaf);
		oldx = x;
		x = oldx + (alpha*P);
		% Update memory
		Sk(:,index+1) = x - oldx;
		Yk(:,index+1) = fOneGrad(x, alphaf) - fOneGrad(oldx, alphaf);
		k = k + 1;
		error = fOne(x, alphaf);
		errorLog = log10(error);
		record = [record ; errorLog];
	end

	% Print final solution
	%disp('final solution x_sol:')
	%disp(x)

	% Print value of f at final solution
	disp('f(x_sol)')
	disp(error)
	% L-BFGS algorithm ends here

	% Plot convergence and save in PNG file
	axis = 0:length(record) - 1;
	plot(axis, record);
	hold on
	grid on;
	grid minor;
end

xlabel('iteration');
ylabel('log(f(x_k) - f(x^*))');
legend('L-BFGS, m = 1', 'L-BFGS, m = 3', 'L-BFGS, m = 10', 'L-BFGS, m = 15');
saveas(gcf,'L_BFGS.png');

% Function to optimize, f(x) = sum(alphaf(x_{2i} - x_{2i-1}^2)^2 + (1 - x_{2i-1})^2), 1 <= i <= n/2
function fX = fOne(x, alphaf)
	nf = length(x);
	x2 = x.^2;
	alphaOdd = zeros(nf,1);
	alphaOdd(1:2:end) = alphaf;
	alphaEven = ones(nf,1);
	alphaEven(2:2:end) = alphaf;
	xOdd = ones(nf,1);
	xOdd(2:2:end) = x2(1:2:end);
	fX = ((x2.*x2)'*alphaOdd)+(x2'*alphaEven)-(2*alphaEven'*(x.*xOdd)) + (nf/2);
end

% Gradient of function f to optimize
function [fXGradient] = fOneGrad(x, alphaf)
	ng = length(x);
	xAux = zeros(ng,1);
	xAux(2:2:end) = x(2:2:end);
	xx = zeros(ng,1);
	xx(1:2:end) = x(1:2:end).*xAux(2:2:end);
	x2Aux2 = x.^2;
	x2Aux = zeros(ng,1);
	x2Aux(2:2:end) = x2Aux2(1:2:end);	
	x3 = x.^3;
	x3(2:2:end) = 0;
	xAux2 = zeros(ng,1);
	xAux2(1:2:end) = x(1:2:end);
	Aux = ones(ng,1);
	Aux(2:2:end) = 0;
	fXGradient = (2*(xAux2 - Aux)) + (2*alphaf*(xAux - x2Aux + (2*(x3 - xx))));
end

% Phi(alpha) = f(x_k + alpha*P_k)
function phi = phiOne(x_k, P_k, alpha, alphaf)
	x = x_k + (alpha*P_k);
	phi = fOne(x, alphaf);
end

% Derivative of Phi(alpha) = dot product(Gradient of f(x_k + alpha*P_k), P_k)
function phiPrime = phiOneDeriv(x_k, P_k, alpha, alphaf)
	x = x_k + (alpha*P_k);
	fGrad = fOneGrad(x, alphaf);
	phiPrime = dot(P_k, fGrad);
end

% (Inexact) Line Search algorithm, using Wolfe conditions (Algorithm 3.5 in Nocedal)
function alpha = lineSearch(x_k, P_k, alphaf)
	c1 = 0.01;
	c2 = 0.9;
	alphaPrev = 0;
	%alpha = (alphaPrev + alphaMax)/2;
	alpha = 1;
	phiZero = phiOne(x_k, P_k, 0, alphaf);
	phiZeroPrime = phiOneDeriv(x_k, P_k, 0, alphaf);
	i = 1;

	while 1
		phiAlpha = phiOne(x_k, P_k, alpha, alphaf);
		phiAlphaPrev = phiOne(x_k, P_k, alphaPrev, alphaf);
		if (phiAlpha > (phiZero + (c1*alpha*phiZeroPrime))) || ((i > 1) && (phiAlpha >= phiAlphaPrev))
			alpha = zoomLS(alphaPrev, alpha, x_k, P_k, phiZero, phiZeroPrime, c1, c2, alphaf);
			break
		else
			phiAlphaPrime = phiOneDeriv(x_k, P_k, alpha, alphaf);
			if abs(phiAlphaPrime) <= (-c2*phiZeroPrime)
				break
			elseif phiAlphaPrime >= 0
				alpha = zoomLS(alpha, alphaPrev, x_k, P_k, phiZero, phiZeroPrime, c1, c2, alphaf);
				break
			else
				alphaPrev = alpha;
				alpha = 2*alphaPrev;
			end
		end
		i = i + 1;
	end
end

% Zoom algorithm (Algorithm 3.6 in Nocedal)
function alpha = zoomLS(alphaLo, alphaHi, x_k, P_k, phiZero, phiZeroPrime, c1, c2, alphaf)
	iter = 1;
	maxIter = 100;
	while 1
		alpha = (alphaLo + alphaHi)/2;
		phiAlpha = phiOne(x_k, P_k, alpha, alphaf);
		phiAlphaLo = phiOne(x_k, P_k, alphaLo, alphaf);
		if (phiAlpha > (phiZero + (c1*alpha*phiZeroPrime))) || (phiAlpha >= phiAlphaLo)
			alphaHi = alpha;
		else
			phiAlphaPrime = phiOneDeriv(x_k, P_k, alpha, alphaf);
			if abs(phiAlphaPrime) <= (-c2*phiZeroPrime)
				break
			elseif phiAlphaPrime*(alphaHi - alpha) >= 0
				alphaHi = alphaLo;
				alphaLo = alpha;
			else
				alphaLo = alpha;
			end
		end
		iter = iter + 1;
		if iter > maxIter
			break
		end 
	end
end

% L-BFGS two-loop recursion (Algorithm 7.4 in Nocedal)
function BFGSRecursion = BFGSR(x, alphaf, k, m, H, Sk, Yk)
	q = fOneGrad(x, alphaf);
	% Memory to save alpha_i in first loop
	ALPHAUX = [];
	counter = 0;
	if k > 0
		for i = k:-1:max(k-m+1,1)
			index = mod(i,m);
			if index == 0
				index = m;
			end
			alphaux = (1/(Yk(:,index)'*Sk(:,index)))*Sk(:,index)'*q;
			q = q - (alphaux*Yk(:,index));
			ALPHAUX = [ALPHAUX, alphaux];
			counter = counter + 1;
		end
	end
	r = H*q;
	if k > 0
		for i = max(k-m+1,1):k
			index = mod(i,m);
			if index == 0
				index = m;
			end
			beta = (1/(Yk(:,index)'*Sk(:,index)))*Yk(:,index)'*r;
			r = r + ((ALPHAUX(counter) - beta)*Sk(:,index));
			counter = counter - 1;
		end
	end
	BFGSRecursion = -r;
end
