% Steepest Descent algorithm

clear all
close all
clc

% Initial random guess
x = rand(1,2)

% Tolerance for stopping criteria
tol = 0.01

% Constant used in function f to minimize
c = 10

% Max alpha for line search
alphaMax = 1000

% Record of errors in log scale
record = [log10(fOne(x, c))];

% Main algorithm

% Gradient evaluation
fgrad = fOneGrad(x, c);
fgradNorm = norm(fgrad);

% Stopping criteria: gradient norm
while fgradNorm > tol
	P = -fgrad;
	alpha = lineSearch(c, x, P, alphaMax);
	x = x + (alpha*P);
	fgrad = fOneGrad(x, c);
	fgradNorm = norm(fgrad);
	record = [record ; log10(fOne(x, c))];
end

% Print final solution
disp('final solution:')
disp(x)

% Print value of f at final solution
disp('f(x*)')
disp(fOne(x, c))

% Plot convergence and save in PNG file
figure;
plot(record);
xlabel('iteration');
ylabel('log(f(x))');
grid on;
grid minor;
saveas(gcf,'steep_desc_ceq10.png')

% Function f to optimize
function fX = fOne(x, c)
	fX = (((c*x(1)) - 2)^4) + ((x(2)^2)*(((c*x(1)) - 2)^2)) + ((x(2) + 1)^2);
end

% Gradient of function f to optimize
function [fXGradient] = fOneGrad(x, c)
	fXGradient = [(4*c*(((c*x(1)) - 2)^3)) + (2*c*(x(2)^2)*((c*x(1)) - 2)), (2*x(2)*(((c*x(1)) - 2)^2)) + (2*(x(2) + 1))];
end

% Phi(alpha) = f(x_k + alpha*P_k)
function phi = phiOne(c, x_k, P_k, alpha)
	x = x_k + (alpha*P_k);
	phi = fOne(x, c);
end

% Derivative of Phi(alpha) = dot product(Gradient of f(x_k + alpha*P_k), P_k)
function phiPrime = phiOneDeriv(c, x_k, P_k, alpha)
	x = x_k + (alpha*P_k);
	fGrad = fOneGrad(x, c);
	phiPrime = dot(P_k, fGrad)
end

% (Inexact) Line Search algorithm, using Wolfe conditions (Algorithm 3.5 in Nocedal)
function alpha = lineSearch(c, x_k, P_k, alphaMax)
  % Values suggested in Nocedal for c1 and c2
	c1 = 0.1
	c2 = 0.9
	alphaPrev = 0
	alpha = (alphaPrev + alphaMax)/2
	phiZero = phiOne(c, x_k, P_k, 0);
	phiZeroPrime = phiOneDeriv(c, x_k, P_k, 0);
	i = 1

	while 1
		phiAlpha = phiOne(c, x_k, P_k, alpha);
		phiAlphaPrev = phiOne(c, x_k, P_k, alphaPrev);
    % Check Armijo condition
		if (phiAlpha > (phiZero + (c1*alpha*phiZeroPrime))) || ((i > 1) && (phiAlpha >= phiAlphaPrev))
			alpha = zoomLS(alphaPrev, alpha, c, x_k, P_k, phiZero, phiZeroPrime, c1, c2);
			break
		else
			phiAlphaPrime = phiOneDeriv(c, x_k, P_k, alpha);
      % Check curvature condition
			if abs(phiAlphaPrime) <= (-c2*phiZeroPrime)
				break
			elseif phiAlphaPrime >= 0
				alpha = zoomLS(alpha, alphaPrev, c, x_k, P_k, phiZero, phiZeroPrime, c1, c2);
				break
			else
				alphaPrev = alpha;
				alpha = (alphaPrev + alphaMax)/2;
			end
		end
		i = i + 1;
	end
	alpha
end

% Zoom algorithm (Algorithm 3.6 in Nocedal)
function alpha = zoomLS(alphaLo, alphaHi, c, x_k, P_k, phiZero, phiZeroPrime, c1, c2)
	while 1
		alpha = (alphaLo + alphaHi)/2;
		phiAlpha = phiOne(c, x_k, P_k, alpha);
		phiAlphaLo = phiOne(c, x_k, P_k, alphaLo);
		if (phiAlpha > (phiZero + (c1*alpha*phiZeroPrime))) || (phiAlpha >= phiAlphaLo)
			alphaHi = alpha;
		else
			phiAlphaPrime = phiOneDeriv(c, x_k, P_k, alpha);
			if abs(phiAlphaPrime) <= (-c2*phiZeroPrime)
				break
			elseif phiAlphaPrime*(alphaHi - alpha) >= 0
				alphaHi = alphaLo;
				alphaLo = alpha;
			else
				alphaLo = alpha;
			end
		end
	end
end
