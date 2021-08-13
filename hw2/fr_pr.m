% FR, FR with restart, and PR algorithms

clear all
close all
clc

% Dimension of the problem
n = 100

% Initial guess
x0 = 2*rand(n,1);
x = x0;

% Tolerance
tol = 0.001

% Max alpha for line search  
alphaMax = 10

% FR with restart algorithm (Algorithm 5.4 in Nocedal, with restart condition as in equation 5.52)

% vu value (suggested by Nocedal)
vu = 0.1;

% Record of errors in log scale
recordFRR = [log10(fOne(x))];

fgrad = fOneGrad(x);
fgradNorm = norm(fgrad);
P = -fgrad;
FRRIter = 0;
while fgradNorm > tol
	alpha = lineSearch(x, P, alphaMax);
	x = x + (alpha*P);
	fgradPrev = fgrad;
	fgrad = fOneGrad(x);
	fgradNorm = norm(fgrad);
	if abs(fgrad'*fgradPrev)/(fgrad'*fgrad) >= vu
		betaFRR = 0;
	else
		betaFRR = (fgrad'*fgrad)/(fgradPrev'*fgradPrev);
	end
	P = -fgrad + (betaFRR*P);
	recordFRR = [recordFRR ; log10(fOne(x))];
  FRRIter = FRRIter + 1
end

% Print final solution
disp('final solution x_sol for FR-r:')
disp(x)

% Print value of f at final solution
disp('f(x_sol) for FR-r:')
disp(fOne(x))
% FR with restart algorithm ends here

% PR algorithm (Algorithm 5.4 in Nocedal, with beta update as in equation 5.46)

% Initial guess
x = x0;

% Record of errors in log scale
recordPR = [log10(fOne(x))];

fgrad = fOneGrad(x);
fgradNorm = norm(fgrad);
P = -fgrad;
PRIter = 0;
while fgradNorm > tol
	alpha = lineSearch(x, P, alphaMax);
	x = x + (alpha*P);
	fgradPrev = fgrad;
	fgrad = fOneGrad(x);
	fgradNorm = norm(fgrad);
	betaPR = (fgrad'*(fgrad-fgradPrev))/(fgradPrev'*fgradPrev);
	P = -fgrad + (betaPR*P);
	recordPR = [recordPR ; log10(fOne(x))];
  PRIter = PRIter + 1
end

% Print final solution
disp('final solution x_sol for PR:')
disp(x)

% Print value of f at final solution
disp('f(x_sol) for PR:')
disp(fOne(x))
% PR algorithm ends here

% FR algorithm (Algorithm 5.4 in Nocedal)

% Initial guess
x = x0;

% Record of errors in log scale
recordFR = [log10(fOne(x))];

fgrad = fOneGrad(x);
fgradNorm = norm(fgrad);
P = -fgrad;
FRIter = 0;
while fgradNorm > tol
	alpha = lineSearch(x, P, alphaMax);
	x = x + (alpha*P);
	fgradPrev = fgrad;
	fgrad = fOneGrad(x);
	fgradNorm = norm(fgrad);	
	betaFR = (fgrad'*fgrad)/(fgradPrev'*fgradPrev);
	P = -fgrad + (betaFR*P);
	recordFR = [recordFR ; log10(fOne(x))];
	FRIter = FRIter + 1
	if FRIter > 2*max(FRRIter,PRIter)
		break
	end
end

% Print final solution
disp('final solution x_sol for FR:')
disp(x)

% Print value of f at final solution
disp('f(x_sol) for FR:')
disp(fOne(x))
% FR algorithm ends here

% Plot convergence and save in PNG file
axis1 = 0:length(recordFR) - 1;
axis2 = 0:length(recordFRR) - 1;
axis3 = 0:length(recordPR) - 1;
figure;
plot(axis1, recordFR, axis2, recordFRR, '--', axis3, recordPR, '-.');
xlabel('iteration');
ylabel('log(f(x))');
legend('FR algorithm', 'FR algorithm with restart', 'PR algorithm');
grid on;
grid minor;
saveas(gcf,'FR_PR.png');

% Function f to optimize
function fX = fOne(x)
	nf = length(x);
	x2 = x.^2;
	x2(nf) = 0;
	oneVec = ones(nf,1);
	oneVec(nf) = 0;
	x2n0 = [x(2:nf);0];
	fX = (100*(x2'*x2)) + (nf-1) + (101*(x'*x)) - (100*(x(1)^2)) - (x(nf)^2) - (2*(oneVec'*x)) - (200*(x2'*x2n0));
end

% Gradient of function f to optimize
function [fXGradient] = fOneGrad(x)
	ng = length(x);	
	x2Aux = x.^2;
	x2 = [0;x2Aux(1:ng-1)];
	oneVec = ones(ng,1);
	oneVec(ng) = 0;
	x2n0 = [x(2:ng);0];
	x3 = x.^3;
	x3(ng) = 0;
	x1nminus10 = [x(1:ng-1);0];
	x02n = [0;x(2:ng)];
	fXGradient = (200*(x02n-x2)) + (400*(x3-(x.*x2n0))) + (2*(x1nminus10-oneVec));
end

% Phi(alpha) = f(x_k + alpha*P_k)
function phi = phiOne(x_k, P_k, alpha)
	x = x_k + (alpha*P_k);
	phi = fOne(x);
end

% Derivative of Phi(alpha) = dot product(Gradient of f(x_k + alpha*P_k), P_k)
function phiPrime = phiOneDeriv(x_k, P_k, alpha)
	x = x_k + (alpha*P_k);
	fGrad = fOneGrad(x);
	phiPrime = dot(P_k, fGrad);
end

%(Inexact) Line Search algorithm, using Wolfe conditions (Algorithm 3.5 in Nocedal)
function alpha = lineSearch(x_k, P_k, alphaMax)
	c1 = 0.01;
	c2 = 0.45;
	alphaPrev = 0;
	alpha = (alphaPrev + alphaMax)/2;
	phiZero = phiOne(x_k, P_k, 0);
	phiZeroPrime = phiOneDeriv(x_k, P_k, 0);
	i = 1;

	while 1
		phiAlpha = phiOne(x_k, P_k, alpha);
		phiAlphaPrev = phiOne(x_k, P_k, alphaPrev);
		if (phiAlpha > (phiZero + (c1*alpha*phiZeroPrime))) || ((i > 1) && (phiAlpha >= phiAlphaPrev))
			alpha = zoomLS(alphaPrev, alpha, x_k, P_k, phiZero, phiZeroPrime, c1, c2);
			break
		else
			phiAlphaPrime = phiOneDeriv(x_k, P_k, alpha);
			if abs(phiAlphaPrime) <= (-c2*phiZeroPrime)
				break
			elseif phiAlphaPrime >= 0
				alpha = zoomLS(alpha, alphaPrev, x_k, P_k, phiZero, phiZeroPrime, c1, c2);
				break
			else
				alphaPrev = alpha;
				alpha = (alphaPrev + alphaMax)/2;
			end
		end
		i = i + 1;
	end
	alpha;
end

%Zoom algorithm (Algorithm 3.6 in Nocedal)
function alpha = zoomLS(alphaLo, alphaHi, x_k, P_k, phiZero, phiZeroPrime, c1, c2)
	while 1
		alpha = (alphaLo + alphaHi)/2;
		phiAlpha = phiOne(x_k, P_k, alpha);
		phiAlphaLo = phiOne(x_k, P_k, alphaLo);
		if (phiAlpha > (phiZero + (c1*alpha*phiZeroPrime))) || (phiAlpha >= phiAlphaLo)
			alphaHi = alpha;
		else
			phiAlphaPrime = phiOneDeriv(x_k, P_k, alpha);
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
