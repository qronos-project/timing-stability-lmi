% Proof and example for the "Extreme Quadratic Lyapunov Function" remark in:
% Gaukler et al. (2019/2020): Stability Analysis of Multivariable Digital Control Systems with Uncertain Timing. Submitted for publication.
%
% This file is concerned with the statement:
% In the general case, there is no lower $P$-norm than the one guaranteed by the Theorem "Extreme Quadratic Lyapunov Function". Especially, it is not generally possible to find a $P$ such that $\|A\|_P=\rho(\{A\})$ holds exactly.

% This file requires MATLAB with the symbolic and LTI toolbox. Tested on version 2018a.
% Unfortunately it does not run on Octave due to missing collect() function in the symbolic toolbox.

% Here, we consider the example
clear all
syms rho positive
A = [ rho 1; 0 rho ]

% with general P>0:

% Parameterize P via Cholesky decomposition: PD matrix is uniquely factored into P_half P_half.' ,
# where P_half is lower triangular with positive diagonal
syms a c positive
syms b real
P_half = [a 0; b c]
P = P_half * P_half.'
assert(all(all(isAlways(P==P.'))))
M = P_half.' * A * (P_half.'^(-1))


%% Proof of Theorem

syms sigmaSquare real
determinant = collect(det(M.'*M - sigmaSquare*eye(2)), sigmaSquare);
assert(isAlways(determinant == sigmaSquare^2 - (a^2/c^2 + 2*rho^2) * sigmaSquare + rho^4))
% The above is the key intermediate result. The actual result directly follows, but is not machine-checked here.


%% Explicit determination of singular value

% P-norm is max. singular value of von P_half_T A P_half_T^(-1) 
sv = svd(M)
sv_1 = sv(1) % one of the singular values

e = (sv_1^2 - rho^2)
expand(simplify(e))
% At this point, you could add a check to prove that sv_1 or sv_2 is >0.

% Check that the above singular value is correct
assert(isAlways(det((M).' * (M) - sv(1)^2 * eye(2)) == 0))



%% Numeric Example

format long
rho_n = 0.5
A_n = [ rho_n 1; 0 rho_n ]
% X = dlyap(A,Q) solves  A*X*A' - X + Q = 0
% We need a solution of A'PA-P=-Q (transposed A as compared to dlyap)
P_n = dlyap(A_n.' / rho_n * 0.999, eye(2))
% R = chol(A)    solves   R'*R = A     (Cholesky)
% The definition of Cholesky used in the publication is different from MATLAB's: is P_half * P_half.' = P.
P_half = chol(P_n).'
% almost-diagonal transformed system M_n
M_n = P_half.'*A_n*(P_half.'^(-1))
% P-norm
norm(M_n, 2)

% initial response (response of each state to each initial state component) showing the overshoot in the original system
impulse(ss(A_n, eye(2), eye(2), [], 1))

