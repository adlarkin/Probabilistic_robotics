function [mu,Sig] = meas_up_EKF(X,mu,Sig,m,sig)

% This function performs the measurement update corresponding to a 
% specific landmark m. See lines 9-20 of Table 7.2 in Probabilistic
% Robotics by Thrun, et al.

x = X(1);           % true states used to create measurements
y = X(2);
th = X(3);

mu_x = mu(1);       % estimated states to be updated by measurement
mu_y = mu(2);
mu_th = mu(3);

mx = m(1);          % known land mark location and signature
my = m(2);

sig_r = sig(1);     % s.d. of noise levels on measurements
sig_ph = sig(2);

% Measurements: truth + noise
range = sqrt((mx-x).^2 + (my-y).^2) + sig_r*randn;
bearing = atan2(my-y,mx-x) - th + sig_ph*randn;
z = [range; bearing];

% Calculate predicted measurement based on state estimate
q = (mx-mu_x)^2 + (my-mu_y)^2;
zhat = zeros(2,1);
zhat(1) = sqrt(q);
zhat(2) = atan2((my-mu_y),(mx-mu_x)) - mu_th;

% Jacobian of measurement function wrt state
H = zeros(2,3);
H(1,1) = -(mx-mu_x)/sqrt(q); 
H(1,2) = -(my-mu_y)/sqrt(q); 
H(2,1) = (my-mu_y)/q; 
H(2,2) = -(mx-mu_x)/q;
H(2,3) = -1;

% Total uncertainty in predicted measurement
Q = diag([sig_r^2, sig_ph^2]);
S = H*Sig*H' + Q;

% Kalman gain
K = (Sig*H')/S;

% Measurment update
mu = mu + K*(z-zhat);
Sig = (eye(3) - K*H)*Sig;

end

