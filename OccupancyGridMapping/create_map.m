% create_map.m
% Create a simple occupancy grid map
% Map dimension is N by N

clear all;

N = 100;

map = zeros(N,N);        % map dimension

% Initialize walls and obstacle maps as empty
walls = zeros(N,N);
obs1 = zeros(N,N);
obs2 = zeros(N,N);
obs3 = zeros(N,N);

% Create exterior walls
walls(1,1:10) = 1;
walls(1,26:N) = 1;
walls(:,1) = 1;
walls(N,:) = 1;
walls(1:70,N) = 1;
walls(85:100,N) = 1;

% Create single obstacle
obs1(20:40,70) = 1;
obs1(40,70:80) = 1;
obs1(40:-1:20,80) = 1;
obs1(20,80:-1:70) = 1;

% Another obstacle
obs2(50,1:8) = 1;

% A third obstacle
obs3(55:65,20) = 1;
obs3(55:65,35) = 1;
obs3(65,20:35) = 1;

% Put walls and obstacles into map
map = walls + obs1 + obs2 + obs3;

% Create vector of robot poses (x,y,theta)
X1 = [25*ones(1,61); 5:1:65; pi/2*ones(1,61)];
X2 = [25*ones(1,10); 65*ones(1,10); (pi/2:pi/2/9:pi)];
X3 = [25:-1:15; 65*ones(1,11); pi*ones(1,11)];
X4 = [15*ones(1,10); 65*ones(1,10); (pi:-pi/2/9:pi/2)];
X5 = [15*ones(1,26); 65:1:90; pi/2*ones(1,26)];
X6 = [15*ones(1,10); 90*ones(1,10); (pi/2:-pi/2/9:0)];
X7 = [15:1:60; 90*ones(1,46); 0*ones(1,46)];
X8 = [60*ones(1,5); 90*ones(1,5); (0:-pi/4/4:-pi/4)];
X9 = [60:1:90; 90:-1:60; -pi/4*ones(1,31)];
X10 = [90*ones(1,10); 60*ones(1,10); (-pi/4:-pi/2/9:-3*pi/4)];
X11 = [90:-1:75; 60:-1:45; -3*pi/4*ones(1,16)];
X12 = [75*ones(1,5); 45*ones(1,5); (-3*pi/4:pi/4/4:-pi/2)];
X13 = [75*ones(1,36); 45:-1:10; -pi/2*ones(1,36)];
X14 = [75*ones(1,10); 10*ones(1,10); (-pi/2:-pi/2/9:-pi)];
X15 = [75:-1:20; 10*ones(1,56); pi*ones(1,56)];
X16 = [20*ones(1,37); 19*ones(1,37); (-pi:pi/4/36:-3*pi/4)];

% Create a single pose vector from multiple components
X = [X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16];

[~,Nsteps] = size(X);

% Laser pointing angles
% thk = pi*[-1/2 -1/3 -1/6 0 1/6 1/3 1/2];
thk = pi*[-1/2 -3/8 -1/4 -1/8 0 1/8 1/4 3/8 1/2];
% thk = -pi/2;
K = length(thk);

% Preallocate measurement vectors
% 2 measurements (range, bearing), K laser beams, Nsteps different poses
zw = zeros(2,K,Nsteps);
zo1 = zeros(2,K,Nsteps);
zo2 = zeros(2,K,Nsteps);
zo3 = zeros(2,K,Nsteps);
z = zeros(2,K,Nsteps);

% Create measurements based on pose, sensor pointing angle and 
% cells occupied by walls, obstacles
for n = 1:Nsteps
    for k = 1:K
       zw(:,k,n) = create_meas(walls,X(:,n),thk(k));
       zo1(:,k,n) = create_meas(obs1,X(:,n),thk(k));
       zo2(:,k,n) = create_meas(obs2,X(:,n),thk(k));
       zo3(:,k,n) = create_meas(obs3,X(:,n),thk(k));
    end
end

% Sort through walls and obstacles to find shortest range return
% Eliminates "see through" walls and obstacles
for n = 1:Nsteps
    for k = 1:K
        zmat = [zw(:,k,n) zo1(:,k,n) zo2(:,k,n) zo3(:,k,n)];
        [zmin,iz] = min(zmat(1,:));
        z(1,k,n) = zmin;
        z(2,k,n) = zmat(2,iz);        
    end
end

% Plot map
% Sort through the cells to determine the x-y locations of occupied cells
[M,N] = size(map);
xm = [];
ym = [];
    for i = 1:M
        for j = 1:N
            if map(i,j)
                xm = [xm i];
                ym = [ym j];
            end
        end
    end

figure(1); clf;
plot(xm,ym,'.');
axis([0 100 0 100]);
axis('square'); 
hold on;
pause;

% Plot the sensed cells (hits) based on robot pose and sensor measurements 
for n = 1:Nsteps
    xhit = X(1,n) + z(1,:,n).*cos(X(3,n)+thk);
    yhit = X(2,n) + z(1,:,n).*sin(X(3,n)+thk);
    plot(xhit,yhit,'ro');
    plot(X(1,n),X(2,n),'b+');
    pause(0.2);
end 

hold off;

save 'new_meas_data.mat' X z thk

