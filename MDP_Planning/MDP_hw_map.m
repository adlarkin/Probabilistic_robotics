% MDP_hw_map.m
%
% Create a map for MDP path planning homework

clear all;

N = 100;
Np = 100 + 2;

map = zeros(Np,Np);        % map dimension

% Initialize walls and obstacle maps as empty
walls = zeros(Np,Np);
obs1 = zeros(Np,Np);
obs2 = zeros(Np,Np);
obs3 = zeros(Np,Np);
goal = zeros(Np,Np);

% Create exterior walls
walls(2,2:N) = 1;
walls(2:N+1,2) = 1;
walls(N+1,2:N+1) = 1;
walls(2:N+1,N+1) = 1;

% Create single obstacle
obs1(20:40,30:80) = 1;
obs1(10:20,60:65) = 1;

% Another obstacle
obs2(45:65,10:45) = 1;

% Another obstacle
obs3(43:92,75:85) = 1;
obs3(70:80,50:75) = 1;

% The goal states
goal(75:80,96:98) = 1;

% Put walls and obstacles into map
map = walls + obs1 + obs2 + obs3 + goal;

% Plot map
% Sort through the cells to determine the x-y locations of occupied cells
[Mm,Nm] = size(map);
xm = [];
ym = [];
    for i = 1:Mm
        for j = 1:Nm
            if map(i,j)
                xm = [xm i];
                ym = [ym j];
            end
        end
    end

figure(1); clf;
plot(xm,ym,'.');
axis([0 Np+1 0 Np+1]);
axis('square'); 

save('mdp_data.mat')


