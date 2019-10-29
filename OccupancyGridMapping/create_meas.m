function [Z]= create_meas(m,X,thk)
% Function to create measurment corresponding to the robot pose, sensor
% pointing direction, and occupied cells in given map. This function is
% intended to produce what a range-finder type sensor would generate
%
% Map corresponds to map of walls, or map of one obstacle in map

    % Size of map
    [M,N] = size(m);
    
    % Robot pose
    x = X(1);
    y = X(2);
    th = X(3);
    
    % Initialize variables
    imin = M+1;
    rmin = M+N;
    phmin = pi;

    % Sort through all map cells
    % Pretty sure this could be done faster by operating on map matrix and using
    % min function...
    for i = 1:M
        for j = 1:N
            if m(i,j)   % cell is occupied
                r = sqrt((i-x).^2 + (j-y).^2);  % range
                ph = atan2(j-y,i-x) - th;       % bearing
                if (ph > pi)                    % wrap bearing angles into (-pi, pi]
                    ph = ph - 2*pi;
                elseif (ph <= -pi)
                    ph = ph + 2*pi;
                end
                % calculate pointing error magnitude: error between laser pointing
                % angle and bearing to center of grid cell
                ph_err = abs(ph-thk);
                % if pointing error is within bounds of current cell and
                % current cell is closest cell to vehicle that satisfies
                % pointing error, then save it off
                if ((ph_err*r <= 0.707) && (r < rmin))
                    rmin = r;
                    phmin = ph;
                    imin = i;
                end
            end
        end
    end
    
    % if no cells detected, return Nan's for sensor reading
    if imin == M+1
        rmin = nan;
        phmin = nan;
    end
    
    % return the closest cell that satisfies the pointing error constraint
    Z = [rmin; phmin];
    
end

