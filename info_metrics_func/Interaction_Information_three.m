function I_XYZ = Interaction_Information_three(X, Y, Z)
% % Mutual_Information_weighted %
%PURPOSE:   Compute mutual information I(X;Y) from two logical vectors 
%AUTHORS:   Jae Hyung Woo 04/10/2025; adapted from codes by Ethan Trepka 10/05/2020
%
%INPUT ARGUMENTS
%   Y:  vector of conditioned variable, each uq value is an event; e.g., 1 = stay, 0 = switch
%   X:  vector of conditioning variable, each value is an event; e.g., 1 = win, 0 = lose 
%   metric_name: string, name of conditional entropy metric
%   decomp_map: map from "events" in x to strings to label to decompositions that corresponds with the label, 
%               e.g., {1,0 -> "win","lose"}
% 
%OUTPUT ARGUMENTS
%   output: 
%       (metric_name) : equivalent to I(X;Y)
%       (u + metric_name + decomp_x) : (unweighted) M.I. for specific instance of X, equal to I(Y; X = x)
%       (metric_name + decomp_x) : weighted version decomp, such that decompositions sum up to whole metric;
%                                  equivalent to P(x) * I(Y; X = x)

    assert(length(X)==length(Y));
    assert(length(Z)==length(Y));
    if sum(any(isnan([X, Y, Z]),2))>0
        nanIDX = isnan(Y)|isnan(X)|isnan(Z); % exclude NaN entries if any
        X(nanIDX) = [];
        Y(nanIDX) = [];
        Z(nanIDX) = [];
    end    
    y_unique = unique(Y);
    x_unique = unique(X);
    z_unique = unique(Z);
    
    %% sum through all cases 
    H_X = Shannon_Entropy(X);
    H_Y = Shannon_Entropy(Y);
    H_Z = Shannon_Entropy(Z);

    H_XY = joint_entropy_two(X, Y);
    H_YZ = joint_entropy_two(Y, Z);
    H_ZX = joint_entropy_two(Z, X);

    % loop through each instance of X
    sum_X = nan(1, length(x_unique));
    for x_num = 1:length(x_unique)
        x_signal = (X==x_unique(x_num));
        
        % loop through all instances of Y
        sum_Y = nan(1, length(y_unique));
        for y_num = 1:length(y_unique)
            y_signal = (Y==y_unique(y_num));

            % loop through all instances of Z
            sum_Z = nan(1, length(z_unique));
            for z_num = 1:length(z_unique)
                z_signal = (Z==z_unique(z_num));

                prob_xyz = mean(x_signal & y_signal & z_signal);
                
                sum_Z(z_num) = - prob_xyz * log2(prob_xyz);
            end
            sum_Y(y_num) = nan_sum(sum_Z);
        end        
        sum_X(x_num) = nan_sum(sum_Y);
    end
    H_XYZ = nan_sum(sum_X);

    I_XYZ = (H_X + H_Y + H_Z) - (H_XY + H_YZ + H_ZX) + H_XYZ;
end

function H_XY = joint_entropy_two(X, Y)
    y_unique = unique(Y);
    x_unique = unique(X);

    % loop through each instance of X; e.g., {win, lose}
    joint_ent_X = nan(1, length(x_unique));
    for x_num = 1:length(x_unique)
        x_signal = (X==x_unique(x_num));

        % loop through all instances of Y in {stay, switch}
        joint_ent_y = nan(1, length(y_unique));
        for y_num = 1:length(y_unique)
            y_signal = (Y==y_unique(y_num));

            prob_x_and_y = mean(y_signal & x_signal);
            joint_ent_y(y_num) = - prob_x_and_y * log2(prob_x_and_y);
        end        
        joint_ent_X(x_num) = nan_sum(joint_ent_y);
    end
    H_XY = nan_sum(joint_ent_X);

end

function Sum = nan_sum(Array)
    % avoid returning zero when all elements are NaN
    if all(isnan(Array))
        Sum = NaN;
    else
        Sum = sum(Array, 'omitnan');    
    end
end