function output = Conditional_Mutual_Information(X, Y, Z, metric_name, decomp_map)
% % Tranfer_Entropy %
%PURPOSE:   Compute conditional mutual information I(X;Y|Z)
%AUTHORS:   Jae Hyung Woo 04/10/2025
%
%INPUT ARGUMENTS
%   Y:  column vector of response variable, each uq value is an event; e.g., 1 = stay, 0 = switch
%   X:  column vector of predictor variable, each value is an event; e.g., 1 = win, 0 = lose 
%   Z:  column vector of the third, conditioning variable in the equation I(X;Y|Z)
%   metric_name: string, name of conditional entropy metric
%   decomp_map: map from "events" in x to strings to label to decompositions that corresponds with the label, 
%               e.g., {1,0 -> "win","lose"}
%   cond_label: string, optional input for specifying the label of conditioning variable 
%OUTPUT ARGUMENTS
%   output: 
%       (metric_name) : equivalent to I(X;Y)
%       (metric_name + decomp_x) : weighted version decomp, such that decompositions sum up to whole metric;
%                                  equivalent to P(x) * I(Y; X = x)
    assert(length(X)==length(Y)&&length(X)==length(Z));
    assert(iscolumn(X), "input should be column vector");
   
    if sum(any(isnan([X, Y, Z]),2))>0
        nanIDX = isnan(X)|isnan(Y)|isnan(Z); 
        X(nanIDX) = [];
        Y(nanIDX) = [];
        Z(nanIDX) = []; % exclude NaN entries; drops the first trial
    end        
    x_unique = unique(X);
    y_unique = unique(Y);
    z_unique = unique(Z); 

    if ~exist('decomp_map', 'var')||isempty(decomp_map)
        decompose_flag = false;
    else
        decompose_flag = true;
    end
    
    if decompose_flag
        output.(metric_name) = NaN;
        decomp_vals = values(decomp_map);
        for i = 1:decomp_map.Count
            output.(metric_name+"_"+decomp_vals(i)) = NaN;
        end
    end    
    %% sum through all cases 
    all_X_sum = nan(1, length(x_unique)); % intialize whole metric decompositions (vector)
    PMI_x = nan(1,length(x_unique)); % defintion based on entropy diff.
    prob_X = nan(1,length(x_unique));

    % loop through each instance of X; e.g., {win, lose}    
    for x_num = 1:length(x_unique)        
        x_signal = (X==x_unique(x_num));
        prob_x = mean(x_signal);  % e.g., p(win)           
        prob_X(x_num) = prob_x;

        % loop through all instances of Z
        x_Z_sum = nan(1, length(z_unique)); % sum over Z for this case of X=x
        PMI_z = nan(1,length(z_unique));
        for z_num = 1:length(z_unique)            
            z_signal = (Z==z_unique(z_num));
            prob_z = mean(z_signal);
            prob_z_given_x = mean(z_signal & x_signal) / prob_x;
            
            % loop through all instances of Y(t)
            z_Y_sum = nan(1, length(y_unique)); % sum over Y for this case of Z=z
            PMI_y = nan(1,length(y_unique));
            % H_Y_given_z  = nan(1,length(y_unique)); % H(Y|Z=z)
            % H_Y_given_xz = nan(1,length(y_unique)); % H(Y|X=x,Z=z)
            for y_num = 1:length(y_unique)
                y_signal = (Y==y_unique(y_num));

                prob_x_and_z = mean(x_signal & z_signal);
                prob_y_and_z = mean(y_signal & z_signal);                
                prob_x_y_z   = mean(x_signal & z_signal & y_signal);
                           
                z_Y_sum(y_num) = prob_x_y_z * log2( prob_x_y_z* prob_z / (prob_x_and_z * prob_y_and_z) );          
                if isinf(z_Y_sum(y_num))
                    z_Y_sum(y_num) = NaN;
                end

                % pmi decompositions: based on entropy diff.     
                prob_y_given_z  = prob_y_and_z / prob_z;
                prob_y_given_xz = prob_x_y_z / prob_x_and_z;
                H_y_given_z  = - prob_y_given_z * log2(prob_y_given_z);   % H(Y|Z=z)                
                H_y_given_xz = - prob_y_given_xz * log2(prob_y_given_xz); % H(Y|X=x,Z=z)
                if prob_y_given_z==0
                    H_y_given_z = 0;
                end
                if prob_y_given_xz==0
                    H_y_given_xz = 0;
                end
                PMI_y(y_num) = (H_y_given_z - H_y_given_xz);

                % rectify multiplication by zero
                if prob_x_y_z==0
                    z_Y_sum(y_num) = 0; % convert any Inf to zero
                    % PMI_y(y_num) = 0;
                end
            end
            x_Z_sum(z_num) = nan_sum(z_Y_sum); 
            PMI_z(z_num) = prob_z_given_x * nan_sum(PMI_y);
        end
        % disp(all_Y_sum);
        all_X_sum(x_num) = nan_sum(x_Z_sum);
        PMI_x(x_num) = nan_sum(PMI_z);

        if decompose_flag
            output.(metric_name+"_"+decomp_map(x_unique(x_num))) = prob_x * PMI_x(x_num);
        end        
    end
    output.(metric_name) = nan_sum(all_X_sum);
    % output.("alt_"+metric_name) = nan_sum(prob_X .* PMI_x); % should be identical to above 

    % Normalized metrics
    tempH = Conditional_Entropy_weighted(Y, Z, "H_Y_cond_Z"); % independent of X
    metric_fields = fieldnames(output);
    for m = 1:length(metric_fields)
        output.("N_"+metric_fields{m}) = output.(metric_fields{m}) / tempH.H_Y_cond_Z;
    end    
end

function Sum = nan_sum(Array)
    % avoid returning zero when all elements are NaN
    if all(isnan(Array))
        Sum = NaN;
    else
        Sum = sum(Array, 'omitnan');    
    end
end