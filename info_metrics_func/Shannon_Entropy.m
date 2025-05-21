function H_X = Shannon_Entropy(X, numOutcomes)
%PURPOSE:   Compute Shannon entropy H(X) = -[P(X)*log2(P(X)) + p(~X)*log2(P(~X))] 
%AUTHORS:   Jae Hyung Woo 04/10/2025
    if ~exist('numOutcomes','var')||numOutcomes==2
        % binary random variable X
        H_X = -(mean(X)*log2(mean(X)) + mean(~X)*log2(mean(~X)));
        if isnan(H_X)
            if mean(~X)==0||mean(X)==0
                H_X = 0;    % this prevents zero*-Inf = NaN
            end
        end
    else
        % non-binary case: 
        %H_X = -[mean(X==0)*log2(mean(X==0)) + mean(X==1)*log2(mean(X==1))+ ... + mean(X==N)*log2(mean(X==N))];
        terms = nan(1,numOutcomes);
        for n = 1:numOutcomes
            terms(n) = mean(X==n-1)*log2(mean(X==n-1));
            if mean(X==n-1)==0||mean(X~=n-1)==0
                terms(n) = 0;
            end
        end
        H_X = -sum(terms);
    end
    
end
