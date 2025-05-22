function SEM = sem(x, DIM)
    if nargin<2
        DIM = 1;
    end
    SEM = std(x,[],DIM,'omitnan')./sqrt(sum(~isnan(x),DIM));
end