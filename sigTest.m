
% [pVal] = sigTest(dataStatistic, nullStatistic, 'upper', 'gamma')
function [pVal] = sigTest(dataStatistic, nullStatistic, tail, option)

if size(nullStatistic,2)~= length(dataStatistic)
    nullStatistic = nullStatistic.';
end
pVal = nan(size(dataStatistic));

numSamples = size(nullStatistic, 1);
for i = 1:length(dataStatistic)
    if strcmpi(option, 'empirical')
        %% empirical distripution based pVal value
        if strcmpi(tail, 'upper')
            pVal(i) = sum(nullStatistic(:,i)>=dataStatistic(i))/numSamples;
        elseif strcmpi(tail, 'lower')
            pVal(i) = sum(nullStatistic(:,i)<=dataStatistic(i))/numSamples;
        end
    elseif strcmpi(option, 'gamma')
        %% fit gamma to empirical null samples (if positive only)
        [prms, ~] = gamfit(nullStatistic(:,i));
        if strcmpi(tail, 'upper')
            pVal(i) = 1-gamcdf(dataStatistic(i),prms(1), prms(2));
        elseif strcmpi(tail, 'lower')
            pVal(i) = gamcdf(dataStatistic(i),prms(1), prms(2));
        end
    elseif strcmpi(option, 'gauss')
        %% fit gaussian to empirical null samples (if positive and negative only)
        mu = mean(nullStatistic(:,i));
        sigma = std(nullStatistic(:,i));
        if strcmpi(tail, 'upper')
            pVal(i) = 1- normcdf(dataStatistic(i),mu,sigma);
        elseif strcmpi(tail, 'lower')
            pVal(i) = normcdf(dataStatistic(i),mu,sigma);
        end
    elseif strcmpi(option, 'Wilcoxon')
        %% returns the p-value of a two-sided Wilcoxon rank sum test. ranksum tests the null hypothesis that data in x and y are samples from continuous distributions with equal medians, against the alternative that they are not. The test assumes that the two samples are independent. x and y can have different lengths.This test is equivalent to a Mann-Whitney U-test. Mann?Whitney U test
        if strcmpi(tail, 'upper')
            pVal(i) = ranksum(dataStatistic(i),nullStatistic(:,i), 'tail', 'right');
        elseif strcmpi(tail, 'lower')
            pVal(i) = ranksum(dataStatistic(i),nullStatistic(:,i), 'tail', 'left');
        elseif strcmpi(tail, 'two')
            pVal(i) = ranksum(dataStatistic(i),nullStatistic(:,i));
        end
    end
end



end