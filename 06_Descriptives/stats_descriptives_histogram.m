%%
%   COURSE: Master statistics and machine learning: intuition, math, code										
%      URL: udemy.com/course/statsml_x/?couponCode=202006 
% 
%  SECTION: Descriptive statistics
%    VIDEO: Histogram bins
% 
%  TEACHER: Mike X Cohen, sincxpress.com
%

%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear

%% create some data


% number of data points
n = 1000;

% number of histogram bins
k = 40;

% generate log-normal distribution
data = exp( randn(n,1)/2 );


figure(1), clf

% one way to show a histogram
histogram(data,k)
xlabel('Value'), ylabel('Count')

%% try the Freedman-Diaconis rule

r = 2*iqr(data)*n^(-1/3);
b = ceil( (max(data)-min(data) )/r );

histogram(data,b)

% without iqr function in stats toolbox
histogram(data,'BinMethod','fd')

xlabel('Value'), ylabel('Count')
title([ 'F-D "rule" using ' num2str(b) ' bins' ])

%% movie showing histograms with increasing bins

bins2try = round( linspace(5,n/2,30) );

figure(2), clf
h = plot(0,1,'ks-','linew',2,'markersize',10,'markerfacecolor','w');
set(gca,'xlim',[0 max(data)+.5])
xlabel('Value'), ylabel('Count')

for bini=1:length(bins2try)
    % use hist to get histogram graph data for this bin count
    [y,x] = histcounts(data,bins2try(bini));
    x = (x(1:end-1)+x(2:end))/2;
    
    % update xdata and ydata
    set(h,'XData',x,'ydata',y);
    
    % update title
    title([ 'Histogram with ' num2str(bins2try(bini)) ' bins.' ])
    pause(.5)
end    

%% done.
