%%
%   COURSE: Master statistics and machine learning: intuition, math, code										
%      URL: udemy.com/course/statsml_x/?couponCode=202006 
% 
%  SECTION: Descriptive statistics
%    VIDEO: Inter-quartile range
% 
%  TEACHER: Mike X Cohen, sincxpress.com
%

%%

% a clear MATLAB workspace is a clear mental workspace
close all; clear
pkg load statistics % load Octave stats package

%% "discrete" entropy

% generate data
N = 1000;
numbers = ceil( 8*rand(N,1).^2 );


% get counts and probabilities
u = unique(numbers);
probs = zeros(length(u),1);

for ui=1:length(u)
    probs(ui) = sum(numbers==u(ui)) / N;
end


% compute entropy
entropee = -sum( probs.*log2(probs+eps) );


% plot
figure(1), clf
bar(u,probs)
title([ 'Entropy = ' num2str(entropee) ])
xlabel('Data value'), ylabel('Probability')

%% for random variables

% create Brownian noise
N = 1123;
brownnoise = cumsum( sign(randn(N,1)) );

figure(1), clf
subplot(211)
plot(brownnoise)
xlabel('Data index'), ylabel('Data value')
title('Brownian noise')

subplot(212)
hist(brownnoise,30) %%% Octave doesn't support: ,'Normalization','probability')
xlabel('Data value'), ylabel('Counts')


%% now compute entropy

% number of bins
nbins = 50;

% bin the data and convert to probability
[nPerBin,bins] = hist(brownnoise,nbins);
probs = nPerBin ./ sum(nPerBin);

% compute entropy
entro = -sum( probs.*log2(probs+eps) );

title([ 'Entropy = ' num2str(entro) ])

%% done.
