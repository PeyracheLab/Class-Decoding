% Supervised and unsupervised techniques to decode signals from a
% population of neurons: the case of head-direction cells.
% Workshop first given at the Qlife 2020 winter school (Paris)
% 
% This tutorial will guide through the analysis of HD cell population data
% and how to extract a head-direction and simple topological features from
% the population activity. It starts with PCA and then shows how to use
% IsoMap to extract the topology without the inherent constraints of PCA.

% Copyright (C) 2020 Adrien Peyrache
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.

%% Load data
load dataHD.mat

% what it contains:
% - Q{run,sws and rem} are three binned spike trains from 19 simultaneously
%   recorded head-drection (HD) neurons in the anterodorsal nucleus of a
%   freely moving mouse. Bin size: 10ms
% - angRun: animal's HD (same time bins as Qrun). Two column matrix (times,
%   data)
% - prefHD: preferred HD of each neuron.
% - hdTuning: tuning curves. First column are the angular bin values
%
% For more details regarding the recording:
% Peyrache et al. (2015) Internally organized mechanisms of the head
% direction sense. Nature Neuroscience 18, 569–575.


%% Decoding in PCA space during exploration
%Compute the correlation matrix
C = corrcoef(Qrun);

%Compute PCA from the correlation matrix
PCweights = pcacov(C);

%Sort the neurons by preferred direction
[~,prefIx] = sort(prefHD);

%Renaming the variables and plot them twice
theta = prefHD(prefIx);
theta = [theta;theta+2*pi];

w1 = PCweights(prefIx,1);
w2 = PCweights(prefIx,2);

% Plot of the first two PCs. What is your interpretation?
figure(1),clf
stem(theta,[w1;w1],'b');
hold on
stem(theta,[w2;w2],'r');
legend('PC 1','PC 2','location','EastOutside')
ylabel('PC weights')
xlabel('Prefered Direction')
set(gca,'XTick',[0:pi/2:4*pi])
set(gca,'XTickLabel',{'0';'\pi/2';'\pi';'3\pi/2';'0';'\pi/2';'\pi';'3\pi/2'})

%PC score for the first 2 PCs
PCscoreRun = zscore(Qrun)*PCweights(:,1:2);

%Smoothing of the score in time to remove noise
PCscoreRun = gaussFilt(PCscoreRun,20,0);

%Compute the color code to display HD
cmap = hsv;

%1st column of angRun is time, 2nd is data.
ang = angRun(:,2);
angContrast = (ang - min(ang)) / (max(ang) - min(ang));
colRunIx = floor(angContrast * 63) + 1;

% Here we plot PCA decoding coloured by HD
figure(2),clf
    scatter(PCscoreRun(:,1),PCscoreRun(:,2),10,cmap(colRunIx,:))
    xlabel('PC1 score')
    ylabel('PC2 score')

    
%QUESTION: How can we decode a HD signal?

% Using atan
angPCA = atan2(PCscoreRun(:,2),PCscoreRun(:,1));

% Problem! The angular offset of PCA decoding is arbitrary.
% Let's find it by estimating the average offset with actual HD.

angDiff = angPCA - angRun(:,2);

% Here, let's compute angular mean
angDiff = [cos(angDiff) sin(angDiff)];
angDiff = atan2(mean(angDiff(:,2)),mean(angDiff(:,1)));

% Correct for the offset
angPCA = mod(angPCA - angDiff,2*pi);

figure(3),clf
plot(angRun(:,1),angRun(:,2))
hold on
plot(angRun(:,1),angPCA)
legend({'True HD';'PCA HD'})

%% PCA decoding during REM
PCscoreREM = zscore(Qrem)*PCweights(:,1:2);
PCscoreREM = gaussFilt(PCscoreREM,20,0);

angSleep = atan2(PCscoreREM(:,2),PCscoreREM(:,1));
angContrast = (angSleep - min(angSleep)) / (max(angSleep) - min(angSleep));
colRemIx = floor(angContrast * 63) + 1;

figure(4),clf
    scatter(PCscoreREM(:,1),PCscoreREM(:,2),10,cmap(colRemIx,:))
    xlabel('PC1 score')
    ylabel('PC2 score')
    
% QUESTION:
% How can we test that the HD system is a 'ring attractor' during
% exploration and sleep?
  
% By looking at distrance from center
PCcenter = mean(PCscoreREM);
distFromCenter = sqrt( sum( (PCscoreREM-repmat(PCcenter,[size(PCscoreREM,1) 1])).^2,2));

figure(5),clf
[h,b] = hist(distFromCenter,100);
bar(b,h/sum(h),1)
xlabel('Distrance from center')
ylabel('Percentage')
title('Is it a ring?')

%% Other unsupervised technique: IsoMap
% This section needs the following toolbox to work:
% https://lvdmaaten.github.io/drtoolbox/
%
% see these three papers:
%      Original IsoMap paper:
%      J. B. Tenenbaum, V. de Silva, J. C. Langford (2000).  A global
%      geometric framework for nonlinear dimensionality reduction.  
%      Science 290 (5500): 2319-2323. 
%
%      Tpology of HD cell population:
%      R. Chaudhuri et al., (2019). The intrinsic attractor manifold and
%      population dynamics of a canonical cognitive circuit across waking
%      and sleep. Nature Neuroscience 22(9):1512-1520
%
%      An example of HD decoding using IsoMap:
%      G. Viejo and A. Peyrache (2019). Precise coupling of the thalamic
%      head-direction system to hippocampal ripples. bioRxiv 
%      https://doi.org/10.1101/809657 
% 

% We need to downsample data (IsoMap needs a lot of memory...)
dataSelect = 1:50:min(100000,size(Qrem,1));
data = gaussFilt(Qrun,50,0); %We filter at the downsampling rate
data  = zscore(data(dataSelect,:));
mapRun = compute_mapping(data, 'Isomap', 2);

data = gaussFilt(Qrem,50,0); %We filter at the downsampling rate
data  = zscore(data(dataSelect,:));
mapRem = compute_mapping(data, 'Isomap', 2);

figure(6),clf
subplot(1,2,1)
    scatter(mapRun(:,1),mapRun(:,2),10,cmap(colRunIx(dataSelect),:))
    xlabel('IsoMap dim. 1')
    ylabel('IsoMap dim. 2')
    title('RUN')
subplot(1,2,2)
    scatter(mapRem(:,1),mapRem(:,2),10,'k')
    xlabel('IsoMap dim. 1')
    ylabel('IsoMap dim. 2')
    title('REM')
    
    
%% Bayesian decoding 
q = gaussFilt(Qrun,20,0,0); %non-normalized smoothing (to preserve actual number of spikes)

angBayes = BayesReconstruction_1D(hdTuning(:,2:end),q,hdTuning(:,1),0.01);
angBayes = mod(angBayes,2*pi);

%time:
t = angRun(:,1);

figure(5),clf
plot(t,angRun(:,2))
hold on
plot(t,angPCA)
plot(t,angBayes)

%% QUESTION: compare each method

errorPCA = abs(angPCA-angRun(:,2));
errorPCA(errorPCA>pi) = 2*pi-errorPCA(errorPCA>pi);
errorPCA = mean(errorPCA);

errorBayes = abs(angBayes-angRun(:,2));
errorBayes(errorBayes>pi) = 2*pi-errorBayes(errorBayes>pi);
errorBayes = mean(errorBayes);

figure(6),clf
bar([errorPCA errorBayes])
ylabel('Average error (rad.)')
set(gca,'Xtick',[1 2])
set(gca,'XtickLabel',{'PCA';'Bayesian'})
title('Average error (rad.)')