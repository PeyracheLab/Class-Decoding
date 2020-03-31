function [thetaEst,thetaP,thetaMat] = BayesReconstruction_1D(pf,Q,thetaVec,tau,varargin) 

% Bayesian reconstruction of a one dimensional signal (e.g. head-direction)
% based on neuron tuning curves and instantaneous firing.
% 
%  USAGE
%
%    [thetaEst,p,thetaMat] = HeadDirectionReconstruction(allPF,binnedSpk,Px,binVal,tau)
%
%    allPf:     a cell array of HD tuning curves (Nbins x Ncells) in Hz
%    binnedSpk: a TxNcells matrix of binned spikes trains (T time bins)
%    thetaVec:  Nbins position values (e.g. angles in 
%    tau:       time bin duration (in seconds)    
%
%    Px (optional): occupancy map (histogram of time spent in each bin)
%                   if omitted, assumed to be flat.
%
%
%  OUTPUT
%
%    thetaEst:  estimated values of theta (the 1D position value)
%    thetaP:    posterior probabilities of the reconstructed positions
%    thetaMat:  matrix of posterior probabilities (for all position bins)
%

% Copyright (C) 2014-2018 by Adrien Peyrache
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.

N = size(pf,2);
if size(Q,2)~=N
    error('insconsistent size between place field array and binned spike train matrix')
end

n = size(pf,1);
nt = length(thetaVec);
if n~=nt
    error('insconsistent size between place field array and spatial bin vector ')
end

if ~isempty(varargin)
    Px = varargin{1};
    if n~=length(Px(:))
        error('insconsistent size between place field array and occupancy map')
    end

    Px(Px==0)=min(Px(Px>0));
    Px = Px./sum(Px(:));
    constTerm = log(Px(:))-tau*sum(pf,2);
else
    constTerm = -tau*sum(pf,2);
end

B = size(Q,1);
thetaEst = zeros(B,1);
thetaMat = zeros(B,length(thetaVec));
thetaP = zeros(B,1);

pf = log(pf);
pf(pf<-16) = -16;

lChunk = 100;
nbChunk = floor(B/lChunk);
cT = repmat(constTerm(:)',[lChunk 1]);
pfR = repmat(pf,[1 1 lChunk]);
pfR = permute(pfR,[3 2 1]);

for ii=1:nbChunk-1
    ix = (ii-1)*lChunk+1:ii*lChunk;
    dq = Q(ix,:);
    pv = repmat(dq,[1 1 n]);
    
    logPxn = squeeze(sum(pv.*pfR,2)) + cT;
    logPxn = exp(logPxn);
    logPxn = logPxn./repmat(sum(logPxn,2),[1 nt]);
    [thetaP(ix),maxIx] = max(logPxn,[],2);
    thetaEst(ix) = thetaVec(maxIx);
    thetaMat(ix,:) = logPxn;
end

%Here, we consider the last chunk AND the remainder just in case the remainder is only one time bin
remainderChunk = B-(nbChunk-1)*lChunk;
if remainderChunk>0 
    ix = (nbChunk-1)*lChunk+1:B;
    cT = repmat(constTerm(:)',[remainderChunk 1]);

    dq = Q(ix,:);
    pv = repmat(dq,[1 1 n]);
    pfR = repmat(pf,[1 1 remainderChunk]);
    pfR = permute(pfR,[3 2 1]);
    logPxn = squeeze(sum(pv.*pfR,2)) + cT;
    logPxn = exp(logPxn);
    logPxn = logPxn./repmat(sum(logPxn,2),[1 nt]);
    [thetaP(ix),maxIx] = max(logPxn,[],2);
    thetaEst(ix) = thetaVec(maxIx);
    thetaMat(ix,:) = logPxn;
else
    
end
