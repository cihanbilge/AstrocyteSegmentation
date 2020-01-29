function [filts dirs] = dirFilters2D(mSize,mzSize,nBands)
% This function computes a set of filters for Soma Detection.
% 
% INPUT:
%  - mSize: The size of the filters.
%  - nBands: Number of bands.
% 
% OUTPUT:
%  - filts: Cell containing the filters in question
%  - dirs: The direction of the corresponding filters
% 

filts = cell(nBands,1);
dirs = zeros(2,nBands);

theta = (0:(nBands-1))*pi/nBands;
rho = ones(1,nBands);
[X Y] = pol2cart(theta, rho);
dirs(1,:) = X;
dirs(2,:) = Y;
ang=atan(mzSize/mSize);
dist=ceil(sqrt((mzSize/2)^2+(mSize/2)^2));
kt=0;
for k = pi/nBands:pi/nBands:pi
    ang1 = (k-ang); %pi/nBands;
    ang2 = (k+ang); %pi/nBands;
    theta = [ang1, ang2, ang1, ang2, ang1];
    rho = [1,1,-1,-1,1]*dist;
    [X Y] = pol2cart(theta, rho);
    X = X + ceil(mSize/2);
    Y = Y + ceil(mSize/2);
    F = poly2mask(X,Y,mSize,mSize);
    %figure; imshow(F,[]);
    N = numel(find(F==1));
    filts{kt+1} = F/N; kt=kt+1;
   
end
end

%Cihan Bilge Kayasandik, Demetrio Labate
%January,2020