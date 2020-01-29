function [result_im, DR, Id]=checkBlobness(im)
%this eliminates the string like or blob like cell regions.
%activate this if only star-shaped astrocytes are target. 
C=connComp(im);
nBands=20; % number of orientation
final_z=zeros(size(im));
filtIm=zeros(size(im,1),size(im,2),nBands);
sx=7;
sy=2;%double(ceil(sqrt(sx)));
[filts ~] = dirFilters2D(sx,sy,nBands);
for i=1:nBands
    filtIm(:,:,i)=conv2(im, filts{i},'same');
end
[M,I]=max(filtIm,[],3); I(isnan(I))=0;  Id=I;
[m,Im]=min(filtIm,[],3); Im(isnan(Im))=0;
DR= (m.^2)./M;
% to eliminate blob-like cells:
st=strel('disk',3);
im2=double(imopen(im,st)); im2i=im2./max(im2(:));
im2=im2i-1; I=-im2;
im3=double(imclose(im,st));im3i=im3./max(im3(:)); 
%to eliminate string-like objects:
for i=1:C.compNum
    z=zeros(size(im));zi=z;
    z(C.compIdx{i})=1; z2=z;
    C11=connComp(im2i.*z);
    zi=z.*I;
    zi(isnan(zi))=0;
    cc=connComp(zi); K=find(cc.compCard>20);
    if (length(K)>1)
        final_z=final_z+z2;
    end
end
result_im=final_z;

end

%Cihan Bilge Kayasandik, Demetrio Labate
%January,2020