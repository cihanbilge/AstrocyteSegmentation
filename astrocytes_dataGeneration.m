%random data generation astro
%this is the algorithm that I use to gerneate sub images and label data for
%horvath data. Then this sub images are passed to python code. This code is
%not for error calculation.
%%
function []=astrocytes_dataGeneration(image_list, subimage_size)
%input:
% image list: list of the images which will be processed.

%output:
%all necessary output files are saved in selected directories. To change
%the directories see the code below.
%image_list=[743 745 746 747 750 751 752];%list of the images which will be processed.
imsizeFactor=1.5; % DL segmentation model is designed to work on cells which fits on a [128,128] window. 
%If the cell size in targetted image is smaller or larger than that certain
%size, imresizing will be necessary to optimize method for new image. 
for k=1:length(image_list)
    % here the images will be processed to detect possible cell centers,
    % then subimages will be generated for each cell center. These
    % subimages will feed the DL network.
    k1=image_list(k);
    %file_path='/Volumes/GoogleDrive/My Drive/Safari_Downloads/images'; %path of the images
    file_name=sprintf('%d.tif',k1); %title of the images
    fold_name=sprintf('cell_%d',k1);%title of subimages
    fold_name_label=sprintf('cell_%d_label',k1);%?
    mkdir(pwd, fold_name)%create a directory to store subimages 
    mkdir(pwd, fold_name_label)%create a directory to store labels of subimages if exist?
    path=sprintf('../cell_%d',k1);%path of the subimages
    %read the raw image and normalize the intensities
    stacks = tiffRead(fullfile(file_path,file_name),{'MONO'});
    image_raw = double(stacks.MONO);
    image_raw =max(image_raw ,[],3);
    %image_raw=-image_raw; image_raw=image_raw-min(image_raw(:)); image_raw=image_raw./max(image_raw(:)); 
    %s=max(size(image_orig)); images=zeros(s); images(1:size(image_orig,1), 1:size(image_orig,2))=image_orig;
    %image=double(shearDen2D(image));
    image =imgaussfilt(image_raw); %smooth the image
    image = image./max(image(:)); %image normalization
    %figure; imshow(image,[])
    image = imresize(image,imsizeFactor);
    %figure; imshow(image,[])
    unselectedOnes=[];%?
    %% detection of possible cell centers
    %this part base on directional ratio analysis to determine hihgly
    %anisotropic (star-shaped astrocytes) regions on the image.
    nBands=10; % the analysis based on 10 directions
    filtIm=zeros(size(image,1),size(image,2),nBands);
    sx=3;%size of filters in majow axis
    sy=1;%size of filters in minor axis
    [filts ~] = dirFilters2D(sx,sy,nBands); %filter generation
    for i=1:nBands
        filtIm(:,:,i)=conv2(image, filts{i},'same');
    end
    [M,I]=max(filtIm,[],3); I(isnan(I))=0; 
    [m,I]=min(filtIm,[],3); I(isnan(I))=0; 
    DirectionalRatio= (m.^2)./(M); %modified directional ratio 
    DirectionalRatio(DirectionalRatio<0.5)=0; 
    %due to high noise level, background regions may have large anisotropy.
    %Hence, we eliminate anisotropic regions with low intensity value,
    %whihc are assumed to be part of background.
    DirectionalRatio(find(DirectionalRatio))=1; %generating a mask for possible foreground regions
    %figure; imshow(DirectionalRatio,[])
    %to catch the cells close to the boundray efficiently we zero-padd the
    %image and Directional Ratio from each side
    imagebase=zeros(max(size(image))+subimage_size);
    side=subimage_size/2; %half length of one side of the boxes. Complete cell region must be inside of these boxes so need to be determined accordingly.
    imagebase(side:side-1+sm(1),side:side-1+sm(2))=DirectionalRatio;
    DirectionalRatio=imagebase;
    imagebase=zeros(sm_max+subimage_size);
    imagebase(side:side-1+sm(1),side:side-1+sm(2))=image;
    image=imagebase;
    clear imagebase
    detections=zeros(size(image));
    C=connComp(DirectionalRatio); %number of possible cell regions. 
    tt=0; V=[];
    for i=1:C.compNum
        base=zeros(size(image));
        base(C.compIdx{i})=1; %ROI
        CC=bwconncomp(base);
        S=regionprops(CC,'Centroid'); % try to locate box at the centroid of the cell region
        c=S.Centroid; c=ceil(fliplr(c));
        patch=image(max(c(1)-side,1):min(c(1)+side,size(image,1)), max(c(2)-side,1):min(c(2)+side,size(image,2)));
        patch=patch./max(patch(:)); %normalize patches
        if (all(size(patch)>128)) % if the box is not square, then detected cell is so close to the boundary, can be eliminated
            patch=patch(1:128,1:128); %remove later, to eliminate possible issues..
            patch=patch./max(patch(:));
            detections_path=fullfile(path, sprintf('%d.tif',i+tt));
            imwrite(patch,detections_path,'WriteMode','append');
        else
            V=[V; i]; % store the possible detections which would be eliminated later.
            %For dispatching them back, this list will be needed. 
        end
    end
    filename=sprintf('myimage_%d_connComp_V.mat', k1);
    save(filename,'V','image','C','unselectedOnes');
end
end



%Cihan Bilge Kayasandik, Demetrio Labate
%January,2020