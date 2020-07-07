function astrocytes_postProcessing(image_list, subimage_size)
%locate segmented cells on their original placein the original image. 
%eliminates bad segmentation, and incorrect cell detections. 
for im=1:length(image_list)
    
    side=subimage_size/2; %half length of one side of the boxes. Complete cell region must be inside of these boxes so need to be determined accordingly.
    centre=subimage_size*subimage_size/2; %Centre of the image. Since we assume that cell 
    %body is on the centre of the image, all detections whose centre is empty will be eliminated instantly.
    load(sprintf('myimage_%d_connComp_V',image_list(im))); %all information extracted from target images, and saved in preprocessing step.
    
    im_size=size(image);
    image_base=zeros(im_size);
    invalids=0;
    %unselectedOnes=[];
    file_path=pwd;
    figure;
    for i=1:C.compNum
        if (~ismember(i,V))
            z=zeros(size(im_size));
            z(C.compIdx{i})=1;
            CC=bwconncomp(z);
            S=regionprops(CC,'Centroid');
            c=S.Centroid; c=ceil(fliplr(c));
            file_name=sprintf('%d.tif', i+tt);
            try
                
                stacks = tiffRead(fullfile(file_path,file_name),{'MONO'});
                patch = double(stacks.MONO); patch=double(max(patch,[],3))./255;
                patch(patch>0.05)=1; patch(patch~=1)=0;
                Cp=connComp(patch);
                z=zeros(subimage_size);
                for u=1:Cp.compNum
                    cp=Cp.compIdx{u}; cpp=intersect(cp, centre);
                    if (isempty(cpp)==0 && Cp.compCard(u)>10)
                        z(cp)=1;
                    end
                end
                patch=z;
                if (sum(sum(double((patch-ones(subimage_size)==zeros(subimage_size)))))==0)
                    patch=zeros(128);
                end
                image_base(max(c(1)-side,1):min(max(c(1)-side,1)+127,min(c(1)+side,size(im_size,1))), max(c(2)-side,1):min(max(c(2)-side,1)+127,min(c(2)+side,size(im_size,2))))=patch+image_base(max(c(1)-side,1):min(max(c(1)-side,1)+127,min(c(1)+side,size(im_size,1))), max(c(2)-side,1):min(max(c(2)-side,1)+127,min(c(2)+side,size(im_size,2))));
            catch
                invalids=invalids+1;
                continue
            end
        end
    end
    
    mask=image_base;
    mask(mask>0)=1;
    [mask, ~, ~]=checkBlobness(mask); %elimiates non star-shaped segmentations. %,0.5,0.1);
    rgbImage = cat(3, image, mask, zeros(size(image))); %output image 
    imwrite(rgbImage, sprintf('segmentation_image%d.tif', im));
end
end



%Cihan Bilge Kayasandik, Demetrio Labate
%January,2020
