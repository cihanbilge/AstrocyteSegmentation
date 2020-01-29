%main script for astrocyte detection. 

%firstly generate a list of the names of the images which will be
%segmented.
subimage_size=128;
astrocytes_dataGeneration(image_list, subimage_size);
%input: 
%image_list: names of the images which will be processed;
%images are assumed to be in tif format,
%image names are assumed to be numbers. 
%e.g: if images 1.tif and 2.tif will be analyzed set image_list=[1 2];
%subimage_size: length of one side of the boxes to generate subimages to
%segment. One box is generated for each cell in the Original input image. 
%Some boxes might overlap. One cell should be covered by one box. For our
%dataset this value is set to 128. 

%this part generates sub images to be segmented by the GESU-net. Images 
%firstly analyzed to find possible astrocyte regions. Then, sub images are 
%generated for each of the detection. These subimages are saved to the 
%directory. Then GESU-net must be fed by these sub images and associated 
%segmentations will be generated. These segmentations will be collected by 
%the below postprocessing step, processed according to morphology and bad 
%segmentations will be eliminated.  

astrocytes_postProcessing(image_list, subimage_size);
%segmented images are processed and properly located in the original image. 
%the output will be a rgb image, formed by the original image (red channel)
%and segmented cells (green channel), which is saved in the target directory.


%Cihan Bilge Kayasandik, Demetrio Labate
%January,2020
