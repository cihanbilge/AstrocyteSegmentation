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

astrocytes_postProcessing(image_list, subimage_size);
%segmented images are saved in target directory.


%Cihan Bilge Kayasandik, Demetrio Labate
%January,2020
