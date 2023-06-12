# Image segmentation for nuclei project
## Project detail
An algorithm that can automatically detect nuclei to expedite research on a wide range of diseases, including cancer, heart disease, and rare disorders. 
Such tool has the potential to significantly improve the speed at which cures are developed, benefiting those suffering from various health conditions, including chronic obstructive pulmonary disease, Alzheimer's, diabetes, and even the common cold.
Hence, the identification of cell nuclei is a crucial first step in many research studies because it enables researchers to analyze the DNA contained within the nucleus, which holds the genetic information that determines the function of each cell. 
By identifying the nuclei of cells, researchers can examine how cells respond to different treatments and gain insights into the underlying biological processes at play. 
An automated AI model for identifying nuclei, offers the potential to streamline drug testing and reduce the time for new drugs to become available to the public.
Thus, a model for semantic segmentation for images containing cell neuclei is created.
## Guide
Inside the github the model file is provided. You may use it to perform your project.
You may download the model and use it for your project. The test dataset needs to be converted to full numpy array and expand the mask dimension to include the channel axis.
Convert the mask value into just 0 and 1 and normalize the images pixel value. The next step is to convert the numpy array into tensorflow tensors and combine features and labels together to form a zip dataset.
Then the data is converted into a prefecth dataset batch the testing dataset. The step and code on how to perform the step is shown in the python file. Then the project is developed using the transfer learning. The plan is to apply transfer learning by using a pretrained model as the feature extractor.
Then, proceed to build  upsampling path with the tensorflow_example module imported + other default keras layers.
## Output of the Project
- This is the base model architecture 
![alt text](model_arch.png)
- This UNET model 
![alt text](UNETmodel_arch.png)
- This is the model training in which the validation accyracy and loss is shown
![alt text](model_training.PNG)
-This is the model deployment using the test dataset provided
![alt text](model_deployment.PNG)
## Credits for the dataset
https://www.kaggle.com/competitions/data-science-bowl-2018/overview