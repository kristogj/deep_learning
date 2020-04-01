# Semantic Segmentation of Cityscapes
# Main
How should you run each of our models? Each individual architecture has their own starter file. 
These files include all the settings used during training, validation and testing.
The files are:
* starter.py - for the baseline model
  * In dataloader.py in the `_transform` function, make sure the section for resize is uncommented and the argument for `transforms.Resize` is 512.  Further, comment out the section for random cropping.
* altner_starter.py - for the alternative architecture
  * In dataloader.py in the `_transform` function, make sure the section for random cropping is uncommented and the input/output sizes are 512 respectively.  Further, comment out the section for resizing.
* transfer_learning_starter.py - for the transfer learning model
  * In dataloader.py in the `_transform` function, make sure the section for random cropping is uncommented and the input/output sizes are 512 respectively.  Further, comment out the section for resizing.

Each of them are runnable using `python3 *starter.py`.

To use `weighted_loss`, set the flag of the same name to `True` in the starter file.
Please change the minibatch sizes in any of the respective starter files if an out of memory error occurs.

After a model has finished training, the outputs are `res.csv`, a log of the pixel accuracies and IoUs for every epoch, `app.log`, a log of the outputs during training, `test.png` is the segmentation of the test image, and `baseline_train_val_loss.png` will be the graph comparing training/validation loss per epoch.  The latter's name is the same regardless of what model you run.  

To overlay the segmentation and original image on the cluster, go into a python3 environment, `import utils` and in the PA3 folder, do `utils.overlayImages('test.png', '/datasets/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png')`.

# Model architectures
The architecture classes can be found in **models.py**, along with the code for how data is forwarded 
through the network

# Model Optimization
Training, validation and test loops can be found in **optimization.py**.
This would be where all the training and generating of numerical results
are happening.

# Utility functions
We have several utility functions helping us evaluating the models. These can be found in **utils.py**, 
where we have methods for
* iou_class - computing the iou for a specific class
* iou - average iou over all classes
* pixel_acc - pixel accuracy
* overlayImages - generating the test image with a the predicted mask overlaid.

For generating the plots we used the function in **plotting.py**

