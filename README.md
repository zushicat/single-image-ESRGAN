# single-image-ESRGAN
Enhance low-res images with enhanced super-resolution ESRGAN (Keras / Tensorflow implementation).

The overall structure is based on this SRGAN implementation: https://github.com/zushicat/single-image-SRGAN    

Please check out the according README for details on usage and the mechanics, since those basically apply here as well. 
Also, I try to avoid unnecessary redundancy and rather like to point out how the implementations differ from each other.    


## Differences between the SRGAN and ESRGAN implementations
### Data Loader
- Pass the percentage of the training set you like to compute with *percent_of_training_set* when initializing the DataLoader class (in \__init\__ of Pretrainer and Trainer)    
The default is 0.5 (50%)
- in function *load_images*:    
I additionally use "pre-crops" of resized versions of the images (i.e. 1600x1600 pixel -> 800x800 pixel and 400x400 pixel, applied to 96x96 pixel -> 24x24 pixel crops) which seems to improve the quality of generated images with input of images with overall lower quality (see below: Results)    
I admittedly didn't really *measure* the results, so it's up to you: If you consider this as utter useless, change following loop accordingly:
```
for size in [None, original_img.height//2, original_img.height//4]
````



### ESRGAN
Check out this YT video for a great overview about the differences between SRGAN and ESRGAN:     
["How Super Resolution Works" by Leo Isikdogan](https://www.youtube.com/watch?v=KULkSwLk62I)    

Also, have a look at this blogpost with implementation examples:     
["ESRGAN: Enhanced Super-Resolution Generative Adversarial Network using Keras" by Chhaya Vankhede](https://medium.com/analytics-vidhya/esrgan-enhanced-super-resolution-generative-adversarial-network-using-keras-a34134b72b77)    

In fact, there are not too many things that changed.    

**class ESRGAN**    
The network architecture of the generator has changed. 
- no batch normalization (which can also cause dreaded artifacts)
- additionally in function *upsample*
    - use SubpixelConv2D function (instead of UpSampling2D)
- more layers and connections (see: functions *RRDB* and *dense_block*)    

**class Trainer**    
The loss functions (called in function *train_step*) have changed (using a relativistic discriminator, trying to predict the probability if a real image is relatively more realistic than a fake image).

### Inference
No differences.


## Results
xxx

Resolution of input image  | Input images | Output images 1 |  Output images 2 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
400x400 pixel |<img src="readme_images/cropped/high_quality_inputs/input_400.png" width="300" />|![](readme_images/high_quality_inputs/cropped/output_400_fixed_size.png)|![](readme_images/high_quality_inputs/cropped/output_400_var_size.png)
300x300 pixel |<img src="readme_images/cropped/high_quality_inputs/input_300.png" width="300" />|![](readme_images/high_quality_inputs/cropped/output_300_fixed_size.png)|![](readme_images/high_quality_inputs/cropped/output_300_var_size.png)
200x200 pixel |<img src="readme_images/cropped/high_quality_inputs/input_200.png" width="300" />|![](readme_images/high_quality_inputs/cropped/output_200_fixed_size.png)|![](readme_images/high_quality_inputs/cropped/output_200_var_size.png)
150x150 pixel |<img src="readme_images/cropped/high_quality_inputs/input_150.png" width="300" />|![](readme_images/high_quality_inputs/cropped/output_150_fixed_size.png)|![](readme_images/high_quality_inputs/cropped/output_150_var_size.png)
100x100 pixel |<img src="readme_images/cropped/high_quality_inputs/input_100.png" width="300" />|![](readme_images/high_quality_inputs/cropped/output_100_fixed_size.png)|![](readme_images/high_quality_inputs/cropped/output_100_var_size.png)

xxx
