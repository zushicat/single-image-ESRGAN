# single-image-ESRGAN
Enhance low-res images with enhanced super-resolution ESRGAN (Keras / Tensorflow implementation).

The overall structure is based on this SRGAN implementation: https://github.com/zushicat/single-image-SRGAN.    
Please check out the according README for details on usage and the mechanics, since those basically apply here as well.     
Also, I try to avoid unnecessary redundancy and rather like to point out how the implementations differ from each other.    


## Differences between the SRGAN and ESRGAN implementations
### Data Loader
- Pass the percentage of the training set you like to compute with *percent_of_training_set* when initializing the DataLoader class (in \__init\__ of Pretrainer and Trainer)
- in function *load_images*:    
I additionally use "pre-crops" of resized versions of the images (i.e. 1600x1600 pixel -> 800x800 pixel and 400x400 pixel, applied to 96x96 pixel -> 24x24 pixel crops) which seems to improve the quality of generated images with input of images with overall lower quality (see below: Results)    
I admittedly didn't really *measure* the results, so it's up to you: If you consider this as utter useless, change following loop accordingly:
```
for size in [None, original_img.height//2, original_img.height//4]
````



### ESRGAN
In fact, there are not too many things that changed.    
Check out this video for a great overview about the differences: https://www.youtube.com/watch?v=KULkSwLk62I
Also, have a look at this blogpost with implementation examples:    
https://medium.com/analytics-vidhya/esrgan-enhanced-super-resolution-generative-adversarial-network-using-keras-a34134b72b77

**class ESRGAN**
The network architecture of the generator has changed. 
- no batch normalization (which also can cause dreaded artifacts)
- additionally in function *upsample*
    - use SubpixelConv2D function (instead of UpSampling2D)
- more layers and connections (see: functions *RRDB* and *dense_block*)

**class Trainer**
The loss functions (called in function *train_step*) have changed (using a relativistic discriminator, trying to predict the probability if a real image is relatively more realistic than a fake image).

### Inference
No differences.


## Results
xxx

