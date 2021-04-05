# High-Res-Images-From-SRGAN
A neural network which compresses images and re-generates high resolution versions of these images, to enable efficient transfer of images over a network.  

# Overview
The general working of the system is that we want to have the Encoder part from the Variational AutoEncoder at the server side or one end of the system to convert the given image to the latent space. This latent space array is then sent through the network to the other end/client side. The latent space array is now reconstructed back using the decoder network. This reconstructed image is passed through a image super resolution network, in our case, the SRGAN to upsample and create a high-res image and then display.
# Training 
1. We used the Image Align Celeb dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train the network.
2. We wrote a custom data-loader, which takes one sample image as input and generates two output images – one in low resolution and one in high resolution. The low-res image is created by scaling down the size of the original image by 4, whereas the high-res image is the scaled up by 2. 3
3. We train the generator and discriminator in tandem. For every batch, the generator takes a low-res image as input and generates a high-res version. The discriminator then tries to distinguish between this generated high-res image and the original high-res image.
4. We trained our model for 50 epochs, with a learning rate of 0.0002 using the Adam optimizer. Our model was trained on an RTX 2080Ti. After every epoch, we saved the weights of the model. The weights of the model which provided the lowest loss value were saved for inference. We also applied learning rate decay after every 10 epochs. 

# Instructions to run the code
Running the VAE : python vae_mnist.py
Running the VAE (for images) : python vae.py
Training the SRGAN : python train.py
Running the SRGAN (to produce high-res images): python inference.py 

Note : Please modify the input directory in the files to match the location of the inputs images in the system you are running it on. While running inference.py, the output directory (to store the high res images) needs to be specified as well. 
