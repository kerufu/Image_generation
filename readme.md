A simple django website myP to gerarate pictures with GANs and classify if pictures is required type

Experiment on
1. training GAN to work like auto encoder
2. training CAAE with cross attention layer
3. training with continuously increasing dataset
4. effect of flatten in image generation (will degenerate the model)

In learning mode, show pictures randomly, user labels pictures. Train the models with mini batch learning at the end.
In non-learning mode, only show pictures similar to labeled pictures.

Must specify the path of image dataset in myP.settings.PICTURE_DIR
