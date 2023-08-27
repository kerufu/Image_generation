
batchSize = 64
imageSize = 64
featureVectorLength = 1024

StandardGANPath = [
    "learningModel/savedModel/StandardGAN/Generator",
    "learningModel/savedModel/StandardGAN/Discriminator"
]

CAAEPath = [
    "learningModel/savedModel/CAAE/Encoder",
    "learningModel/savedModel/CAAE/DiscriminatorOnEncoder",
    "learningModel/savedModel/CAAE/Decoder",
    "learningModel/savedModel/CAAE/DiscriminatorOnDecoder",
]