
# Transformer Based Image Captioning

The transformer based image captioning uses the standard transformer architecture on the decoder side, but uses a special encoder module with visual attention. 

![](https://miro.medium.com/max/2880/1*BHzGVskWGS_3jEcYYi6miQ.png)

The encoder takes a feature map from InceptionV3 (the second to last layer) in order to produce a (1,2048) feature map. The feature map is then sent to a CNN Encoder that takes these features and turns them into a (1,256) feature embedding. The feature embeddings are then sent to a customized Bahdanau Attention module that computes an attention matrix for the feature embeddings. This attantion matrix is then multiplied by the feature embeddings to produce an attention map on the original embedding. Finally, the attention-mapped feature embeddings are sent to a fully connected layer that transforms the dimensions of the attention-mapped feature embeddings to (1,256), so the dimensions are compatible with multiheaded attention unit of the decoder block.
## Inspiration
Inspired by Tensorflow's [Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer)
along with Tensorflow's [Image captioning with visual attention](https://www.tensorflow.org/tutorials/text/image_captioning)

## Dataset
The model uses a subset of the [MS COCO 2017](https://cocodataset.org/#home) image-caption pairs (~100k images) for training. 

### <center>Example Image</center>
![](https://www.researchgate.net/profile/Konda-Reddy-Mopuri/publication/317164046/figure/fig1/AS:498120245956609@1495772529994/Sample-image-caption-pairs-from-MSCOCO-dataset-11-The-caption-gives-more-information.png)

## Results
TBA
