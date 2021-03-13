## Question 1

### Part a

Please implement a box filter and gaussian filter function that can be applied to images. (The functionallity must be just like `cv2.filter2D`)

### Part b

Considering given image `einstein.jpg`, apply a box filter with size three, gaussian filter with sigma equals to 1 and a gaussian filter with sigma eqalst to $\sqrt2$ consequencetly and compair the resualts by the measure of human eye quality and RMSE. (let's assume that filter size is first odd number less than 6$\sigma$)

## Question 2

suppose that we apply a gaussian filter to the image with sigma equals to $\sqrt2$ and next time we filter the image with a gaussian filter that the sigma is $1$, towice and stored the resualts. Are these two equvalent?(Compair them with the measure of RMS and human eye quality)

Could be shown that these two operations are equal mathematically?(gassuan with $\sigma=\sqrt2$ is equals to dubble filtration using gassian with $\sigma=1$)
