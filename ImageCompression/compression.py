import numpy as np
import matplotlib.pyplot as plt
from utils import *
from newFunctions import *


original_image = plt.imread('./bird_small.png')

#plt.imshow(original_image)
#plt.show()

#print("Shape of original image is:",original_image.shape) #gives height,width,no of color channels

X_image = np.reshape(original_image,(original_image.shape[0]*original_image.shape[1],3))

K = 16
max_iters = 10

intitial_centroids = KMeans_init_centroids(X_image,K)

centroids,idx = run_KMeans(X_image,intitial_centroids,max_iters)

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

plot_kMeans_RGB(X_image, centroids, idx, K)

show_centroid_colors(centroids)


# # Find the closest centroid of each pixel
# idx = find_closest_centroids(X_image, centroids)

# # Replace each pixel with the color of the closest centroid
# X_recovered = centroids[idx, :] 

# # Reshape image into proper dimensions
# X_recovered = np.reshape(X_recovered, original_image.shape) 


# # Display original image
# fig, ax = plt.subplots(1,2, figsize=(16,16))
# plt.axis('off')

# ax[0].imshow(original_image)
# ax[0].set_title('Original')
# ax[0].set_axis_off()


# # Display compressed image
# ax[1].imshow(X_recovered)
# ax[1].set_title('Compressed with %d colours'%K)
# ax[1].set_axis_off()