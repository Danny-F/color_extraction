## Color Extraction
Extracting colors from images

# Introduction
For a different project, I needed to be able to get more detailed color information for products and so I created a process to extract colors from image files.

The process uses dynamic clustering on the RGB pixel values from an image file. The process is dynamic in the sense that the number of clusters that are identified varies automatically for each image. Rather than hard code a cluster number for the clustering model, the number of clusters identified is increased until the decrease in "inertia" (the bend of the elbow graph) is not significant enough to warrant increasing the cluster size further.

Also, instead of using RGB values for the clustering model, the RGB values are converted into L*a*b values (https://en.wikipedia.org/wiki/CIELAB_color_space). The important difference between RGB and L*a*b is that the distance between L*a*b values is more representative of how the human eye detects differences in colors. RGB values are more-or-less just ordinal data and each unit of distance between RGB values is not as meaningful. For example, a super pale orange, a color that essentially looks yellow, would appear closer to the primary color of yellow on the L*a*b spectrum than if plotted on an RGB spectrum. Using L*a*b instead of RGB, helped improve the accuracy of the clustering model since L*a*b values make more sense when plotted in a mathematical landscape.

# Requirements
Python packages:
- matplotlib
- skimage
- numpy
- pandas
- itertools
- matplotlib
- skimage
- sklearn
- streamlit
