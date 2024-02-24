                          Parcellation and Sampling: An Examination at Mesoscopic and Microscopic Levels


A classic problem in microscope image analysis is quantifying regions with high clustering, where cells cannot be properly segmented from each other. To address this, this proposal aims to identify different regions based on their density and then apply Ripley's K function to quantify the level of clustering in these regions.

This project aims to achieve unbiased identification of regions of interest by employing computational algorithms on images obtained through microscopy using 10x objectives. Subsequently, microscopic-level sampling is conducted using 40x objectives with the MorphoGlia tool.

Here, we are implementing the HDBSCAN algorithm on the images. To do this, it is necessary to convert the images into numpy arrays to utilize this clustering method, and then reconvert these arrays back into images while preserving pixel isometry.

This is an example utilizing simulated points generated with the NumPy library in Python.



![Simulation](https://github.com/Maya-Arteaga/Parcellation/assets/70504322/d0dc7ff9-bcda-4cf6-8822-9abf39ba6d32)





This example demonstrates its usage using a fluorescence image acquired through widefield microscopy. Note that the preprocessing of the images does not convert the image to binary; instead, it is left in grayscale to better delineate highly clustered regions.

![Parcellation](https://github.com/Maya-Arteaga/Parcellation/assets/70504322/23aedb26-2d8b-4fb0-865c-f52e34be6c98)
