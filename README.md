## **Car inpainting in satellite images**

This project is a tool used to *automatically detect cars in images*, and use the resulting segmentation so as to define masks for inpainting.

*Inpainting* is a process in which a missing part of an artwork is filled to present a complete image. It is now used for privacy purposes to fill areas where sensitive areas have been removed, without the possibility of detection to the human eye, in contrast with blur.

**Car detection**

Several works in the machine learning community already addressed the detection of cars on satellite imaging. The major drawback is the detection rate is highly dependent on the satellite used to take the images, i.e., resolution, zoom level, instrumentation. It is also possible to detect cars with non-optical imagery, such as synthetic-aperture radar (SAR)

For this proof of concept, we relied on an existing solution from R. Cole [1] for car and swimming pool detection on satellite images. An API allows to use the trained model from a Python script, which returns the detected cars and the corresponding boxes.

**Inpaiting**

The bounding boxes are used to derive *a mask*, which corresponds to the area to be removed.  It is then possible to inpaint the corresponding area using open-source stable diffusion. The resulting images is forwarded to a new directory with privacy-preserving images without the cars.

**Why remove cars in the first place ?**

A diversity of actors may be interested in spotting cars for very different reasons. Some instances are:

 1. The states for compliance with the law, e.g., check if car are not in a prohibited parking spots. Past occurences of such methods exist, notably the French tax agency spotting illegal swimming pools and issuing fines afterwards [2]
 2. Burglars, in order to infer occupancy of the houses or spot the most interesting houses detecting the most expensive cars
 3. Insurance companies, to verify claims regarding parking habits.

Good data protection habits, as well as several data protection regulations (including GDPR) require data minimization. Removing privacy-sensitive objects from satellite images will likely become compulsory in the future unless the users' consent is clearly stated (it is not asked today) or the use case related to cars in a demonstrable way (e.g., smart cities, urban planning...) 

**Bibliography**

[1] Abdul Mutalib. "CarsAndSwimmingPool Dataset." Roboflow Universe. Roboflow, August 2023. Accessed June 5, 2024. https://universe.roboflow.com/abdul-mutalib-wthjt/carsandswimmingpool.

[2] Kim Willsher. French tax officials use AI to spot 20,000  
undeclared pools.  The Guardian, Aug. 2022.  URL:  
https://www.theguardian.com/world/2022/aug/29/french-tax-officials-use-ai-to-spot-20000-undeclared-pools
