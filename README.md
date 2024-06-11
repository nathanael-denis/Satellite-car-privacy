## **Car inpainting in satellite images**

This project is a tool used to *automatically detect cars in images*, and use the resulting segmentation so as to define masks for inpainting.

*Inpainting* is a process in which a missing part of an artwork is filled to present a complete image. It is now used for privacy purposes to fill areas where sensitive areas have been removed, without the possibility of detection to the human eye, in contrast with blur.

**Car detection**

Several works in the machine learning community already addressed the detection of cars on satellite imaging. The major drawback is the detection rate is highly dependent on the satellite used to take the images, i.e., resolution, zoom level, and instrumentation. It is also possible to detect cars with non-optical imagery, such as synthetic aperture radar (SAR)

For this proof of concept, we relied on an existing solution from R. Cole [1] for car and swimming pool detection on satellite images. An API allows to use a trained model from a Python script, which returns the detected cars and the corresponding boxes.

**Inpainting**

The bounding boxes are used to derive *a mask*, which corresponds to the area to be removed.  It is then possible to inpaint the corresponding area using open-source stable diffusion. The resulting images is forwarded to a new directory with privacy-preserving images without the cars.

The following image shows a off-nadir UAV capture from Google Earth, with several cars and unveiling house layout:
![Screenshot 2024-06-10 105028](https://github.com/nathanael-denis/Satellite-car-privacy/assets/43931834/d6efe85c-2fab-4ad5-87ba-42276f8867e6)

We proceed to inpaint the front house layout and remove the cars:
![US_house_off_nadir_pincel_app_highlights](https://github.com/nathanael-denis/Satellite-car-privacy/assets/43931834/25fa376e-a1ac-4631-82e0-ddaa3ac2147e)


**Wow, that's a lot of cars**

Inpainting is fine when two or three cars are in the picture but what can we expect from inpainting more than 10 cars. We did it:

* First, by applying inpainting for each car independently
* Second, by inpainting only once using the aggregated mask

The second solution is much better, as it requires only one iteration instead of `N`where `N` is the number of cars detected by the trained model.

![woomed_512](https://github.com/nathanael-denis/Satellite-car-privacy/assets/43931834/cd2deee8-a671-42ab-9606-b5469504bd91)

After inpainting, the image with the whole parking lot erased turn into:

![generated_image_1](https://github.com/nathanael-denis/Satellite-car-privacy/assets/43931834/042bb9ab-07b2-4f48-9e6a-3611403a4e53)


**Why remove cars in the first place ?**

A diversity of actors may be interested in spotting cars for very different reasons. Some instances are:

 1. The states for compliance with the law, e.g., check if a car is not in a prohibited parking spot. Past occurrences of such methods exist, notably the French tax agency spotting illegal swimming pools and issuing fines afterward [2]
 2. Burglars, in order to infer occupancy of the houses or spot the most interesting houses detecting the most expensive cars
 3. Insurance companies, to verify claims regarding parking habits.

Good data protection habits, as well as several data protection regulations (including GDPR) require data minimization. Removing privacy-sensitive objects from satellite images will likely become compulsory in the future unless the users' consent is clearly stated (it is not asked today) or the use case is related to cars in a demonstrable way (e.g., smart cities, urban planning...) 

**Bibliography**

[1] Abdul Mutalib. "CarsAndSwimmingPool Dataset." Roboflow Universe. Roboflow, August 2023. Accessed June 5, 2024. https://universe.roboflow.com/abdul-mutalib-wthjt/carsandswimmingpool.

[2] Kim Willsher. French tax officials use AI to spot 20,000  
undeclared pools.  The Guardian, Aug. 2022.  URL:  
https://www.theguardian.com/world/2022/aug/29/french-tax-officials-use-ai-to-spot-20000-undeclared-pools
