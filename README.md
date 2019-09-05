# UAV-Tree-Detection

The deployment is intended in a Google Compute Engine, if an external IP is set for the virtual machine, then you can run python app.py. Thorugh port :5000 you can acess a simple Flask Web App.

You can upload a large UAV raster directly to Google Storage System. Then the system sends it directly to the Compute Engine. The systems crops fixed 256 x 256 windows and process the batch with GPU for maximum speed. 

Mask-RCNN is implemented to detect and segment tree crows. After finishing the whole batch, a direct download path is provided with a tree density raster and also a CSV file with the position lat/lon of every tree in the area of interest. 

You have to provide in the same folder pre-trained weights based on COCO dataset and an access keyfile to Google Storage system.
