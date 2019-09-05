# UAV-Tree-Detection
Tree crown detection system

The deployment is intended in a Google Compute Engine, if an external IP is set, then you can run python app.py and run python app.py. Thorugh port :5000 you can acess a simple Flask Web App were you can upload a large UAV raster directly to Google Storage System. In the backend Mask-RCNN is implemented to detect and segment tree crows. After finishing the whole process, a direct download path is provided with a tree density raster and also a CSV file with the position lat/lon of every tree in the area of interest. You have to provide in the same folder pre-trained weights based on COCO dataset and an access keyfile to Google Storage system.
