import fiona
import rasterio
import rasterio.mask
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import logging
from flask import Flask, session, request, render_template, redirect, url_for
from google.cloud import storage
import os, glob, sys
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
import numpy.ma as ma
import sys
import re
from scipy.spatial import distance
# Import Mask RCNN
import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisissupposedtobesecret!'

class TreesConfig(Config):
    # Give the configuration a recognizable name
    NAME = "trees"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 10

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + tree
    
    IMAGE_CHANNEL_COUNT = 3

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    

class InferenceConfig(TreesConfig):
    GPU_COUNT = 2
    IMAGES_PER_GPU = 10


def init():
    
    global model,graph
    # load the pre-trained Keras model
    model_path = '/tmp/'
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=model_path)

    # LOAD TRAINDED WEIGHTS
    print("Loading weights from ", model_path)
    model.load_weights('weights.h5', by_name=True)
    graph = tf.get_default_graph()


# Configure this environment variable via app.yaml
#CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
def raster_to_matrix(raster):
    mtx = []
    count = raster.RasterCount
    if count > 3:
        count = 3 
    
    for i in range(1, count+1):
        band = raster.GetRasterBand(i)
        mtx.append(band.ReadAsArray())
    if raster.RasterCount > 1 :
        mtx = np.stack(mtx, axis= -1)
    else: 
        mtx = np.array(mtx[0])
    
    mtx[mtx == mtx.max()] = 65535

    return mtx

def clip_raster(raster, shapefile_path):
    try:
        with fiona.open(shapefile_path, "r") as shapefile:
            features = [feature["geometry"] for feature in shapefile]
        
        
        with rasterio.open(raster) as src:
            out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
            out_meta = src.meta.copy()
        
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
        with rasterio.open(raster, "w", **out_meta) as dest:
            dest.write(out_image)

        return 'El raster fue cortado exitosamente'

    except:
        return 'El raster y el shapefile deben estar en la misma proyeccion'

def upload_files(csv_file, den_file):
    # Create a Cloud Storage client.
    client = storage.Client.from_service_account_json('ADL-forestal-segmentation-7dc429779824.json')   
    bucket = client.bucket('images-arauco-forestal')
    blob_csv = bucket.blob(csv_file)
    blob_den = bucket.blob(den_file)
    blob_csv.upload_from_filename('/tmp/'+csv_file)
    blob_den.upload_from_filename('/tmp/densidad.tif')
    blob_csv.make_public()
    blob_den.make_public()
    os.remove('/tmp/'+csv_file)
    os.remove('/tmp/densidad.tif')
    print(csv_file)
    print(den_file)
    url_csv = blob_csv.public_url
    url_den = blob_den.public_url
    return url_csv, url_den

def closest_node(node, nodes):
    if nodes:
        dist = distance.cdist([node], nodes).min()     
    else: 
        dist = 100
    return dist

def save_raster(name, leny, lenx, geotransform, density, projection):
    # Save a density map raster
    driver = gdal.GetDriverByName('GTiff')
    new_raster = driver.Create(name,leny+1, lenx+1, 1, gdal.GDT_UInt16)
    new_raster.SetGeoTransform(geotransform)  
    new_raster.GetRasterBand(1).WriteArray(density)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(projection)
    new_raster.SetProjection(outRasterSRS.ExportToWkt())
    new_raster.FlushCache()

def predictions(model, imagenes, height_list, width_list, param):
    
    # Parameters
    xOrigin, yOrigin, pixelWidth, pixelHeight, xf, yf, projection = param

   # Density mask parameters
    factor = 20
    rodal_length = 22.36
    numberpixelWidth = rodal_length/pixelWidth
    numberpixelHeight = rodal_length/pixelHeight
    lenx = int(xf/numberpixelWidth)
    leny = int(yf/numberpixelHeight)
    density = np.zeros((lenx+1,leny+1))

    # Dictionary
    info = {}
    centroids = []
    index =0
    copa_id = 0
    total = int(len(imagenes))     

    for i in range(int(len(imagenes)/20)):
    
        results = model.detect(imagenes[20*i:20*(i+1)], verbose=1)        
        print('porcentajes: ',20*(i+1)/total)
    
        for r in results:
    
            dv_height = height_list[index]
            dv_width = width_list[index]
            centroids = []
            for i, roi in enumerate(r['rois']):
                
                c_x, c_y = (roi[1] + roi[3])/2., (roi[0] + roi[2])/2.
                distance = closest_node((c_x, c_y), centroids)
                
                if c_x < 246 and c_y < 246 and distance > 10:
                    info[copa_id] = []
                    info[copa_id].append(c_x + dv_width)
                    info[copa_id].append(c_y + dv_height)
                    py = int((c_y + dv_height)/numberpixelHeight)
                    px = int((c_x + dv_width)/numberpixelWidth)
                    density[py,px] = density[py,px]+factor
                    centroids.append([c_x, c_y])

                    copa_id += 1

            index = index+1

    # Save a density map raster
    geotransform = [xOrigin, rodal_length, 0, yOrigin, 0, -rodal_length]
    save_raster('/tmp/densidad.tif', leny, lenx, geotransform, density, projection)

    return info

def preprocessing(image):      
    
    height = image.shape[0]
    width  = image.shape[1]
    dv_height = 0                           
    size = 256
    imagenes = []
    width_list = []
    height_list = []

    while dv_height + size < height:
        dv_width  = 0
        while dv_width + size < width:
            
            img = image[dv_height:dv_height+size, dv_width:dv_width+size, :]

            valid_pixels = np.count_nonzero(~np.isnan(img))
            total_pixels = size*size
            perc_pixels = valid_pixels/total_pixels
            
            if(perc_pixels > 0.5):
                imagenes.append(img)
                width_list.append(dv_width)
                height_list.append(dv_height)

            dv_width += int(size)
        dv_height += int(size)
          
    return imagenes, width_list, height_list

def pixel2coord(col, row, geotransform):

    c, a, b, f, d, e = geotransform

    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)



@app.route('/')
def index():
    return """
<form method="POST" enctype="multipart/form-data" action="/upload">
  <input type="file" name="file[]" multiple="">
  <input type="submit" value="add">
</form>
"""


@app.route('/upload', methods=['POST'])
def upload():
    """Process the uploaded file and upload it to Google Cloud Storage."""
    uploaded_files = request.files.getlist("file[]")
    client = storage.Client.from_service_account_json('ADL-forestal-segmentation-7dc429779824.json')    
    bucket = client.bucket('images-arauco-forestal')


    for uploaded_file in uploaded_files:
        
        blob = bucket.blob(uploaded_file.filename)
        blob.upload_from_string(uploaded_file.read(), content_type=uploaded_file.content_type)
        print(uploaded_file.filename[-3:])
        session[uploaded_file.filename[-3:]] = uploaded_file.filename
    
    if not 'tif' in session:
        return 'No raster uploaded.', 400
    
    id_predio = session['tif'][:-4]

    return redirect(url_for('prediction', predio=id_predio))

    


@app.route('/prediction/<predio>')
def prediction(predio):

    #region_name = predio[:-4]
    region_name = predio

    for key, value in session.items():
        with file_io.FileIO('gs://images-arauco-forestal/' + value, 'rb') as infile:
            with file_io.FileIO('/tmp/' + value, 'w') as outfile:
                outfile.write(infile.read())
    
    print(session)

    if ('shp' in session) and ('shx' in session) and ('dbf' in session):
        mensaje = clip_raster('/tmp/'+session['tif'], '/tmp/'+session['shp'])
	
 
    raster = gdal.Open('/tmp/'+session['tif'])

    geo_transform = raster.GetGeoTransform()
    matrix_raster = raster_to_matrix(raster)
    x,y,z = matrix_raster.shape

    param = [geo_transform[0], geo_transform[3], geo_transform[1], -geo_transform[5], x, y, raster.GetProjectionRef()]
    
    imagenes, width_list, height_list = preprocessing(matrix_raster)
    
    with graph.as_default():
        info = predictions(model, imagenes, height_list, width_list, param)

    df = pd.DataFrame.from_dict(info, orient='index', columns=['Centroide_x', 'Centroide_y'])

    trans = df[['Centroide_x', 'Centroide_y']].values
    coords = pixel2coord(trans[:,0], trans[:,1], geo_transform)
    df['Centroide_x'] = coords[0]
    df['Centroide_y'] = coords[1]

    file_csv = 'predicciones-'+region_name+'.csv'
    file_den = 'densidad-'+region_name+'.tif'
    df.to_csv('/tmp/'+file_csv)
    url1, url2 = upload_files(file_csv, file_den)

    return 'Region: '+ predio + ' density raster and csv file succesfully processed. Url de csv: '+url1+ ' -Url de raster: ' + url2


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    init()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
