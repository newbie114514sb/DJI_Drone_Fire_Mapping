from PIL import Image
import piexif
import pprint

img_path='examples/sample_hyperlapse/IMG_0000.jpg'
img=Image.open(img_path)
exif=img.info.get('exif')
print('EXIF data:')
# also inspect raw bytes for "Exif" marker
with open(img_path,'rb') as f:
    data=f.read()
print('raw length',len(data), 'has Exif?', b'Exif' in data)
print(data[:200])
if exif:
    pprint.pprint(piexif.load(exif))
else:
    print('No EXIF data')
