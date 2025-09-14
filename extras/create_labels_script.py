# Script for creating labels.csv
import os, csv

images_dir = 'data/images'
out_csv = 'data/labels.csv'
files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image','clean','damaged','group_id'])
    for fn in files:
        writer.writerow([fn,0,0,fn.split('.')[0]])
