"""Download YOLO model manually."""
import requests
import os
from tqdm import tqdm

url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'
output_file = 'yolov8n.pt'

print('Downloading YOLOv8n model...')
print(f'URL: {url}')

# Disable SSL verification to avoid certificate issues
response = requests.get(url, stream=True, verify=False)
total_size = int(response.headers.get('content-length', 0))

with open(output_file, 'wb') as f:
    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

file_size = os.path.getsize(output_file) / (1024 * 1024)
print(f'\nDownloaded successfully: {file_size:.1f} MB')
print(f'Saved to: {output_file}')
