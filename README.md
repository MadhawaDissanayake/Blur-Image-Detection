# Blur-Image-Detection
AI-powered wildlife photo sorter that uses YOLOv8 subject detection and Laplacian variance to automatically cull blurry images while preserving artistic bokeh.

Wildlife Focus-Sorter is a high-performance Python tool designed for photographers to automate the culling process. Unlike standard blur detectors, this tool uses Computer Vision (YOLOv8) to identify the subject (bird, animal, or person) and only evaluates sharpness within that specific region. This ensures that photos with intentional background blur (bokeh) are kept, while shots with missed focus are moved to a separate directory.
