# Sphere Extraction

This project extracts spheres from images and saves the estimated circle parameters of the spheres as pickle files.

## Installation

1. Clone and download the following models as per their instructions:
   - [SAM2 Model](https://github.com/facebookresearch/sam2)  
     Download the `sam2_hiera_large` config and checkpoint (links available on the repository).
   - [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)  

2. Ensure the models are properly set up before running the script.

3. Clone this repo, ```cd``` into it and install the required packages using ```pip install -r requirements.txt```. You will have to adjust the packages to your specific setup - maybe just install packages as you go.

## Usage

To run the extraction process, use the following command (adjust to your needs):

```bash
python sphere_mask_extraction.py --image-folder /path/to/image/directory --output-folder /path/to/output/directory --padding 0 --num-spheres 18 --restarts 5
```
