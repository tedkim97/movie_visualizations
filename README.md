# Movie Visualizations
This repository contains a new movie barcode visualization that uses a clustering algorithm to extract more accurate representations of colors in a movie. This new visualization shows the relative presence of the different colors extracted for a user-defined number of clusters. In addition, this repository contains implementations of other popular movie visualizations, test images, and notebooks for demonstrative reasons.  

### KMeans Extracted Visualization (with proportional representation)
![visualizations1](final_figures/kmeans_color/k5/coco.png?raw=true)
![visualizations2](final_figures/kmeans_color/k5/madmax.png?raw=true)
![visualizations3](final_figures/kmeans_color/k5/spiderman.png?raw=true)

### KMeans Extracted Visualization (with equal representation)
![visualizations4](final_figures/kcolor/coco.png?raw=true)

### Most Frequent Color Visualization (derived from our KMeans Extracted Vis)
![visualizations5](final_figures/most_freq/simpson.png?raw=true)

### Least Frequent Color Visualization (derived from our KMeans Extracted Vis)
![visualizations6](final_figures/least_freq/spiderman.png?raw=true)

# Instructions
Depending on the type of visualization you want, you can run the bottom. `kmeans_prop` is the function that generates a KMeans Extracted Visualization (with proportional representation).

```python
from movie_vis import color_avg_vis, kmeans_prop
from PIL import Image

videofile = 'file_name_of_movie'
samplerate = 24 
sw = 1
sh = 500

avg_vis = color_avg_vis(videofile, samplerate, sw, sh)
avg_barcode = Image.fromarray(avg_vis)
avg_barcode.save('outputfilename')

number_of_clusters = 5
down_sampling_rate = 0.25
kmean_vis = kmean_prop(videofile, samplerate, sw, sh, 
						down_sample=down_sampling_rate, 
						num_clusters=number_of_clusters)
kmean_barcode = Image.fromarray(kmean_vis)
kmean_barcode.save('different_outputfilename')
```

## New Feature: CLI 
Instead of modifying the code in `movie_vis.py`. You can use the `process_movie.py` script in order to generate the same barcode. For example the example below generates a barcode at `movie_kmeans_vis.png` with `some_movie.mp4` using a sample rate of `1/24`, `5` clusters, and a height multiplier of `4`. 
```bash
python3 process_movie.py -f "some_movie.mp4" -o "movie_kmeans_vis" -s 24 -n 5 -m 4
```
Full arguments here:
```bash
optional arguments:
  -h, --help            show this help message and exit
  -f FILE_PATH, --filepath FILE_PATH
                        Filepath to a movie file
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Filepath to output image (png)
  -s SAMPLE_RATE, --sample_rate SAMPLE_RATE
                        Sample rate for frames of the movie. Whole number
                        greater than 0. Defaults to 24
  -n NUM_COLORS, --numcolors NUM_COLORS
                        Number of clusters used in the KMeans Visualization
  -w SLICE_WIDTH, --width SLICE_WIDTH
                        Slice width of each color extracted frame. Defaults to
                        1.
  -m SLICE_HEIGHT_MULTIPLIER, --height-multiplier SLICE_HEIGHT_MULTIPLIER
                        Multiplier to increase slice height. Slice height is
                        determined by calculating the least common multiple of
                        numbers 1 through num_clusters. Defaults to 2
  -c, --cuda            Option to enable CUDA acceleration for KMeans.
                        Requires cuML. Defaults to False
```

## New Feature: Minor CUDA acceleration
Leveraging the KMeans implementation - there is a \~1.6x-1.9x speed up of the visualization process. Either use with `cuda=True` or using the `-c/--cuda` flag in `process_movie.py` CLI 



##### NOTE: Some visualizations (`most_freq_col`, `least_freq_col`, `kcolors`) require an already existing KMeans barcode file in order to be created (to reduce redundant compute). 

Running the code would be like 
```python
from movie_vis import most_freq_col, least_freq_col, kcolors
from PIL import Image

kmeans_vis = 'file_name_of_kmean_prop_output'

img1 = Image.fromarray(most_freq_col(kmeans_vis))
img1.save('somefilename')

img2 = Image.fromarray(least_freq_col(kmeans_vis))
img2.save('somefilename')

img3 = Image.fromarray(kcolors(kmeans_vis))
img3.save('somefilename')
```

##### Figure annotations can be done, but it only wraps existing matplotlib customization. You don't need this function to annotate
```python
from movie_vis import annotate_vis

fname = 'file_name_of_visualization_barcode'
vis = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_RGB2BGR)
h, w, d = vis.shape

plot_param = {'fig_size': (20,10),
            'font_size': 15,
            'title_font_size': 30,
            'title': 'PLOT TITLE OF {}'.format('SOMENAME'),
            'yt': [],
            'ytl': None,
            'xt': np.arange(0, w, 300),
            'xtl': ['{}'.format(x//60) for x in np.arange(0, w, 300)],
            'xlabel': 'Timing of frame (minutes)',
            'ylabel': None
         	}

annotate_vis('annotated_output_fname', vis, **plot_param)
```

##### If you don't want to use PILLOW to save images
```python
import cv2
cv2.imwrite('outputfilename', numpy_array)
```

### Dependencies: 
- python3 >= 3.7
- opencv2
- scikit-learn
- numpy
- matplotlib
- tqdm (not required but will require small tweaks to the code)
- Pillow (you could replace all instances of PIL.Image.open with cv2)
- cuML >= 0.18 (if you plan to use the CUDA functionality - requires you to install using conda)