from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import cv2
import warnings
import os

# Driver code to scale visualization dimensions
def gcd(a, b):
    if (b == 0):
        return a
    return gcd(b, a % b)

def lcm(a, b):
    return (a * b) / gcd(a,b)

def lcm_n(nums: list):
    return int(reduce(lcm, nums))

def downsample_vis(fname, sample_rate, slice_w, slice_h):
    """Produce a "downsample" visualization from a video file by iterating over
        the frames of the source.

    Refer to (https://redd.it/d7nw9p) or (https://redd.it/gc4mbi) to get an idea

    Parameters
    ----------

    fname : str
        File name (and path) of the video file. 
    
    sample_rate : int
        Sampling rate on which the frames are processed
    
    slice_w : int
        Width of each "slice" in the visualization
    
    slice_h : int
        Height of each "slice" in the visualization
    
    Returns
    -------
    numpy.ndarray
        This will return a tensor with dimensions (slice_h, W*, 3), where W* 
        is the calculated width of the visualization. The 3 represents the 
        colors (R,G,B) of the vis
    """
    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_slices = total_frames // sample_rate
    vis_output = np.zeros((slice_h, slice_w * num_slices, 3), dtype='uint8')
    
    for i in tqdm(range(total_frames)):
        ret = cap.grab()
        if (i % sample_rate) == 0:
            ret, frame = cap.retrieve()
            tmpf = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            tind = int(i // sample_rate)
            downsamp_img = cv2.resize(tmpf, (slice_w, slice_h), interpolation=cv2.INTER_LANCZOS4)
            vis_output[:,(tind * slice_w):((tind + 1) * slice_w),:] = downsamp_img
    
    cap.release()
    return vis_output

def color_avg_vis(fname, sample_rate, slice_w, slice_h):
    """Produce an "average" visualization from a video file by iterating over
        the frames of the source

    Refer to : (https://redd.it/d6l2d0) to get an idea

    Parameters
    ----------
    
    fname : str
        File name (and path) of the video file. 
    
    sample_rate : int
        Sampling rate on which the frames are processed
    
    slice_w : int
        Width of each "slice" in the visualization
    
    slice_h : int
        Height of each "slice" in the visualization
    
    Returns
    -------
    numpy.ndarray
    """
    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_slices = total_frames // sample_rate
    vis_output = np.zeros((slice_h, slice_w * num_slices, 3), dtype='uint8')
    
    for i in tqdm(range(total_frames)):
        ret = cap.grab()
        if (i % sample_rate) == 0:
            ret, frame = cap.retrieve()
            tmpf = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            avg_color = np.around(np.mean(tmpf.reshape((-1, 3)), axis=0))
            tind = int(i // sample_rate)
            vis_output[:,(tind * slice_w):((tind + 1) * slice_w),:] = avg_color
    
    cap.release()
    return vis_output

def prom_col_vis(fname, sample_rate, slice_w, slice_h, down_sample=0.25, num_clusters=8):
    """Produce a "most frequent" color visualization video file by iterating
        over the frames of the source and using a kmean's clustering algorithm

    DO NOT USE THIS FUNCTION - this is just for demonstration purposes
    """
    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_slices = total_frames // sample_rate
    
    vis_output = np.zeros((slice_h, slice_w * num_slices, 3), dtype='uint8')
    cluster_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=20)
    
    for i in tqdm(range(total_frames)):
        ret = cap.grab()
        if (i % sample_rate) == 0:
            ret, frame = cap.retrieve()
            temp_f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            reduc = cv2.resize(temp_f, None, fx=down_sample, fy=down_sample, interpolation=cv2.INTER_CUBIC)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cluster_model.fit(reduc.reshape((-1,3)))
            _, counts = np.unique(cluster_model.labels_, return_counts=True)
            max_ind = np.argmax(counts)
            prom_color = np.around(cluster_model.cluster_centers_[max_ind]).astype('uint8')
            tind = int(i // sample_rate)
            vis_output[:, (tind*slice_w):((tind+1)*slice_w),:] = prom_color 
    cap.release()
    return vis_output

def kmeans_prop(fname, sample_rate, slice_w, slice_h, down_sample=0.25, num_clusters=8):
    """Produce a "most frequent" color visualization video file by iterating
        over the frames of the source and using a kmean's clustering algorithm

    Parameters
    ----------
    fname : str
        File name (and path) of the video file. 
    
    sample_rate : int
        Sampling rate on which the frames are processed
    
    slice_w : int
        Width of each "slice" in the visualization
    
    slice_h : int
        Height of each "slice" in the visualization
    
    down_sample : float
        Ratio that resizes a frame with CV2's bicubic interpolation method. 
        EX: a (1920x1080x3) image becomes a (480x270x3) image after processing.

        NOTE: Resizing frames will decrease the quality of the frame for an 
        increase in speed. The decrease in the quality of the frame will result
        in slight shifts in the color. The default ratio of (0.25) seems to 
        preserve qualities well. 

        Bicubic interpolation was chosen because other visualizations used it,
        I have not done serious experimentation has been done with other 
        interpolation methods. When I compared the original and resized image, 
        it was "good enough" for me. There are discussions and comparisons 
        about the optimal interpolation method on the web. See:
        1) https://stackoverflow.com/questions/23853632/
        2) https://pillow.readthedocs.io/ (search interpolation)

    num_clusters : int
        The number of clusters to use in our KMeans clustering algorithm. You 
        can think of it as the number of "colors" to extract for each frame. 
        Increasing this parameter will increase the amount of time, it takes
        to process the whole movie.
        
        NOTE: You are not guaranteed to find the full number of clusters in
        each frame. The Scikit-learn library will give a warning that this 
        function suppresses. 

    Returns
    -------
    numpy.ndarray
    """
    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_slices = total_frames // sample_rate

    vis_output = np.zeros((slice_h, slice_w * num_slices, 3), dtype='uint8')
    cluster_model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=20)

    for i in tqdm(range(total_frames)):
        ret = cap.grab()
        if (i % sample_rate) == 0:
            ret, frame = cap.retrieve()
            temp_f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            reduc = cv2.resize(temp_f, None, fx=down_sample, fy=down_sample, interpolation=cv2.INTER_CUBIC)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cluster_model.fit(reduc.reshape((-1,3)))
                colors = np.around(cluster_model.cluster_centers_).astype('uint8')
            lbl, counts = np.unique(cluster_model.labels_, return_counts=True)
            cut = slice_h / np.sum(counts)
            ordering = np.argsort(counts)[::-1]
            tind = int(i // sample_rate)
            prev_ind = 0
            for i, val in enumerate(ordering):
                height = int(round(cut * counts[val]))
                l_ind = (tind * slice_w)
                r_ind = (tind + 1) * slice_w
                vis_output[prev_ind:prev_ind+height, l_ind:r_ind] = colors[val]
                prev_ind += height
            
    cap.release()
    return vis_output

def kmeans_prop_cuda(fname, sample_rate, slice_w, slice_h, down_sample=0.25, num_clusters=8):
    """Produce a "most frequent" color visualization video file by iterating
        over the frames of the source and using a kmean's clustering algorithm using cuda
    """

    from cuml.cluster import KMeans as KMeansCuda 

    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_slices = total_frames // sample_rate

    vis_output = np.zeros((slice_h, slice_w * num_slices, 3), dtype='uint8')
    cluster_model = KMeansCuda(n_clusters=num_clusters, init='scalable-k-means++', n_init=20)

    for i in tqdm(range(total_frames)):
        ret = cap.grab()
        if (i % sample_rate) == 0:
            ret, frame = cap.retrieve()
            temp_f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            reduc = cv2.resize(temp_f, None, fx=down_sample, fy=down_sample, interpolation=cv2.INTER_CUBIC).astype('float32')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cluster_model.fit(reduc.reshape((-1,3)))
                colors = np.around(cluster_model.cluster_centers_).astype('uint8')
            lbl, counts = np.unique(cluster_model.labels_, return_counts=True)
            cut = slice_h / np.sum(counts)
            ordering = np.argsort(counts)[::-1]
            tind = int(i // sample_rate)
            prev_ind = 0
            for i, val in enumerate(ordering):
                height = int(round(cut * counts[val]))
                l_ind = (tind * slice_w)
                r_ind = (tind + 1) * slice_w
                vis_output[prev_ind:prev_ind+height, l_ind:r_ind] = colors[val]
                prev_ind += height
            
    cap.release()
    return vis_output

def kmeans_prop_cuda_batch(fname, sample_rate, slice_w, slice_h, down_sample=0.25, num_clusters=8, batch_size=1):
    """Produce a "most frequent" color visualization video file by iterating
        over the frames of the source and using a kmean's clustering algorithm using cuda
    """

    from cuml.cluster import KMeans as KMeansCuda 

    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_slices = total_frames // sample_rate

    vis_output = np.zeros((slice_h, slice_w * num_slices, 3), dtype='uint8')
    cluster_model = KMeansCuda(n_clusters=num_clusters, init='scalable-k-means++', n_init=20)

    for i in tqdm(range(total_frames)):
        ret = cap.grab()
        if (i % sample_rate) == 0:
            ret, frame = cap.retrieve()
            temp_f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            reduc = cv2.resize(temp_f, None, fx=down_sample, fy=down_sample, interpolation=cv2.INTER_CUBIC).astype('float32')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cluster_model.fit(reduc.reshape((-1,3)))
                colors = np.around(cluster_model.cluster_centers_).astype('uint8')
            lbl, counts = np.unique(cluster_model.labels_, return_counts=True)
            cut = slice_h / np.sum(counts)
            ordering = np.argsort(counts)[::-1]
            tind = int(i // sample_rate)
            prev_ind = 0
            for i, val in enumerate(ordering):
                height = int(round(cut * counts[val]))
                l_ind = (tind * slice_w)
                r_ind = (tind + 1) * slice_w
                vis_output[prev_ind:prev_ind+height, l_ind:r_ind] = colors[val]
                prev_ind += height
            
    cap.release()
    return vis_output

def most_freq_col(fname):
    """Function to produce a "most frequent" visualization from an existing
    kmeans visualization.

    This will mimic the parameters of the original visualization (assuming 
    the image was saved correctly)
 
    Parameters
    ----------
    fname : str
        File name of the video file. 

    Returns
    -------
    numpy.ndarray
    """
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_RGB2BGR)
    vis_output = np.zeros(img.shape, dtype='uint8')
    prom_colors = img[0,:]
    vis_output[:,] = prom_colors
    return vis_output

def least_freq_col(fname):
    """Function to produce a "least frequent" visualization from an existing
    kmeans visualization

    TODO: This function and [most_freq_col] can be refactored, but I'll keep 
    them seperate for now
    
    Parameters
    ----------
    fname : str
        File name of the KMeans visualization produced by [kmeans_prop] 

    Returns
    -------
    numpy.ndarray
    """
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_RGB2BGR)
    vis_output = np.zeros(img.shape, dtype='uint8')
    prom_colors = img[-1,:]
    vis_output[:,] = prom_colors
    return vis_output

def kcolors(fname):
    """Function to output the colors of each slice in the visualization, in 
    equal magnitude on the y-axis. 
    
    Parameters
    ----------
    fname : str
        The file name of the video file 

    Returns
    -------
    numpy.ndarray
        This will return a tensor with dimensions (slice_h, W*, 3), where W* 
        is the calculated width of the visualization. The 3 represents the 
        colors (R,G,B) of the vis
    """
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_RGB2BGR)
    h, w, d = img.shape
    vis_output = np.zeros(img.shape, dtype='uint8')
    for ind in range(w):
        cols = np.unique(img[:,ind], axis=0)
        hslice = h // len(cols)
        for hind, color in enumerate(cols):
            vis_output[hind*hslice:(hind+1)*hslice,ind] = color
    return vis_output

def produce_vis(vis_func, str_format, nickname, fname, framerate, slice_w, slice_h):
    """Compute and writes to disk a visualization given a visualization process,
    naming format, nickname, video file name, etc.

    Parameters
    ----------

    vis_func : func 
        Visualization function

    str_format : str
        A generic string format that will be completed with the nickname param

    nickname : str
        The specific "name" or label used for the visualization

    Returns
    -------
    bool
        Returns True if the function completes with no error. 

    """
    vis_out = vis_func(fname, framerate, slice_w, slice_h)
    im = Image.fromarray(vis_out)
    im.save(str_format.format(nickname))
    print("done processing:", fname)
    return True

# TODO: decide to keep or delete the str_format & nickname for image naming.
# It was an artifact when I ran visualization in batches
def process_kmean_vis(str_format, nickname, fname, framerate, slice_w, slice_h, num_col, cuda=False):
    """Computes and writes to disk a KMeans visualization from the function
    [kmeans_prop] with the parameters above.

    Parameters
    ----------

    str_format : str
        A generic string format that will be completed with the nickname param

    nickname : str
        The specific "name" or label used for the visualization

    Returns
    -------
    bool
        Returns True if the function completes with no error.

    """
    if (cuda):
        vis = kmeans_prop_cuda(fname, framerate, slice_w, slice_h, down_sample=0.25, num_clusters=num_col)
    else:
        vis = kmeans_prop(fname, framerate, slice_w, slice_h, down_sample=0.25, num_clusters=num_col)
    im = Image.fromarray(vis)
    im.save(str_format.format(nickname))
    print("done processing:", fname)
    return True

def load_image(fname):
    """Returns an image as a matplotlib compatabile numpy array with cv2

    TODO: I'll probably delete this because it's pretty redundant and only useful
    in jupyter notebooks
    """
    return cv2.cvtColor(cv2.imread(fname), cv2.COLOR_RGB2BGR)

def annotate_vis(outf: str, img: np.ndarray, **kwargs):
    """Takes a visuzalization numpy array, annotates with matplotlib, and saves
    the visualization

    Parameters
    ----------

    outf : str
        Output file name
    
    img : numpy.ndarray
        Visualization loaded as a numpy array

    **kwargs : dict
        A dictionary of (matplotlib) parameters to title the visualization
        and customize various parameters

    Returns
    -------
    None
    """
    plt.figure(figsize=kwargs['fig_size'])
    plt.title(kwargs['title'], fontsize=kwargs['title_font_size'], pad=15)
    plt.yticks(kwargs['yt'], kwargs['ytl'], fontsize=kwargs['font_size'])
    plt.xticks(kwargs['xt'], kwargs['xtl'], fontsize=kwargs['font_size'])
    plt.ylabel(kwargs['ylabel'], fontsize=kwargs['font_size'], labelpad=15)
    plt.xlabel(kwargs['xlabel'], fontsize=kwargs['font_size'], labelpad=15)
    plt.imshow(img)
    plt.savefig(outf)
    plt.show()

if __name__ == '__main__':
    # Driver Code
    movie_file_path = '' # some path to an mp4 file
    num_colors = 7 # for KMeans extractions
    slice_width = 1 # in pixels
    slice_height_multiplier = 3 # to increase visual clarity
    # By locking the height to the least common multiple of numbers [1, 8+1]
    # we can guarentee that the visualization will fit all the colors without
    # any off-by-one errors for all clusters <= 8
    slice_height = lcm_n([x for x in range(1, num_colors + 1)]) * slice_height_multiplier

    process_kmean_vis('visualization_image_{}.png', 'moviename', movie_file_path, 24, slice_width, slice_height, num_colors, cuda=False)
    
    print('done')
