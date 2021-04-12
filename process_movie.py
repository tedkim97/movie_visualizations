import sys
import argparse
import movie_vis as mv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a movie barcode using KMeans')
    parser.add_argument('-f', '--filepath', dest='file_path', action='store', required=True, type=str, help='Filepath to a movie file')
    parser.add_argument('-o', '--output_path', dest='output_path', action='store', required=True, type=str, help='Filepath to output image (png)')
    parser.add_argument('-s', '--sample_rate', dest='sample_rate', action='store', default=24, type=int, help='Sample rate for frames of the movie. Whole number greater than 0. Defaults to 24')
    parser.add_argument('-n', '--numcolors', dest='num_colors', action='store', required=True, type=int, help='Number of clusters used in the KMeans Visualization')
    parser.add_argument('-w', '--width', dest='slice_width', action='store', default=1, type=int, help='Slice width of each color extracted frame. Defaults to 1.')
    parser.add_argument('-m', '--height-multiplier', dest='slice_height_multiplier', action='store', default=2, type=int, help='Multiplier to increase slice height. Slice height is determined by calculating the least common multiple of numbers 1 through num_clusters. Defaults to 2')
    parser.add_argument('-c', '--cuda', dest='enable_cuda', action='store_true', help='Option to enable CUDA acceleration for KMeans. Requires cuML. Defaults to False')
    args = parser.parse_args()

    output = args.output_path if ('.png' in args.output_path) else '{}.png'.format(args.output_path)
    slice_height = mv.lcm_n([x for x in range(1, args.num_colors + 1)]) * args.slice_height_multiplier

    print('Starting Visualization With Parameters ({})'.format(args.file_path))
    print('Number of colors = {}'.format(args.num_colors))
    print('Sampling Rate = 1/{}'.format(args.sample_rate))
    print('Slice Width = {}, Slice Height = {}'.format(args.slice_width, slice_height))
    mv.process_kmean_vis(output, '', args.file_path, args.sample_rate, args.slice_width, slice_height, args.num_colors, cuda=args.enable_cuda)
    print('Done Processing (Results saved to {})'.format(output))
    sys.exit(0)