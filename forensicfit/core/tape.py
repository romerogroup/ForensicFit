# -*- Coding: utf-8 -*-
"""
Created on Sun Jun 28 14:11:02 2020

@author: Pedram Tavadze
"""
import warnings
import numpy.typing as npt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
from ..utils import image_tools
from . import Material, Analyzer
from . import HAS_OPENCV
if not HAS_OPENCV:
    print("To enable analyzer please install opencv")
else:
    import cv2

class Tape(Material):
    def __init__(self,
                 image: npt.ArrayLike,
                 label: str = None,
                 surface: str = None,
                 stretched: bool=False,
                 **kwargs):
        """Tape is a class created for tape images to be preprocessed for 
        Machine Learning. This Class detects the edges, auto crops the image 
        and returns the results in 3 different method coordinate_based, 
        bin_based and max_contrast. 
        """
        assert type(image) is np.ndarray, "image arg must be a numpy array"
        assert image.ndim in [2, 3] , "image array must be 2 dimensional"
        super().__init__(image, **kwargs)
        self.metadata['flip_h'] = False
        self.metadata['flip_v'] = False
        self.metadata['split_v'] = {
            "side": None, "pixel_index": None}
        self.metadata['label'] = label
        self.metadata['material'] = 'tape'
        self.metadata['surface'] = surface
        self.metadata['stretched'] = stretched

    def split_v(self, side, pixel_index=None):
        self.values['original_image'] = self.image.copy()
        if pixel_index is None:
            tape_analyzer = TapeAnalyzer(self)
            x = tape_analyzer.boundary[:, 0]
            pixel_index = int((x.max()-x.min())/2+x.min())
        self.image = image_tools.split_v(
            self.image, pixel_index, side)
        self.values['image'] = self.image
        self.metadata['split_v'] = {
            "side": side, "pixel_index": pixel_index}
        self.metadata['resolution'] = self.shape
        



class TapeAnalyzer(Analyzer):
    def __init__(self,
                 tape: Tape = None,
                 mask_threshold: int=60,
                 gaussian_blur: tuple=(15, 15),
                 n_divisions: int=6,
                 auto_crop: bool=False,
                 correct_tilt: bool=True,
                 padding: str='tape',
                 remove_background: bool=True):
        """
        

        Parameters
        ----------
        tape : TYPE, optional
            DESCRIPTION. The default is None.
        mask_threshold : TYPE, optional
            DESCRIPTION. The default is 60.
        gaussian_blur : TYPE, optional
            DESCRIPTION. The default is (15, 15).
        n_divisions : TYPE, optional
            DESCRIPTION. The default is 6.
        auto_crop : TYPE, optional
            DESCRIPTION. The default is True.
        calculate_tilt : TYPE, optional
            DESCRIPTION. The default is True.
        verbose : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        super().__init__()
        self.metadata['n_divisions'] = n_divisions
        self.metadata['gaussian_blur'] = gaussian_blur
        self.metadata['mask_threshold'] = int(mask_threshold)
        self.metadata['remove_background'] = remove_background
        self.metadata['padding'] = padding
        self.metadata["analysis"] = {}
        if tape is not None:
            self.image = tape.image
            self.metadata += tape.metadata
            self.metadata['resolution'] = self.image.shape
            self.preprocess()
            self.metadata['cropped'] = auto_crop
            self.metadata['tilt_corrected'] = correct_tilt
            if correct_tilt:
                angle = self.get_image_tilt()
                self.image = image_tools.rotate_image(self.image, angle)
                self.metadata.tilt_corrected = True
            if auto_crop:
                self.get_image_tilt()
                self.auto_crop_y()
                self.metadata.cropped = True
            if correct_tilt or auto_crop:
                self.preprocess()
            if remove_background:
                self.image = self['masked']
            
        return

    def preprocess(self):
        """
        
        Parameters
        ----------
        calculate_tilt : TYPE, optional
            DESCRIPTION. The default is True.
        auto_crop : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        image = self.image
        # if self.metadata.gaussian_blur is not None:
        #     image = image_tools.gaussian_blur(
        #         self.image, window=self.metadata.gaussian_blur)
        gray = image_tools.to_gray(image, mode='SD')
        image = image_tools.gaussian_blur(
                gray, window=self.metadata.gaussian_blur)
        contours = image_tools.contours(image,
                                        self.metadata.mask_threshold)
        largest_contour = image_tools.largest_contour(contours)
        self.metadata['boundary'] = largest_contour.reshape(-1, 2)
            
    def flip_v(self):
        self.image = np.fliplr(self.image)
        self.metadata['boundary'] = np.array(self.metadata['boundary'])
        self.metadata['boundary'][:, 0] = self.metadata['boundary'][:, 0]*-1 + self.image.shape[1]
        if 'coordinate_based' in self.metadata['analysis']:
            coords  = np.array(self.metadata['analysis']['coordinate_based']['coordinates'])
            slopes  = np.array(self.metadata['analysis']['coordinate_based']['slopes'])
            coords[:, 0] *= -1 
            coords[:, 0] += self.image.shape[1]
            slopes[:, 1] += slopes[:, 0]*self.image.shape[1]
            slopes[:, 0] *= -1
            # slopes[:, 1] += self.image.shape[1]
            self.metadata['analysis']['coordinate_based']['coordinates'] = coords
            self.metadata['analysis']['coordinate_based']['slopes'] = slopes
        if 'bin_based' in self.metadata['analysis']:
            dynamic_positions = np.array(self.metadata['analysis']['bin_based']['dynamic_positions'])
            for i, pos in enumerate(dynamic_positions):
                for ix in range(2):
                    dynamic_positions[i][0][ix] = pos[0][ix]*-1 + self.image.shape[1]
            dynamic_positions[:, 0, [0, 1]] = dynamic_positions[:, 0, [1, 0]]
            
            self.metadata['analysis']['bin_based']['dynamic_positions'] = dynamic_positions
        self.metadata['flip_v'] =  not(self.metadata['flip_v'])

    def flip_h(self):
        # TODO coordinate based
        self.image = np.flipud(self.image)
        self.metadata['boundary'] = np.array(self.metadata['boundary'])
        self.metadata['boundary'][:, 1] = self.metadata['boundary'][:, 1]*-1 + self.image.shape[0]
        if 'coordinate_based' in self.metadata['analysis']:
            coords = np.array(self.metadata['analysis']['coordinate_based']['data'])
            coords[:, 1] = np.flipud(coords[:, 1])
            self.metadata['analysis']['coordinate_based']['data'] = coords
        if 'bin_based' in self.metadata['analysis']:
            dynamic_positions = np.array(self.metadata['analysis']['bin_based']['dynamic_positions'])
            for i, pos in enumerate(dynamic_positions):
                for ix in range(2):
                    dynamic_positions[i, 1, ix] = pos[1, ix]*-1 + self.image.shape[0]
            dynamic_positions[:, 1, [0, 1]] = dynamic_positions[:, 1, [1, 0]]
            dynamic_positions = np.flipud(dynamic_positions)
            self.metadata['analysis']['bin_based']['dynamic_positions'] = dynamic_positions
        self.metadata['flip_h'] = not(self.metadata['flip_h'])
        
    def load_metadata(self):
        """
        

        Returns
        -------
        None.

        """
        self.metadata["xmin"] = int(self.xmin)
        self.metadata["xmax"] = int(self.xmax)
        self.metadata["x_interval"] = int(self.x_interval)
        self.metadata["ymin"] = int(self.ymin)
        self.metadata["ymax"] = int(self.ymax)
        self.metadata["y_interval"] = int(self.ymax-self.ymin)
        return 

    @classmethod
    def from_dict(cls, image, metadata):
        """
        
        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        values : TYPE
            DESCRIPTION.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if image is None:
            raise Exception(
                "The provided image was empty. Maybe change the query criteria")
        cls = cls()
        cls.image = image
        cls.metadata.update(metadata)
        return cls

    def get_image_tilt(self, plot: bool=False):
        """
        

        Parameters
        ----------
        plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        angle_d : TYPE
            DESCRIPTION.

        """
        figsize=(16, 9)
        stds = np.ones((self.metadata.n_divisions-2, 2))*1000
        conditions_top = []
        # conditions_top.append([])
        conditions_bottom = []
        # conditions_bottom.append([])
        
        boundary = self.boundary
        y_min = self.ymin
        y_max = self.ymax

        x_min = self.xmin
        x_max = self.xmax
        m_top = []
        # m_top.append(None)
        m_bottom = []
        # m_bottom.append(None)
        if plot:
            _ = plt.figure(figsize=figsize)
            ax = plt.subplot(111)
            self.plot_boundary(color='black', ax=ax)
            ax.invert_yaxis()
        for idivision in range(self.metadata.n_divisions-2):
            
            y_interval = y_max-y_min
            cond1 = boundary[:, 1] > y_max-y_interval/self.metadata.n_divisions
            cond2 = boundary[:, 1] < y_max+y_interval/self.metadata.n_divisions

            x_interval = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interval/self.metadata.n_divisions*(idivision+1)
            cond4 = boundary[:, 0] <= x_min + \
                x_interval/self.metadata.n_divisions*(idivision+2)

            cond_12 = np.bitwise_and(cond1, cond2)
            cond_34 = np.bitwise_and(cond3, cond4)
            cond_and_top = np.bitwise_and(cond_12, cond_34)

            # This part is to rotate the images

            if sum(cond_and_top) == 0:
                conditions_top.append([])
                m_top.append(None)
                continue
            if plot:
                ax.plot(boundary[cond_and_top][:, 0],
                         boundary[cond_and_top][:, 1], linewidth=3)
            m_top.append(np.polyfit(
                boundary[cond_and_top][:, 0], boundary[cond_and_top][:, 1], 1)[0])
            std_top = np.std(boundary[cond_and_top][:, 1])
            stds[idivision, 0] = std_top
            conditions_top.append(cond_and_top)

        for idivision in range(self.metadata.n_divisions-2):

            cond1 = boundary[:, 1] > y_min-y_interval/self.metadata.n_divisions
            cond2 = boundary[:, 1] < y_min+y_interval/self.metadata.n_divisions

            x_interval = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interval/self.metadata.n_divisions*(idivision+1)
            cond4 = boundary[:, 0] <= x_min + \
                x_interval/self.metadata.n_divisions*(idivision+2)

            cond_12 = np.bitwise_and(cond1, cond2)
            cond_34 = np.bitwise_and(cond3, cond4)
            cond_and_bottom = np.bitwise_and(cond_12, cond_34)
            if sum(cond_and_bottom) == 0:
                conditions_bottom.append([])
                m_bottom.append(None)
                continue
            if plot:
                ax.plot(
                    boundary[cond_and_bottom][:, 0], boundary[cond_and_bottom][:, 1], linewidth=3)
                ax.set_xlim(0, self.image.shape[1])

            m_bottom.append(np.polyfit(
                boundary[cond_and_bottom][:, 0], boundary[cond_and_bottom][:, 1], 1)[0])

            std_bottom = np.std(boundary[cond_and_bottom][:, 1])

            stds[idivision, 1] = std_bottom
            conditions_bottom.append(cond_and_bottom)
        arg_mins = np.argmin(stds, axis=0)
        m_top[arg_mins[0]] = m_bottom[arg_mins[1]] if m_top[arg_mins[0]] is None else m_top[arg_mins[0]]
        m_bottom[arg_mins[1]] = m_top[arg_mins[0]] if m_bottom[arg_mins[1]] is None else m_bottom[arg_mins[1]]
        m = np.average([m_top[arg_mins[0]], m_bottom[arg_mins[1]]])
        cond_and_top = conditions_top[arg_mins[0]]
        cond_and_bottom = conditions_bottom[arg_mins[1]]
        if plot:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.plot(boundary[:, 0], boundary[:, 1], color='black')
            ax.scatter(boundary[cond_and_top][:, 0],
                        boundary[cond_and_top][:, 1], color='blue')
            ax.scatter(boundary[cond_and_bottom][:, 0],
                        boundary[cond_and_bottom][:, 1], color='red')
            ax.set_xlim(0, self.image.shape[1])
            ax.invert_yaxis()
            
        top = boundary[cond_and_top][:, 1]
        bottom = boundary[cond_and_bottom][:, 1]
        self.metadata.crop_y_top = np.min(top) if len(top) != 0 else self.ymax
        self.metadata.crop_y_bottom = np.max(bottom) if len(bottom) !=0 else self.ymin
        if np.min(stds, axis=0)[0] > 10:
            self.metadata.crop_y_top = y_max
        if np.min(stds, axis=0)[1] > 10:
            self.metadata.crop_y_bottom = y_min
            
        angle = np.arctan(m)
        angle_d = np.rad2deg(angle)
        self.metadata.image_tilt = angle_d
        return angle_d

    @property
    def xmin(self):
        """
        X coordinate of minimum pixel of the boundary

        Returns
        -------
        xmin : int
            X coordinate of minimum pixel of the boundary

        """
        xmin = self.boundary[:, 0].min()
        return xmin

    @property
    def xmax(self):
        """
        X coordinate of minimum pixel of the boundary

        Returns
        -------
        xmax : int
            X coordinate of minimum pixel of the boundary

        """
        xmax = self.boundary[:, 0].max()
        return xmax

    @property
    def x_interval(self):
        """
        interval of the coordinates in X direction

        Returns
        -------
        x_interval : int
            interval of the coordinates in X direction

        """
        return self.xmax - self.xmin

    @property
    def ymin(self):
        """
        Y coordinate of minimum pixel of the boundary

        Returns
        -------
        ymin : int
             Y coordinate of minimum pixel of the boundary

        """
        ymin = self.boundary[:, 1].min()
        return ymin

    @property
    def ymax(self):
        """
        Y coordinate of maximum pixel of the boundary

        Returns
        -------
        ymax : int
             Y coordinate of maximum pixel of the boundary

        """
        ymax = self.boundary[:, 1].max()
        return ymax

    def auto_crop_y(self):
        """
        This method automatically crops the image in y direction (top and bottom)


        Parameters
        ----------
        calculate_tilt : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.image = self.image[int(
            self.metadata.crop_y_bottom):int(self.metadata.crop_y_top), :]

    def _get_points_from_boundary(self,
                                  x_0: float,
                                  x_1: float,
                                  y_0: float,
                                  y_1: float) -> npt.ArrayLike:
        coordinates = self['boundary']
        cond_1 = coordinates[:, 0] >= x_0
        cond_2 = coordinates[:, 0] <= x_1
        cond_x = np.bitwise_and(cond_1, cond_2)
        cond_1 = coordinates[:, 1] >= y_0
        cond_2 = coordinates[:, 1] <= y_1
        cond_y = np.bitwise_and(cond_1, cond_2)
        cond = np.bitwise_and(cond_x, cond_y)
        return coordinates[cond]


    def get_coordinate_based(self,
                             n_points: int=64,
                             x_trim_param: int=6) -> None:
        """
        This method returns the data of the detected edge as a set of points 
        in 2d plain with (x,y)

        Parameters
        ----------
        n_points : TYPE, optional
            DESCRIPTION. The default is 1024.
        x_trim_param : TYPE, optional
            DESCRIPTION. The default is 2.
        plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        was_flipped = self.metadata['flip_v']
        if was_flipped:
            self.flip_v()
        x_min = self.xmin
        x_max = self.xmax
        x_interval = x_max-x_min
        boundary = self.boundary

        # cond_12 = np.bitwise_and(cond1, cond2)
        x_0 = x_min
        x_1 = x_min+x_interval/x_trim_param*0.95
        cond1 = boundary[:, 0] >= x_0
        cond2 = boundary[:, 0] <= x_1
        cond_12 = np.bitwise_and(cond1, cond2)
        # cond_12 = cond1
        data = np.zeros((n_points, 2))
        stds = np.zeros((n_points,))
        slopes = np.zeros((n_points, 2))
        y_min = self.ymin
        y_max = self.ymax
        y_interval = y_max - y_min
        edge = boundary[cond_12]
        self.metadata['edge_x_std'] = float(np.std(edge[:, 0]))
        self.metadata['edge_x_range'] = int(edge[:, 0].max()-edge[:, 0].min())
        
        for i_point in range(0, n_points):
            y_start = y_min+i_point*(y_interval/n_points)
            y_end = y_min+(i_point+1)*(y_interval/n_points)
            points = self._get_points_from_boundary(x_0, x_1, y_start, y_end)
            if len(points) == 0 and x_trim_param != 1:
                if was_flipped:
                    self.flip_v()
                return self.get_coordinate_based(n_points=n_points, 
                                                 x_trim_param=x_trim_param-1)
            data[i_point, :2] = np.average(points, axis=0)
            stds[i_point] = np.std(points[:, 0])
            slopes[i_point, :] = np.polyfit(
                x=points[:, 0], 
                y=points[:, 1], 
                deg=1, 
                full=True)[0]
        self.metadata['analysis']['coordinate_based'] = {
            "n_points": n_points, 
            "x_trim_param": x_trim_param, 
            'coordinates': data,
            'stds': stds,
            'slopes': slopes}
        if was_flipped:
            self.flip_v()
        return 

    def get_bin_based(self,
                      window_background=50,
                      window_tape=1000,
                      dynamic_window=True,
                      size=None,
                      n_bins=10,
                      overlap=0,
                      border: str = 'avg',
                      ):
        """
        This method returns the detected edge as a set of croped images from 
        the edge. The number if images is defined by n_bins. The goal is 
        to try to match the segmentation to the wefts of the tape. 
        In the future this method will try to detecte the wefts automatically.


        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each segment in the backgroung 
            side of the image. The default is 20.
        window_tape : int, optional
            Number of pixels to be included in each segment in the tape side 
            of the image. The default is 300.
        dynamic_window : TYPE, optional
            Whether the windows move in each segment. The default is False.
        size : 2d tuple integers, optional
            The size of each segment in pixels. The default is (300,30).
        n_bins : int, optional
            Number of segments that the tape is going to divided into. The 
            default is 4.
        plot : bool, optional
            Whether or not to plot the divided image. The default is False.


        Returns
        -------
        An array of 2d numpy arrays with images of each segment.

        """
        was_flipped = self.metadata['flip_v']
        if was_flipped:
            self.flip_v()
        overlap = abs(overlap)
        boundary = self.boundary
        x_min = self.xmin
        x_max = self.xmax

        y_min = self.ymin
        y_max = self.ymax

        x_start = x_min-window_background
        x_end = min(x_min+window_tape, self.image.shape[1])

        seg_len = (y_max-y_min)//(n_bins)
        seg_len = (y_max-y_min)/(n_bins)

        segments = []
        dynamic_positions = []
        y_end = y_min
        for iseg in range(0, n_bins):
            y_start = y_end
            y_end = math.ceil(y_min+(iseg+1)*seg_len)
            if dynamic_window:
                cond1 = boundary[:, 1] >= y_start - overlap
                cond2 = boundary[:, 1] <= y_end + overlap
                cond_and = np.bitwise_and(cond1, cond2)
                x_slice = boundary[cond_and, 0]
                if border == 'min' or iseg in [0, n_bins-1]:
                    loc = x_slice.min()
                else:
                    cond_x = x_slice.min() + 500
                    loc = np.average(x_slice[x_slice < cond_x])
                x_start =  loc - window_background
                x_end = loc + window_tape
                if self.image.shape[1] < x_end:
                    diff = self.image.shape[1] - x_end
                    x_start += diff
                    x_end = self.image.shape[1]
            
            ys = y_start - overlap
            ye = y_end + overlap
            
            dynamic_positions.append(
                [[int(x_start), int(x_end)], [int(ys), int(ye)]])

        metadata = {"dynamic_positions": dynamic_positions,
                    "n_bins": n_bins,
                    "overlap": overlap,
                    "dynamic_window": dynamic_window,
                    "window_background": window_background,
                    "window_tape": window_tape,
                    "size": size}

        self.bin_based = segments
        self.metadata['analysis']['bin_based'] = metadata
        if was_flipped:
            self.flip_v()
        return dynamic_positions

    def get_max_contrast(self,
                         window_background: int=100,
                         window_tape: int=600):
        """
        This method returns the detected image as a black image with only the 
        boundary being white.

        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each segment in the background side of the image. The default is 20.
        window_tape : int, optional
            Number of pixels to be included in each segment in the tape side of the image. The default is 300.
        plot : bool, optional
            Whether or not to plot the divided image. The default is False.

        Returns
        -------
        edge_bw : 2d numpy array
            A black image with all the pixels on the edge white.

        """
        
        self.metadata['analysis']['max_contrast'] = {
            "window_background": window_background,
            "window_tape": window_tape}
        return 

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, x):
        if x == 'image':
            return self.image
        elif x == 'original_image':
            return self.original_image
        elif x == 'masked':
            return image_tools.remove_background(self.image, 
                                                 self.largest_contour)
        elif x == 'gray_scale':
            return image_tools.to_gray(self.image)
        elif x == 'binarized':
            return image_tools.binerized_mask(
                self['image'], self['masked'])
        elif x == 'largest_contour':
            return self.boundary.reshape(-1, 1, 2)
        elif x == 'boundary':
            return np.asarray(self.metadata['boundary'])
        elif x == 'coordinate_based':
            return self.metadata[
                'analysis']['coordinate_based']
        elif x == 'edge_bw':
            return  cv2.drawContours(
                np.zeros_like(self.image), 
                [self.boundary], 
                0, (255,)*(len(self.image.shape)) , 2)
        elif x == 'max_contrast':
            ret = self['edge_bw']
            x_min = self.xmin
            window_background = self.metadata[
                'analysis']['max_contrast']['window_background']
            window_tape = self.metadata[
                'analysis']['max_contrast']['window_tape']
            x_start = x_min-window_background
            x_end = x_min+window_tape
            pad_x_start = -1*(min(x_start, 0))
            pad_x_end = -1*(min(ret.shape[1]-x_end, 0))
            x_start = max(0, x_start)
            x_end = min(ret.shape[1], x_end)
            ret = ret[:, x_start:x_end]
            ret = np.pad(
                ret,
                ((0, 0),)
                + ((pad_x_start, pad_x_end),) 
                + ((0, 0),)*(len(ret.shape)-2),
                'constant', 
                constant_values=(0,)
                )
            
            return ret
        elif 'bin_based' in x:
            if 'coordinate_based' in x:
                ret = []
                n_bins = self.metadata.analysis['bin_based']['n_bins']
                n_points = self.metadata.analysis[
                    'coordinate_based']['n_points']//n_bins
                if self.metadata.analysis['bin_based']['overlap']>50:
                    n_points+=2
                for x, y in self.metadata.analysis[
                    'bin_based']['dynamic_positions']:
                    data = np.zeros((n_points, 2))
                    stds = np.zeros((n_points,))
                    slopes = np.zeros((n_points, 2))
                    y_min = y[0]
                    y_max = y[1]
                    pad_y_min = -1*(min(y_min, 0))
                    pad_y_max = -1*(min(self.image.shape[0]-y_max, 0))
                    y_min = max(y_min, 0)
                    y_max = min(self.image.shape[0], y_max)
                    if self.metadata.padding == 'tape':
                        y_min -= pad_y_max
                        y_max += pad_y_min
                    y_interval = y_max - y_min
                    if n_points > len(self._get_points_from_boundary(
                        x[0],x[1],
                        y_min, y_max
                    )):
                        print('The number of points per bin is smaller than'
                              ' the number of points in this bin.'
                              'Consider decreasing the numbers of points for '
                              'the coordinate based method (n_points).')
                    for i_point in range(n_points):
                        y_start = y_min+i_point*(y_interval/n_points)
                        y_end = y_min+(i_point+1)*(y_interval/n_points)
                        points = self._get_points_from_boundary(
                            x[0], 
                            x[1], 
                            y_start, 
                            y_end)

                        assert len(points) != 0, ("No points in this window, " 
                                                  "increase the background window")
                        # assert len(points) != 0, self.metadata.filename
                        data[i_point, :2] = np.average(points, axis=0)
                        stds[i_point] = np.std(points[:, 0])
                        slopes[i_point, :] = np.polyfit(
                            x=points[:, 0], 
                            y=points[:, 1], 
                            deg=1, 
                            full=True)[0]
                    ret.append({
                        'coordinates': data,
                        'stds': stds,
                        'slopes': slopes})
                if self.metadata.analysis['bin_based']['overlap']>50:
                    for i, item in enumerate(ret):
                        for key in item:
                            ret[i][key] = np.delete(item[key], [0, -1], 0)
                return ret
            elif x == 'bin_based':
                image = self['image']
            elif 'max_contrast' in x:
                image = self['edge_bw']
            ret = []
            bin_based =  self.metadata['analysis']['bin_based']
            dynamic_positions = np.array(bin_based['dynamic_positions'])
            if bin_based['n_bins'] > 3:
                delta_y = int(np.diff(dynamic_positions[:, 1][1:-2]).mean())
            else:
                delta_y = int(np.diff(dynamic_positions[:, 1]).mean())
            for seg in dynamic_positions:
                x_start, x_end = seg[0]
                y_start, _ = seg[1]
                y_end = y_start + delta_y
                pad_y_start = -1*(min(y_start, 0))
                y_start = max(y_start, 0)
                pad_y_end = -1*(min(self.image.shape[0]-y_end, 0))
                y_end = min(self.image.shape[0], y_end)
                pad_x_start = -1*(min(x_start, 0))
                x_start = max(x_start, 0)
                if self.metadata.padding == 'black':
                    img = image[y_start:y_end, x_start:x_end]
                    # the (len(img.shape)-1) is to adjust to gray scale and rgb
                    img = np.pad(
                        img,
                        ((pad_y_start, pad_y_end),) 
                        + ((pad_x_start, 0),)
                        + ((0, 0),)*(len(img.shape)-2),
                        'constant', 
                        constant_values=(0, )
                        )
                elif self.metadata.padding == 'tape':
                    y_start -= pad_y_end
                    y_end += pad_y_start
                    img = image[y_start:y_end, x_start:x_end]
                ret.append(img)
            n = [x.shape for x in ret]
            if len(set(n)) > 1:
                print(self.metadata.path)
                print(set(n))
            return np.asarray(ret)
        else:
            raise ValueError(f"TapeAnalyzer does not have attrib {x}")    
        

