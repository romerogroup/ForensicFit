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
                 stretched: bool = False, 
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
        
        
    def split_v(self, side='L', pixel_index=None, ):
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
        # this is temporary



class TapeAnalyzer(Analyzer):
    def __init__(self,
                 tape: Tape = None,
                 mask_threshold: int=60,
                 gaussian_blur: tuple=(15, 15),
                 n_divisions: int=6,
                 auto_crop: bool=False,
                 calculate_tilt: bool=True,
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
        self.metadata["analysis"] = {}
        if tape is not None:
            self.image = tape.image
            self.metadata += tape.metadata
            self.metadata['resolution'] = self.image.shape
            self.preprocess(calculate_tilt, auto_crop)
            if remove_background:
                self.image = self.masked
            
        return

    def preprocess(self, calculate_tilt: bool = True, auto_crop: bool = True):
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
        gray = image_tools.to_gray(image)
        image = image_tools.gaussian_blur(
                gray, window=self.metadata.gaussian_blur)
        contours = image_tools.contours(image,
                                        self.metadata.mask_threshold)
        largest_contour = image_tools.largest_contour(contours)
        self.metadata['boundary'] = largest_contour.reshape(-1, 2)
        if calculate_tilt:
            self.get_image_tilt()
        self.metadata['cropped'] = False
        if auto_crop:
            self.metadata['cropped'] = True
            self.auto_crop_y()
            self.metadata.cropped = True
            gray = image_tools.to_gray(self.image)
            image = image_tools.gaussian_blur(
                gray, window=self.metadata.gaussian_blur)
            self.metadata['cropped'] = True
            contours = image_tools.contours(
                image, self.metadata.mask_threshold)
            largest_contour = image_tools.largest_contour(contours)
            self.metadata['boundary'] = largest_contour.reshape(-1, 2)

        # if self.metadata.remove_background:
        #     self.image = self.masked

    def flip_v(self):
        # TODO coordinate based
        self.image = np.fliplr(self.image)
        self.metadata['boundary'] = np.array(self.metadata['boundary'])
        self.metadata['boundary'][:, 0] = self.metadata['boundary'][:, 0]*-1 + self.image.shape[1]
        if 'coordinate_based' in self.metadata['analysis']:
            coords  = np.array(self.metadata['analysis']['coordinate_based']['data'])
            # coords = np.fliplr(coords)
            coords *= -1
            self.metadata['analysis']['coordinate_based']['data'] = coords
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
            coords  = np.array(self.metadata['analysis']['coordinate_based']['data'])
            coords = np.flipud(coords)
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
        # self.metadata["x_std"] = float(np.std(self.boundary[:, 0]))
        # self.metadata["y_std"] = float(np.std(self.boundary[:, 1]))
        # self.metadata["x_mean"] = float(np.mean(self.boundary[:, 0]))
        # self.metadata["y_mean"] = float(np.mean(self.boundary[:, 1]))


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

    def get_image_tilt(self, plot=False):
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
            _ = plt.figure()
            ax = plt.subplot(111)
            self.plot_boundary(color='black', ax=ax)
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
            plt.figure()
            plt.plot(boundary[:, 0], boundary[:, 1], color='black')
            plt.scatter(boundary[cond_and_top][:, 0],
                        boundary[cond_and_top][:, 1], color='blue')
            plt.scatter(boundary[cond_and_bottom][:, 0],
                        boundary[cond_and_bottom][:, 1], color='red')
        top = boundary[cond_and_top][:, 1]
        bottom = boundary[cond_and_bottom][:, 1]
        self.metadata.crop_y_top = np.average(top) if len(top) != 0 else self.ymax
        self.metadata.crop_y_bottom = np.average(bottom) if len(bottom) !=0 else self.ymin
        if np.min(stds, axis=0)[0] > 10:
            self.metadata.crop_y_top = y_max
        if np.min(stds, axis=0)[1] > 10:
            self.metadata.crop_y_bottom = y_min
            
        angle = np.arctan(m)
        angle_d = np.rad2deg(angle)
        self.metadata.image_tilt = angle_d
        return angle_d

    # @property
    # def edge(self):
    #     """
    #      List of pixels that create the boundary.

    #     Returns
    #     -------
    #     edge_bw : 2d array int
    #         List of pixels that create the boundary.

    #     """
    #     zeros = np.zeros_like(self.image)
    #     edge_bw = cv2.drawContours(
    #         zeros, self.largest_contour, (255, 255, 255), 2)
    #     return edge_bw

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
        # n_xsections = 6
        # cond1 = self.boundary[:, 0] >= self.xmin+self.x_interval/n_xsections*1
        # cond2 = self.boundary[:, 0] <= self.xmin+self.x_interval/n_xsections*2
        # cond_and = np.bitwise_and(cond1, cond2)
        # # using int because pixel numbers are integers
        # ymin = int(self.boundary[cond_and, 1].min())
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
        # n_xsections = 6
        # cond1 = self.boundary[:, 0] >= self.xmin+self.x_interval/n_xsections*1
        # cond2 = self.boundary[:, 0] <= self.xmin+self.x_interval/n_xsections*2
        # cond_and = np.bitwise_and(cond1, cond2)
        # # using int because pixel numbers are integers
        # ymax = int(self.boundary[cond_and, 1].max())
        ymax = self.boundary[:, 1].max()
        return ymax

    def auto_crop_y(self, calculate_tilt=False):
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
        if calculate_tilt:
            self.get_image_tilt()

    def get_coordinate_based(self,
                             n_points: int=64,
                             x_trim_param: int=6,
                             normalize: bool=True,
                             standardize: bool=False,
                             shift: bool=True,
                             plot: bool=False):
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

        cond1 = boundary[:, 0] >= x_min+x_interval/x_trim_param*0
        cond2 = boundary[:, 0] <= x_min+x_interval/x_trim_param*0.95
        cond_12 = np.bitwise_and(cond1, cond2)
        # cond_12 = cond1
        data = np.zeros((n_points, 3))
        y_min = self.ymin
        y_max = self.ymax
        y_interval = y_max - y_min
        edge = boundary[cond_12]
        self.metadata['edge_x_std'] = float(np.std(edge[:, 0]))
        self.metadata['edge_x_range'] = int(edge[:, 0].max() - edge[:, 0].min())
        
        for ipoint in range(0, n_points):
            y_start = y_min+ipoint*(y_interval/n_points)
            y_end = y_min+(ipoint+1)*(y_interval/n_points)
            cond1 = edge[:, 1] >= y_start
            cond2 = edge[:, 1] <= y_end
            cond_and = np.bitwise_and(cond1, cond2)
            points = edge[cond_and]
            if len(points) == 0 and x_trim_param!=1:
                # if self.verbose:
                #     print('coordinate based is missing some points for {} {} {}, decreasing x_trim_param to {}'.format(
                #         self.filename,
                #         self.metadata['image']['split_v']['side'],
                #         ['flipped', 'not_flipped'][int(self.metadata['image']['flip_h'])],
                #         x_trim_param-1))
                return self.get_coordinate_based(n_points=n_points, 
                                                 x_trim_param=x_trim_param - 1, 
                                                 plot=plot)
            data[ipoint, :2] = np.average(points, axis=0)
            data[ipoint, 2] = np.std(points[:, 0])
        if shift:
            data[:, 0] -= np.average(data[:, 0])

        if plot:
            # plt.figure(figsize=(1.65,10))
            plt.figure()
            self.show()
            plt.scatter(data[:, 0], data[:, 1], s=1, color='red')
            plt.xlim(data[:, 0].min()*0.9, data[:, 0].max()*1.1)
        self.metadata['analysis']['coordinate_based'] = {
            "n_points": n_points, 
            "x_trim_param": x_trim_param, 
            'data': data}
        if was_flipped:
            self.flip_v()
            data = self.metadata['analysis']['coordinate_based']['data']
            data -= np.average(data[:, 0])
            self.metadata['analysis']['coordinate_based']['data'] = data
        return 

    def get_bin_based(self,
                      window_background=50,
                      window_tape=1000,
                      dynamic_window=True,
                      size=None,
                      resize=False,
                      n_bins=10,
                      overlap=100,
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
        overlap = abs(overlap)
        if self.metadata.cropped and overlap != 0:
            warnings.warn(("You have selected an overlap larger than 0 with an" 
                           "autocrop option.\n This might result in errors in" 
                           "finding overlap on the side edges of the tape."))
        boundary = self.boundary
        x_min = self.xmin
        x_max = self.xmax

        y_min = self.ymin
        y_max = self.ymax
        # y_min = 0
        # y_max = self.image.shape[0]

        x_start = x_min-window_background
        x_end = min(x_min+window_tape, self.image.shape[1])

        seg_len = (y_max-y_min)//(n_bins)
        seg_len = (y_max-y_min)/(n_bins)

        segments = []
        dynamic_positions = []
        y_end = y_min
        for iseg in range(0, n_bins):
            # y_start = y_min+iseg*seg_len
            y_start = y_end
            y_end = math.ceil(y_min+(iseg+1)*seg_len)
            if dynamic_window:
                cond1 = boundary[:, 1] >= y_start - overlap
                cond2 = boundary[:, 1] <= y_end + overlap
                cond_and = np.bitwise_and(cond1, cond2)
                
                x_start = boundary[cond_and, 0].min() - window_background
                if x_start < 0:
                    x_start=0
                x_end = boundary[cond_and, 0].min() + window_tape
                if self.image.shape[1] < x_end:
                    diff =  self.image.shape[1] - x_end
                    x_start += diff
                    x_end = self.image.shape[1]
#            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
            
            # ys = max(0, y_start - overlap) 
            # ye = min(y_end + overlap, self.image.shape[0]) 
            
            ys = y_start - overlap
            ye = y_end + overlap
            
            dynamic_positions.append(
                [[int(x_start), int(x_end)], [int(ys), int(ye)]])
            # if resize:
            #     isection = cv2.resize(isection, (size[1], size[0]))
            # segments.append(isection)

            # isection = cv2.copyMakeBorder(
            #     isection, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # sy = min([seg[1][1] - seg[1][0] for seg in dynamic_positions])
        
        # for i, seg in enumerate(dynamic_positions):
        #     if seg[1][1] - seg[1][0] != sy:
        #         dynamic_positions[i][1][1] = seg[1][0] + sy

        # segments = np.array(segments)
        metadata = {"dynamic_positions": dynamic_positions,
                    "n_bins": n_bins,
                    "dynamic_window": dynamic_window,
                    "window_background": window_background,
                    "window_tape": window_tape,
                    "size": size}

        self.bin_based = segments
        self.metadata['analysis']['bin_based'] = metadata
        return dynamic_positions

    def get_max_contrast(self,
                         window_background: int=100,
                         window_tape: int=600,
                         size: tuple = None,
                         plot: bool = False):
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
        # if self.verbose:
        #     print("getting max contrast")
        zeros = np.zeros_like(self.image)
        edge_bw = cv2.drawContours(
            zeros, [self.boundary], 0, (255, 255, 255), 2)
        x_min = self.xmin
        x_start = x_min-window_background
        x_end = x_min+window_tape
        edge_bw = edge_bw[:, x_start:x_end]
        if size is not None:
            edge_bw = cv2.resize(edge_bw, (size[1], size[0]))
        if plot:
            plt.figure()
            plt.imshow(edge_bw, cmap='gray')
        self.max_contrast = edge_bw
        self.metadata['analysis']['max_contrast'] = {
            "window_background": window_background,
            "window_tape": window_tape, 
            "size": size}
        return edge_bw

    def copy(self):
        """copies the object

        Returns
        -------
        TapeAnalyer object
            
        """        
        return TapeAnalyzer.from_dict(self.values, self.metadata)


    def __getattr__(self, name):
       return self[name]

    def __getitem__(self, x):
        if x == 'image':
            return self.image
        elif x == 'masked':
            return image_tools.remove_background(self['image'], 
                                                 self.largest_contour)
        elif x == 'gray_scale':
            return image_tools.to_gray(self['image'])
        elif x == 'binarized':
            return image_tools.binerized_mask(
                self['image'], self['masked'])
        elif x == 'largest_contour':
            return self.boundary.reshape(-1, 1, 2)
        elif x == 'boundary':
            return np.asarray(self.metadata['boundary'])
        elif x == 'coordinate_based':
            return np.asarray(self.metadata['analysis']['coordinate_based']['data'])
        elif x == 'max_contrast':
            return self.max_contrast
        elif x == 'bin_based':
            ret = []
            bin_based =  self.metadata['analysis']['bin_based']
            dynamic_positions = np.array(bin_based['dynamic_positions'])
            if self.metadata.analysis['bin_based']['n_bins']>2:
                delta_y = int(np.diff(dynamic_positions[:, 1][1:-2]).mean())
            else:
                delta_y = 0
            for seg in dynamic_positions:
                x_start, x_end = seg[0]
                y_start, _ = seg[1]
                y_end = y_start + delta_y
                pad_y_start = -1*(min(y_start, 0))
                y_start = max(y_start, 0)
                pad_y_end = -1*(min(self.image.shape[0]-y_end, 0))
                y_end = min(self.image.shape[0], y_end)
                img = self.image[y_start:y_end, x_start:x_end]
                # the (len(img.shape)-1) is to adjust to gray scale and rgb
                img = np.pad(
                    img, 
                    ((pad_y_start, pad_y_end),) + ((0, 0),)*(len(img.shape)-1),
                    'constant', 
                    constant_values=(0, )
                    )
                ret.append(img)
            return np.asarray(ret)
        else:
            raise ValueError(f"TapeAnalyzer does not have attrib {x}")    
        

