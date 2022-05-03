# -*- Coding: utf-8 -*-
"""
Created on Sun Jun 28 14:11:02 2020

@author: Pedram Tavadze
"""
import os
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


class TapeAnalyzer(Analyzer):
    def __init__(self,
                 tape=None,
                 mask_threshold=60,
                 gaussian_blur=(15, 15),
                 n_divisions=6,
                 auto_crop=True,
                 calculate_tilt=True,
                 verbose=True,):
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

        Analyzer.__init__(self)
        self.image_tilt = None
        self.calculate_tilt = calculate_tilt
        self.crop_y_top = None
        self.crop_y_bottom = None
        self.n_divisions = n_divisions
        self.verbose = verbose
        self.gaussian_blur = gaussian_blur
        self.mask_threshold = int(mask_threshold)
        self.masked = None        
        self.material = 'tape'
        self.image_tilt = None
        # self.colored = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        if tape is not None:
            if self.verbose:
                print(f" {tape.filename: <25}|{str(tape.metadata['split_v']['side']): ^5}|{['not flipped', 'flipped'][int(tape.metadata['flip_h'])]: >11}")
            self.image = tape.image
            self.label = tape.label
            self.filename = tape.filename
            self.metadata['image'] = tape.metadata
            self.preprocess(calculate_tilt, auto_crop)
            self.load_dict()
            self.load_metadata()
        return

    def preprocess(self, calculate_tilt=True, auto_crop=True):
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
        if self.gaussian_blur is not None:
            if self.verbose:
                print("applying Gaussian Blur")
            image = image_tools.gaussian_blur(
                self.image, window=self.gaussian_blur)
        if self.verbose:
            print("getting the mask")
        self.masked = image_tools.get_masked(image, self.mask_threshold)
        self.binarized = image_tools.binerized_mask(
            image, self.masked)
        self.gray_scale = image_tools.gray_scale(image)
        if self.verbose:
            print("calculating the tilt")
        self.contours = image_tools.contours(image, self.mask_threshold)
        self.largest_contour = image_tools.largest_contour(self.contours)
        self.boundary = self.largest_contour.reshape(
            self.largest_contour.shape[0], 2)
        if calculate_tilt:
            self.get_image_tilt()
        self.metadata['cropped'] = False
        if auto_crop:
            self.metadata['cropped'] = True
            self.auto_crop_y()
            image = image_tools.gaussian_blur(
                self.image, window=self.gaussian_blur)
            self.contours = image_tools.contours(
                image, self.mask_threshold)
            self.largest_contour = image_tools.largest_contour(self.contours)
            self.boundary = self.largest_contour.reshape(
                self.largest_contour.shape[0], 2)
            self.masked = image_tools.get_masked(
                image, self.mask_threshold)
            self.binarized = image_tools.binerized_mask(
                image, self.masked)
            self.gray_scale = image_tools.gray_scale(self.image)

    def flip_v(self):
        self.image = np.fliplr(self.image)
        self.masked = np.fliplr(self.masked)
        self.binarized = np.fliplr(self.binarized)
        self.gray_scale = np.fliplr(self.gray_scale)
        self.boundary[:, 0] = self.boundary[:, 0]*-1 + self.image.shape[1]
        self.load_dict()
        for key in ['bin_based', 'big_picture']:
            if key in self.values:
                temp = np.flip(self[key],axis=2)
                setattr(self, key, temp)
                self.values[key] = temp
                dynamic_positions = self.metadata['analysis'][key]['dynamic_positions']
                for iseg, pos in enumerate(dynamic_positions):
                    for ix in range(2):
                        dynamic_positions[iseg][0][ix] = pos[0][ix]*-1 + self.image.shape[1]
                self.metadata['analysis'][key]['dynamic_positions'] = dynamic_positions




    def flip_h(self):
        self.image = np.flipud(self.image)
        self.masked = np.flipud(self.masked)
        self.binarized = np.flipud(self.binarized)
        self.gray_scale = np.flipud(self.gray_scale)
        self.boundary[:, 1] = self.boundary[:, 1]*-1 + self.image.shape[0]
        self.load_dict()
        for key in ['bin_based', 'big_picture']:
            if key in self.values:
                temp = np.flip(self[key],axis=2)
                setattr(self, key, temp)
                self.values[key] = temp
                dynamic_positions = self.metadata['analysis'][key]['dynamic_positions']
                for iseg, pos in enumerate(dynamic_positions):
                    for ix in range(2):
                        dynamic_positions[iseg][1][ix] = pos[1][ix]*-1 + self.image.shape[0]
                self.metadata['analysis'][key]['dynamic_positions'] = dynamic_positions

        self.load_dict()

    def load_dict(self):
        """
        

        Returns
        -------
        None.

        """
        self.values['image'] = self.image
        self.values['masked'] = self.masked
        self.values['boundary'] = self.boundary
        self.values['binarized'] = self.binarized
        self.values['gray_scale'] = self.gray_scale
        

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
        self.metadata["image_tilt"] = float(self.image_tilt) if self.image_tilt is not None else None
        self.metadata["x_std"] = float(np.std(self.boundary[:, 0]))
        self.metadata["y_std"] = float(np.std(self.boundary[:, 1]))
        self.metadata["x_mean"] = float(np.mean(self.boundary[:, 0]))
        self.metadata["y_mean"] = float(np.mean(self.boundary[:, 1]))
        self.metadata["gaussian_blur"] = self.gaussian_blur
        self.metadata["mask_threshold"] = self.mask_threshold
        self.metadata["n_divisions"] = self.n_divisions
        self.metadata["material"] = self.material
        self.metadata["filename"] = self.filename
        if self.metadata['image']['split_v']:
            self.metadata['side'] = self.metadata['image']['split_v']['side']
        else :
            self.metadata['side'] = None
        self.metadata["analysis"] = {}

    @classmethod
    def from_dict(cls, values, metadata):
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
        if values is None:
            raise Exception(
                "The provided dictionary was empty. Maybe change the query criteria")
        cls = TapeAnalyzer()
        for key in values:
            if not isinstance(getattr(type(cls), key, None), property):
                setattr(cls, key, values[key])
                cls.values[key] = values[key]
                
        cls.metadata = metadata
        for key in metadata:
            if key in ['filename', 'material', 'n_divisions', 'mask_threshold', 'gaussian_blur', 'image_tilt']:
                setattr(cls, key, metadata[key])
        print(f" {cls.filename: <25}  | {str(cls.metadata['image']['split_v']['side']): ^5} | {['not flipped', 'flipped'][int(cls.metadata['image']['flip_h'])]: >11}")
        # cls.binarized = image_tools.binerized_mask(
        #     cls.image, cls.masked)
        # cls.gray_scale = image_tools.gray_scale(cls.image)
        # cls.load_dict()
        # for key in cls.metadata['analysis']:
            # cls.values[key] = eval("cls.%s" % key)
        # print(" {: <25}  | {: ^5} | {: >11}".format(cls.filename,
                                                    # str(cls.metadata['image']['split_v']['side']),
                                                    # ['not flipped', 'flipped'][int(cls.metadata['image']['flip_h'])]))
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
        stds = np.ones((self.n_divisions-2, 2))*1000
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
        for idivision in range(self.n_divisions-2):

            y_interval = y_max-y_min
            cond1 = boundary[:, 1] > y_max-y_interval/self.n_divisions
            cond2 = boundary[:, 1] < y_max+y_interval/self.n_divisions

            x_interval = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interval/self.n_divisions*(idivision+1)
            cond4 = boundary[:, 0] <= x_min + \
                x_interval/self.n_divisions*(idivision+2)

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

        for idivision in range(self.n_divisions-2):

            cond1 = boundary[:, 1] > y_min-y_interval/self.n_divisions
            cond2 = boundary[:, 1] < y_min+y_interval/self.n_divisions

            x_interval = x_max-x_min
            cond3 = boundary[:, 0] >= x_min+x_interval/self.n_divisions*(idivision+1)
            cond4 = boundary[:, 0] <= x_min + \
                x_interval/self.n_divisions*(idivision+2)

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
        self.crop_y_top = np.average(top) if len(top) != 0 else self.ymax
        self.crop_y_bottom = np.average(bottom) if len(bottom) !=0 else self.ymin
        if np.min(stds, axis=0)[0] > 10:
            self.crop_y_top = y_max
        if np.min(stds, axis=0)[1] > 10:
            self.crop_y_bottom = y_min
            
        angle = np.arctan(m)
        angle_d = np.rad2deg(angle)
        self.image_tilt = angle_d
        return angle_d

    @property
    def edge(self):
        """
         List of pixels that create the boundary.

        Returns
        -------
        edge_bw : 2d array int
            List of pixels that create the boundary.

        """
        zeros = np.zeros_like(self.image)
        edge_bw = cv2.drawContours(
            zeros, self.largest_contour, (255, 255, 255), 2)
        return edge_bw

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
        x_interval = self.xmax-self.xmin
        return x_interval

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
        calculate_tilte : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.image = self.image[int(
            self.crop_y_bottom):int(self.crop_y_top), :]
        if calculate_tilt:
            self.get_image_tilt()

    def get_coordinate_based(self,
                             npoints=64,
                             x_trim_param=6,
                             normalize=True,
                             standardize=False,
                             shift=True,
                             plot=False):
        """
        This method returns the data of the detected edge as a set of points 
        in 2d plain with (x,y)

        Parameters
        ----------
        npoints : TYPE, optional
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

        x_min = self.xmin
        x_max = self.xmax
        x_interval = x_max-x_min
        boundary = self.boundary

        cond1 = boundary[:, 0] >= x_min+x_interval/x_trim_param*0
        cond2 = boundary[:, 0] <= x_min+x_interval/x_trim_param*0.95
        cond_12 = np.bitwise_and(cond1, cond2)
        # cond_12 = cond1
        data = np.zeros((npoints, 3))
        stds = np.zeros((npoints, ))
        y_min = self.ymin
        y_max = self.ymax
        y_interval = y_max - y_min
        edge = boundary[cond_12]
        
        for ipoint in range(0, npoints):
            y_start = y_min+ipoint*(y_interval/npoints)
            y_end = y_min+(ipoint+1)*(y_interval/npoints)
            cond1 = edge[:, 1] >= y_start
            cond2 = edge[:, 1] <= y_end
            cond_and = np.bitwise_and(cond1, cond2)
            points = edge[cond_and]
            if len(points) == 0 and x_trim_param!=1:
                if self.verbose:
                    print('coordinate based is missing some points for {} {} {}, decreasing x_trim_param to {}'.format(
                        self.filename,
                        self.metadata['image']['split_v']['side'],
                        ['flipped', 'not_flipped'][int(self.metadata['image']['flip_h'])],
                        x_trim_param-1))
                return self.get_coordinate_based(npoints=npoints, x_trim_param=x_trim_param-1, plot=plot)
            data[ipoint, :2] = np.average(points, axis=0)
            data[ipoint, 2] = np.std(points[:, 0])
        if shift:
            data[:,0]-=np.average(data[:,0
                                       ])

        if plot:
            # plt.figure(figsize=(1.65,10))
            plt.figure()
            self.show()
            plt.scatter(data[:, 0], data[:, 1], s=1, color='red')
            plt.xlim(data[:, 0].min()*0.9, data[:, 0].max()*1.1)
        self.coordinate_based = data
        self.values['coordinate_based'] = data
        self.metadata['analysis']['coordinate_based'] = {
            "npoints": npoints, "x_trim_param": x_trim_param}

        return self.coordinate_based

    def get_bin_based(self,
                      window_background=50,
                      window_tape=1000,
                      dynamic_window=True,
                      size=None,
                      resize=False,
                      n_segments=10,
                      overlap=100,
                      plot=False,
                      verbose=True,
                      ):
        """
        This method returns the detected edge as a set of croped images from 
        the edge. The number if images is defined by n_segments. The goal is 
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
        n_segments : int, optional
            Number of segments that the tape is going to divided into. The 
            default is 4.
        plot : bool, optional
            Whether or not to plot the divided image. The default is False.


        Returns
        -------
        An array of 2d numpy arrays with images of each segment.

        """

        boundary = self.boundary
        x_min = self.xmin
        x_max = self.xmax

        y_min = self.ymin
        y_max = self.ymax
        # y_min = 0
        # y_max = self.image.shape[0]

        x_start = x_min-window_background
        x_end = min(x_min+window_tape, self.binarized.shape[1])
        
        

        seg_len = (y_max-y_min)//(n_segments)
        seg_len = (y_max-y_min)/(n_segments)

        segments = []
        dynamic_positions = []
        y_end = y_min
        for iseg in range(0, n_segments):
            # y_start = y_min+iseg*seg_len
            y_start = y_end
            y_end = math.ceil(y_min+(iseg+1)*seg_len)
            if dynamic_window:
                cond1 = boundary[:, 1] >= y_start - overlap
                cond2 = boundary[:, 1] <= y_end + overlap
                cond_and = np.bitwise_and(cond1, cond2)
                x_start = boundary[cond_and, 0].min() - window_background
                x_end = boundary[cond_and, 0].min() + window_tape
                if self.binarized.shape[1] < x_end:
                    diff =  self.binarized.shape[1] - x_end
                    x_start += diff
                    x_end = self.binarized.shape[1]
#            cv2.imwrite('%s_%d.tif'%(fname[:-4],iseg),im_color[y_start:y_end,x_start:x_end])
            isection = self.binarized[y_start-overlap:y_end+overlap, x_start:x_end]
            dynamic_positions.append(
                [[int(x_start), int(x_end)], [int(y_start-overlap), int(y_end+overlap)]])
            if resize:
                isection = cv2.resize(isection, (size[1], size[0]))
            segments.append(isection)

            isection = cv2.copyMakeBorder(
                isection, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        sy = min([seg.shape[0] for seg in segments])
        for i, seg in enumerate(segments):
            if seg.shape[0] != sy:
                segments[i] = segments[i][:sy, :]

        segments = np.array(segments)
        metadata = {"dynamic_positions": dynamic_positions,
                    "n_segments": n_segments,
                    "dynamic_window": dynamic_window,
                    "window_background": window_background,
                    "window_tape": window_tape,
                    "size": size}
        if n_segments < 10:
            self.big_picture = segments
            self.values['big_picture'] = segments
            self.metadata['analysis']['big_picture'] = metadata
        else:
            self.bin_based = segments
            self.values['bin_based'] = segments
            self.metadata['analysis']['bin_based'] = metadata
        if plot:
            self.plot('bin_based', cmap='gray', show=True)
        return segments

    def get_max_contrast(self,
                         window_background=100,
                         window_tape=600,
                         size=None,
                         plot=False):
        """
        This method returns the detected image as a black image with only the 
        boundary being white.

        Parameters
        ----------
        window_background : int, optional
            Number of pixels to be included in each segment in the backgroung side of the image. The default is 20.
        window_tape : int, optional
            Number of pixels to be included in each segment in the tape side of the image. The default is 300.
        plot : bool, optional
            Whether or not to plot the divided image. The default is False.

        Returns
        -------
        edge_bw : 2d numpy array
            A black image with all the pixels on the edge white.

        """
        if self.verbose:
            print("getting max contrast")
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
        self.values['max_contrast'] = self.max_contrast
        self.metadata['analysis']['max_contrast'] = {
            "window_background": window_background, "window_tape": window_tape, "size": size}
        return edge_bw




class Tape(Material):
    def __init__(self,
                 filename=None,
                 image=None,
                 label=None,
                 surface=None):
        """
        TapeImage is a class created for tape images to be preprocessed for 
        Machine Learning. This Class detects the edges, auto crops the image 
        and returns the results in 3 different method coordinate_based, 
        bin_based and max_contrast. 
        """
        
        Material.__init__(self)
        self.filename = filename
        self.label = label
        self.surface = surface
        self.material = "tape"

        if self.image is not None:
            self.image = image
        elif self.filename is not None:
            self.read(self.filename)

        self.load_dict()
        self.load_metadata()

    def load_dict(self):
        self.values['image'] = self.image
        

    def load_metadata(self):
        self.metadata['flip_h'] = False
        self.metadata['split_v'] = {
            "side": None, "pixel_index": None}
        self.metadata['label'] = self.label
        self.metadata['filename'] = self.filename
        self.metadata['material'] = self.material
        self.metadata['surface'] = self.surface
        
        
    @classmethod
    def from_dict(cls, values, metadata):
        # this is temperory 
        if "split_vertical" in metadata:
            metadata["split_v"] = metadata["split_vertical"]
        cls = Tape()
        cls.values = dict(values)
        cls.metadata = metadata
        cls.filename = metadata['filename']
        cls.image = values['image']
        cls.label = metadata['label']
        cls.surface = metadata['surface']
        cls.material = metadata['material']
        # cls.load_dict()
        # cls.load_metadata()
        return cls

    def split_v(self, pixel_index=None, side='L', flip=True):
        if pixel_index is None:
            tape_analyzer = TapeAnalyzer(self, verbose=False)
            x = tape_analyzer.boundary[:, 0]
            pixel_index = int((x.max()-x.min())/2)+x.min()
        self.image = image_tools.split_v(
            self.image, pixel_index, side, flip)
        self.values['image'] = self.image
        self.metadata['split_v'] = {
            "side": side, "pixel_index": pixel_index}
        # this is temporary


    def resize(self, size):
        self.image = image_tools.resize(self.image, size)
        self.values['image'] = self.image
        self.metadata['resize'] = size

    def flip_h(self):
        self.image = image_tools.flip(self.image)
        self.values['image'] = self.image
        self.metadata['flip_h'] = True


