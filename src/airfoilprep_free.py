#!/usr/bin/env python
# encoding: utf-8
"""
airfoilprep_free.py

Created by Andrew Ning on 2012-04-16.
Copyright (c) NREL. All rights reserved.


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from math import pi, sin, cos, radians, degrees, tan, ceil, floor, factorial
import numpy as np
import copy
import pyXLIGHT
from naca_generator import naca4, naca5
import cmath
import mpmath
from copy import deepcopy
import os
import sys
# from scipy.interpolate import RectBivariateSpline



class Polar(object):
    """
    Defines section lift, drag, and pitching moment coefficients as a
    function of angle of attack at a particular Reynolds number.

    """

    def __init__(self, Re, alpha, cl, cd, cm):
        """Constructor

        Parameters
        ----------
        Re : float
            Reynolds number
        alpha : ndarray (deg)
            angle of attack
        cl : ndarray
            lift coefficient
        cd : ndarray
            drag coefficient
        cm : ndarray
            moment coefficient
        """

        self.Re = Re
        self.alpha = np.array(alpha)
        self.cl = np.array(cl)
        self.cd = np.array(cd)
        self.cm = np.array(cm)


    def blend(self, other, weight):
        """Blend this polar with another one with the specified weighting

        Parameters
        ----------
        other : Polar
            another Polar object to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        polar : Polar
            a blended Polar

        """

        # generate merged set of angles of attack - get unique values
        alpha = np.union1d(self.alpha, other.alpha)

        # truncate (TODO: could also have option to just use one of the polars for values out of range)
        min_alpha = max(self.alpha.min(), other.alpha.min())
        max_alpha = min(self.alpha.max(), other.alpha.max())
        alpha = alpha[np.logical_and(alpha >= min_alpha, alpha <= max_alpha)]
        # alpha = np.array([a for a in alpha if a >= min_alpha and a <= max_alpha])

        # interpolate to new alpha
        cl1 = np.interp(alpha, self.alpha, self.cl)
        cl2 = np.interp(alpha, other.alpha, other.cl)
        cd1 = np.interp(alpha, self.alpha, self.cd)
        cd2 = np.interp(alpha, other.alpha, other.cd)
        cm1 = np.interp(alpha, self.alpha, self.cm)
        cm2 = np.interp(alpha, other.alpha, other.cm)

        # linearly blend
        Re = self.Re + weight*(other.Re-self.Re)
        cl = cl1 + weight*(cl2-cl1)
        cd = cd1 + weight*(cd2-cd1)
        cm = cm1 + weight*(cm2-cm1)

        return type(self)(Re, alpha, cl, cd, cm)



    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """Applies 3-D corrections for rotating sections from the 2-D data.

        Parameters
        ----------
        r_over_R : float
            local radial position / rotor radius
        chord_over_r : float
            local chord length / local radial location
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        polar : Polar
            A new Polar object corrected for 3-D effects

        Notes
        -----
        The Du-Selig method :cite:`Du1998A-3-D-stall-del` is used to correct lift, and
        the Eggers method :cite:`Eggers-Jr2003An-assessment-o` is used to correct drag.


        """

        # rename and convert units for convenience
        alpha = np.radians(self.alpha)
        cl_2d = self.cl
        cd_2d = self.cd
        alpha_max_corr = radians(alpha_max_corr)
        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        # parameters in Du-Selig model
        a = 1
        b = 1
        d = 1
        lam = tsr/(1+tsr**2)**0.5  # modified tip speed ratio
        expon = d/lam/r_over_R

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)
        p = np.polyfit(alpha[idx], cl_2d[idx], 1)
        m = p[0]
        alpha0 = -p[1]/m

        # correction factor
        fcl = 1.0/m*(1.6*chord_over_r/0.1267*(a-chord_over_r**expon)/(b+chord_over_r**expon)-1)

        # not sure where this adjustment comes from (besides AirfoilPrep spreadsheet of course)
        adj = ((pi/2-alpha)/(pi/2-alpha_max_corr))**2
        adj[alpha <= alpha_max_corr] = 1.0

        # Du-Selig correction for lift
        cl_linear = m*(alpha-alpha0)
        cl_3d = cl_2d + fcl*(cl_linear-cl_2d)*adj

        # Eggers 2003 correction for drag
        delta_cl = cl_3d-cl_2d

        delta_cd = delta_cl*(np.sin(alpha) - 0.12*np.cos(alpha))/(np.cos(alpha) + 0.12*np.sin(alpha))
        cd_3d = cd_2d + delta_cd

        return type(self)(self.Re, np.degrees(alpha), cl_3d, cd_3d, self.cm)



    def extrapolate(self, cdmax, AR=None, cdmin=0.001, nalpha=15):
        """Extrapolates force coefficients up to +/- 180 degrees using Viterna's method
        :cite:`Viterna1982Theoretical-and`.

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            aspect ratio = (rotor radius / chord_75% radius)
            if provided, cdmax is computed from AR
        cdmin: float, optional
            minimum drag coefficient.  used to prevent negative values that can sometimes occur
            with this extrapolation method
        nalpha: int, optional
            number of points to add in each segment of Viterna method

        Returns
        -------
        polar : Polar
            a new Polar object

        Notes
        -----
        If the current polar already supplies data beyond 90 degrees then
        this method cannot be used in its current form and will just return itself.

        If AR is provided, then the maximum drag coefficient is estimated as

        >>> cdmax = 1.11 + 0.018*AR


        """

        if cdmin < 0:
            raise Exception('cdmin cannot be < 0')

        # lift coefficient adjustment to account for assymetry
        cl_adj = 0.7

        # estimate CD max
        if AR is not None:
            cdmax = 1.11 + 0.018*AR
        self.cdmax = max(max(self.cd), cdmax)

        # extract matching info from ends
        alpha_high = radians(self.alpha[-1])
        cl_high = self.cl[-1]
        cd_high = self.cd[-1]
        cm_high = self.cm[-1]

        alpha_low = radians(self.alpha[0])
        cl_low = self.cl[0]
        cd_low = self.cd[0]

        if alpha_high > pi/2:
            raise Exception('alpha[-1] > pi/2')
            return self
        if alpha_low < -pi/2:
            raise Exception('alpha[0] < -pi/2')
            return self

        # parameters used in model
        sa = sin(alpha_high)
        ca = cos(alpha_high)
        self.A = (cl_high - self.cdmax*sa*ca)*sa/ca**2
        self.B = (cd_high - self.cdmax*sa*sa)/ca

        # alpha_high <-> 90
        alpha1 = np.linspace(alpha_high, pi/2, nalpha)
        alpha1 = alpha1[1:]  # remove first element so as not to duplicate when concatenating
        cl1, cd1 = self.__Viterna(alpha1, 1.0)

        # 90 <-> 180-alpha_high
        alpha2 = np.linspace(pi/2, pi-alpha_high, nalpha)
        alpha2 = alpha2[1:]
        cl2, cd2 = self.__Viterna(pi-alpha2, -cl_adj)

        # 180-alpha_high <-> 180
        alpha3 = np.linspace(pi-alpha_high, pi, nalpha)
        alpha3 = alpha3[1:]
        cl3, cd3 = self.__Viterna(pi-alpha3, 1.0)
        cl3 = (alpha3-pi)/alpha_high*cl_high*cl_adj  # override with linear variation

        if alpha_low <= -alpha_high:
            alpha4 = []
            cl4 = []
            cd4 = []
            alpha5max = alpha_low
        else:
            # -alpha_high <-> alpha_low
            # Note: this is done slightly differently than AirfoilPrep for better continuity
            alpha4 = np.linspace(-alpha_high, alpha_low, nalpha)
            alpha4 = alpha4[1:-2]  # also remove last element for concatenation for this case
            cl4 = -cl_high*cl_adj + (alpha4+alpha_high)/(alpha_low+alpha_high)*(cl_low+cl_high*cl_adj)
            cd4 = cd_low + (alpha4-alpha_low)/(-alpha_high-alpha_low)*(cd_high-cd_low)
            alpha5max = -alpha_high

        # -90 <-> -alpha_high
        alpha5 = np.linspace(-pi/2, alpha5max, nalpha)
        alpha5 = alpha5[1:]
        cl5, cd5 = self.__Viterna(-alpha5, -cl_adj)

        # -180+alpha_high <-> -90
        alpha6 = np.linspace(-pi+alpha_high, -pi/2, nalpha)
        alpha6 = alpha6[1:]
        cl6, cd6 = self.__Viterna(alpha6+pi, cl_adj)

        # -180 <-> -180 + alpha_high
        alpha7 = np.linspace(-pi, -pi+alpha_high, nalpha)
        cl7, cd7 = self.__Viterna(alpha7+pi, 1.0)
        cl7 = (alpha7+pi)/alpha_high*cl_high*cl_adj  # linear variation

        alpha = np.concatenate((alpha7, alpha6, alpha5, alpha4, np.radians(self.alpha), alpha1, alpha2, alpha3))
        cl = np.concatenate((cl7, cl6, cl5, cl4, self.cl, cl1, cl2, cl3))
        cd = np.concatenate((cd7, cd6, cd5, cd4, self.cd, cd1, cd2, cd3))

        cd = np.maximum(cd, cdmin)  # don't allow negative drag coefficients


        # Setup alpha and cm to be used in extrapolation
        cm1_alpha = floor(self.alpha[0] / 10.0) * 10.0
        cm2_alpha = ceil(self.alpha[-1] / 10.0) * 10.0
        alpha_num = abs(int((-180.0-cm1_alpha)/10.0 - 1))
        alpha_cm1 = np.linspace(-180.0, cm1_alpha, alpha_num)
        alpha_cm2 = np.linspace(cm2_alpha, 180.0, int((180.0-cm2_alpha)/10.0 + 1))
        alpha_cm = np.concatenate((alpha_cm1, self.alpha, alpha_cm2))  # Specific alpha values are needed for cm function to work
        cm1 = np.zeros(len(alpha_cm1))
        cm2 = np.zeros(len(alpha_cm2))
        cm_ext = np.concatenate((cm1, self.cm, cm2))
        if np.count_nonzero(self.cm) > 0:
            cmCoef = self.__CMCoeff(cl_high, cd_high, cm_high)  # get cm coefficient
            cl_cm = np.interp(alpha_cm, np.degrees(alpha), cl)  # get cl for applicable alphas
            cd_cm = np.interp(alpha_cm, np.degrees(alpha), cd)  # get cd for applicable alphas
            alpha_low_deg = self.alpha[0]
            alpha_high_deg = self.alpha[-1]
            for i in range(len(alpha_cm)):
                cm_new = self.__getCM(i, cmCoef, alpha_cm, cl_cm, cd_cm, alpha_low_deg, alpha_high_deg)
                if cm_new is None:
                    pass  # For when it reaches the range of cm's that the user provides
                else:
                    cm_ext[i] = cm_new
        try:
            cm = np.interp(np.degrees(alpha), alpha_cm, cm_ext)
        except:
            cm = np.zeros(len(cl))
        return type(self)(self.Re, np.degrees(alpha), cl, cd, cm)




    def __Viterna(self, alpha, cl_adj):
        """private method to perform Viterna extrapolation"""

        alpha = np.maximum(alpha, 0.0001)  # prevent divide by zero

        cl = self.cdmax/2*np.sin(2*alpha) + self.A*np.cos(alpha)**2/np.sin(alpha)
        cl = cl*cl_adj

        cd = self.cdmax*np.sin(alpha)**2 + self.B*np.cos(alpha)

        return cl, cd

    def __CMCoeff(self, cl_high, cd_high, cm_high):
        """private method to obtain CM0 and CMCoeff"""

        found_zero_lift = False

        for i in range(len(self.cm)):
            if abs(self.alpha[i]) < 20.0 and self.cl[i] <= 0 and self.cl[i+1] >= 0:
                p = -self.cl[i] / (self.cl[i + 1] - self.cl[i])
                cm0 = self.cm[i] + p * (self.cm[i+1] - self.cm[i])
                found_zero_lift = True
                break

        if not found_zero_lift:
            p = -self.cl[0] / (self.cl[1] - self.cl[0])
            cm0 = self.cm[0] + p * (self.cm[1] - self.cm[0])
        self.cm0 = cm0
        alpha_high = radians(self.alpha[-1])
        XM = (-cm_high + cm0) / (cl_high * cos(alpha_high) + cd_high * sin(alpha_high))
        cmCoef = (XM - 0.25) / tan((alpha_high - pi/2))
        return cmCoef

    def __getCM(self, i, cmCoef, alpha, cl_ext, cd_ext, alpha_low_deg, alpha_high_deg):
        """private method to extrapolate Cm"""

        cm_new = 0
        if alpha[i] >= alpha_low_deg and alpha[i] <= alpha_high_deg:
            return
        if alpha[i] > -165 and alpha[i] < 165:
            if abs(alpha[i]) < 0.01:
                cm_new = self.cm0
            else:
                if alpha[i] > 0:
                    x = cmCoef * tan(radians(alpha[i]) - pi/2) + 0.25
                    cm_new = self.cm0 - x * (cl_ext[i] * cos(radians(alpha[i])) + cd_ext[i] * sin(radians(alpha[i])))
                else:
                    x = cmCoef * tan(-radians(alpha[i]) - pi/2) + 0.25
                    cm_new = -(self.cm0 - x * (-cl_ext[i] * cos(-radians(alpha[i])) + cd_ext[i] * sin(-radians(alpha[i]))))
        else:
            if alpha[i] == 165:
                cm_new = -0.4
            elif alpha[i] == 170:
                cm_new = -0.5
            elif alpha[i] == 175:
                cm_new = -0.25
            elif alpha[i] == 180:
                cm_new = 0
            elif alpha[i] == -165:
                cm_new = 0.35
            elif alpha[i] == -170:
                cm_new = 0.4
            elif alpha[i] == -175:
                cm_new = 0.2
            elif alpha[i] == -180:
                cm_new = 0
            else:
                print "Angle encountered for which there is no CM table value (near +/-180 deg). Program will stop."
        return cm_new

    def unsteadyparam(self, alpha_linear_min=-5, alpha_linear_max=5):
        """compute unsteady aero parameters used in AeroDyn input file

        Parameters
        ----------
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        aerodynParam : tuple of floats
            (control setting, stall angle, alpha for 0 cn, cn slope,
            cn at stall+, cn at stall-, alpha for min CD, min(CD))

        """

        alpha = np.radians(self.alpha)
        cl = self.cl
        cd = self.cd

        alpha_linear_min = radians(alpha_linear_min)
        alpha_linear_max = radians(alpha_linear_max)

        cn = cl*np.cos(alpha) + cd*np.sin(alpha)

        # find linear region
        idx = np.logical_and(alpha >= alpha_linear_min,
                             alpha <= alpha_linear_max)

        # checks for inppropriate data (like cylinders)
        if len(idx) < 10 or len(np.unique(cl)) < 10:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

        # linear fit
        p = np.polyfit(alpha[idx], cn[idx], 1)
        m = p[0]
        alpha0 = -p[1]/m

        # find cn at stall locations
        alphaUpper = np.radians(np.arange(40.0))
        alphaLower = np.radians(np.arange(5.0, -40.0, -1))
        cnUpper = np.interp(alphaUpper, alpha, cn)
        cnLower = np.interp(alphaLower, alpha, cn)
        cnLinearUpper = m*(alphaUpper - alpha0)
        cnLinearLower = m*(alphaLower - alpha0)
        deviation = 0.05  # threshold for cl in detecting stall

        alphaU = np.interp(deviation, cnLinearUpper-cnUpper, alphaUpper)
        alphaL = np.interp(deviation, cnLower-cnLinearLower, alphaLower)

        # compute cn at stall according to linear fit
        cnStallUpper = m*(alphaU-alpha0)
        cnStallLower = m*(alphaL-alpha0)

        # find min cd
        minIdx = cd.argmin()

        # return: control setting, stall angle, alpha for 0 cn, cn slope,
        #         cn at stall+, cn at stall-, alpha for min CD, min(CD)
        return (0.0, degrees(alphaU), degrees(alpha0), m,
                cnStallUpper, cnStallLower, alpha[minIdx], cd[minIdx])

    def plot(self):
        """plot cl/cd/cm polar

        Returns
        -------
        figs : list of figure handles

        """
        import matplotlib.pyplot as plt

        p = self

        figs = []

        # plot cl
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        plt.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('lift coefficient')
        ax.legend(loc='best')

        # plot cd
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('drag coefficient')
        ax.legend(loc='best')

        # plot cm
        fig = plt.figure()
        figs.append(fig)
        ax = fig.add_subplot(111)
        ax.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
        ax.set_xlabel('angle of attack (deg)')
        ax.set_ylabel('moment coefficient')
        ax.legend(loc='best')

        return figs


class Airfoil(object):
    """A collection of Polar objects at different Reynolds numbers

    """

    def __init__(self, polars):
        """Constructor

        Parameters
        ----------
        polars : list(Polar)
            list of Polar objects

        """

        # sort by Reynolds number
        self.polars = sorted(polars, key=lambda p: p.Re)

        # save type of polar we are using
        self.polar_type = polars[0].__class__


    @classmethod
    def initFromAerodynFile(cls, aerodynFile, polarType=Polar):
        """Construct Airfoil object from AeroDyn file

        Parameters
        ----------
        aerodynFile : str
            path/name of a properly formatted Aerodyn file

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        # open aerodyn file
        f = open(aerodynFile, 'r')

        # skip through header
        f.readline()
        description = f.readline().rstrip()  # remove newline
        f.readline()
        numTables = int(f.readline().split()[0])

        # loop through tables
        for i in range(numTables):

            # read Reynolds number
            Re = float(f.readline().split()[0])*1e6

            # read Aerodyn parameters
            param = [0]*8
            for j in range(8):
                param[j] = float(f.readline().split()[0])

            alpha = []
            cl = []
            cd = []
            cm = []

            # read polar information line by line
            while True:
                line = f.readline()
                if 'EOT' in line:
                    break
                data = [float(s) for s in line.split()]
                alpha.append(data[0])
                cl.append(data[1])
                cd.append(data[2])
                cm.append(data[3])

            polars.append(polarType(Re, alpha, cl, cd, cm))

        f.close()

        return cls(polars)

    @classmethod
    def initFromCoordinateFile(cls, CoordinateFile, alphas, Re, polarType=Polar):
        """Construct Airfoil object from airfoil coordinate file

        Parameters
        ----------
        CoordinateFile : array of str
            paths/names of properly formatted airfoil coordinate files

        alphas : array of floats
            array of angles of attack

        Re : float
            Reynolds number

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        for i in range(len(CoordinateFile)):
            # read in coordinate file
            with suppress_stdout_stderr():
                airfoil = pyXLIGHT.xfoilAnalysis(CoordinateFile[i])
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 1000

            cl = np.zeros(len(alphas))
            cd = np.zeros(len(alphas))
            cm = np.zeros(len(alphas))
            to_delete = np.zeros(0)

            for j in range(len(alphas)):
                angle = alphas[j]
                cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(angle)
                if lexitflag:
                    cl[j] = -10.0
                    cd[j] = 0.0
            # error handling in case of XFOIL failure
            for k in range(len(cl)):
                if cl[k] == -10.0:
                    if k == 0:
                        cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
                        cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
                    elif k == len(cl)-1:
                        cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
                        cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
                    else:
                        cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
                        cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
                if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
                    to_delete = np.append(to_delete, k)
            cl = np.delete(cl, to_delete)
            cd = np.delete(cd, to_delete)
            alphas = np.delete(alphas, to_delete)

            polars.append(polarType(Re, alphas, cl, cd, cm))

        return cls(polars)

    @classmethod
    def initFromNACA(cls, NACA, alphas, Re, polarType=Polar):
        """Construct Airfoil object from airfoil coordinate file

        Parameters
        ----------
        NACA : array of str
            paths/names of properly formatted airfoil coordinate files

        alphas : array of floats
            array of angles of attack

        Re : float
            Reynolds number

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        for i in range(len(NACA)):
            x = []
            y = []
            if len(NACA[i]) == 4:
                pts = naca4(NACA[i], 60)
            if len(NACA[i]) == 5:
                pts = naca5(NACA[i], 60)
            else:
                'Please input only NACA 4 or 5 series airfoils'
            for j in range(len(pts)):
                x.append(pts[j][0])
                y.append(pts[j][1])

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
            airfoil_shape_file = basepath + os.path.sep + 'naca_coordinates.dat'

            coord_file = open(airfoil_shape_file, 'w')

            print >> coord_file, 'naca' + NACA[i]
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

            coord_file.close()

            # read in coordinate file
            with suppress_stdout_stderr():
                airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file)
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 1000

            cl = np.zeros(len(alphas))
            cd = np.zeros(len(alphas))
            cm = np.zeros(len(alphas))
            to_delete = np.zeros(0)
            for j in range(len(alphas)):
                angle = alphas[j]
                cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(angle)
                if lexitflag:
                    cl[j] = -10.0
                    cd[j] = 0.0
            # error handling in case of XFOIL failure
            for k in range(len(cl)):
                if cl[k] == -10.0:
                    if k == 0:
                        cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
                        cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
                    elif k == len(cl)-1:
                        cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
                        cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
                    else:
                        cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
                        cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
                if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
                    to_delete = np.append(to_delete, k)
            cl = np.delete(cl, to_delete)
            cd = np.delete(cd, to_delete)
            alphas = np.delete(alphas, to_delete)

            polars.append(polarType(Re, alphas, cl, cd, cm))

        return cls(polars)


    @classmethod
    def initFromCST(cls, CST, alphas, Re, polarType=Polar):
        """Construct Airfoil object from airfoil coordinate file

        Parameters
        ----------
        NACA : array of str
            paths/names of properly formatted airfoil coordinate files

        alphas : array of floats
            array of angles of attack

        Re : float
            Reynolds number

        Returns
        -------
        obj : Airfoil

        """
        # initialize
        polars = []

        def __ClassShape(w, x, N1, N2, dz):

            # Class function; taking input of N1 and N2
            C = np.zeros(len(x))
            for i in range(len(x)):
                C[i] = x[i]**N1*((1-x[i])**N2)

            # Shape function; using Bernstein Polynomials
            n = len(w) - 1  # Order of Bernstein polynomials

            K = np.zeros(n+1)
            for i in range(0, n+1):
                K[i] = factorial(n)/(factorial(i)*(factorial((n)-(i))))

            S = np.zeros(len(x))
            for i in range(len(x)):
                S[i] = 0
                for j in range(0, n+1):
                    S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

            # Calculate y output
            y = np.zeros(len(x))
            for i in range(len(y)):
                y[i] = C[i] * S[i] + x[i] * dz

            return y

        try:
            n1 = len(CST[0])/2
            n2 = len(CST)
        except:
            n2 = 1
            n1 = len(CST)/2
            CST = np.array([CST])
        for i in range(n2):
            wu = np.zeros(n1)
            wl = np.zeros(n1)
            for j in range(n1):
                wu[j] = CST[i][j]
                wl[j] = CST[i][j + n1]
            # wu, wl = np.split(af_parameters[i], 2)
            w1 = np.average(wl)
            w2 = np.average(wu)
            if w1 < w2:
                pass
            else:
                higher = wl
                lower = wu
                wl = lower
                wu = higher
            N = 120
            dz = 0.

            # Populate x coordinates
            x = np.ones((N, 1))
            zeta = np.zeros((N, 1))
            for z in range(0, N):
                zeta[z] = 2 * pi / N * z
                if z == N - 1:
                    zeta[z] = 2 * pi
                x[z] = 0.5*(cos(zeta[z])+1)

            # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
            N1 = 0.5
            N2 = 1

            try:
                zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
                zerind = zerind[0][0]
            except:
                zerind = N/2

            xl = np.zeros(zerind)
            xu = np.zeros(N-zerind)

            for z in range(len(xl)):
                xl[z] = np.real(x[z])            # Lower surface x-coordinates
            for z in range(len(xu)):
                xu[z] = np.real(x[z + zerind])   # Upper surface x-coordinates

            yl = __ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
            yu = __ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

            y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
            y = y[::-1]
            # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
            # coord = [x, y]
            x1 = np.zeros(len(x))
            for k in range(len(x)):
                x1[k] = x[k][0]
            x = x1

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
            airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates.dat'

            coord_file = open(airfoil_shape_file, 'w')

            print >> coord_file, 'CST'
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

            coord_file.close()

            # read in coordinate file
            with suppress_stdout_stderr():
                airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file)
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 1000

            cl = np.zeros(len(alphas))
            cd = np.zeros(len(alphas))
            cm = np.zeros(len(alphas))
            to_delete = np.zeros(0)
            for j in range(len(alphas)):
                angle = alphas[j]
                cl[j], cd[j], cm[j], lexitflag = airfoil.solveAlpha(angle)
                if lexitflag:
                    cl[j] = -10.0
                    cd[j] = 0.0

            # error handling in case of XFOIL failure
            # for k in range(len(cl)):
            #     if cl[k] == -10.0:
            #         if k == 0:
            #             cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
            #             cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
            #         elif k == len(cl)-1:
            #             cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
            #             cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
            #         else:
            #             cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
            #             cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
            #     if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
            #         to_delete = np.append(to_delete, k)
            # cl = np.delete(cl, to_delete)
            # cd = np.delete(cd, to_delete)
            # alphas = np.delete(alphas, to_delete)

            polars.append(polarType(Re, alphas, cl, cd, cm))

        return cls(polars)


    def getPolar(self, Re):
        """Gets a Polar object for this airfoil at the specified Reynolds number.

        Parameters
        ----------
        Re : float
            Reynolds number

        Returns
        -------
        obj : Polar
            a Polar object

        Notes
        -----
        Interpolates as necessary. If Reynolds number is larger than or smaller than
        the stored Polars, it returns the Polar with the closest Reynolds number.

        """

        p = self.polars

        if Re <= p[0].Re:
            return copy.deepcopy(p[0])

        elif Re >= p[-1].Re:
            return copy.deepcopy(p[-1])

        else:
            Relist = [pp.Re for pp in p]
            i = np.searchsorted(Relist, Re)
            weight = (Re - Relist[i-1]) / (Relist[i] - Relist[i-1])
            return p[i-1].blend(p[i], weight)



    def blend(self, other, weight):
        """Blend this Airfoil with another one with the specified weighting.


        Parameters
        ----------
        other : Airfoil
            other airfoil to blend with
        weight : float
            blending parameter between 0 and 1.  0 returns self, whereas 1 returns other.

        Returns
        -------
        obj : Airfoil
            a blended Airfoil object

        Notes
        -----
        First finds the unique Reynolds numbers.  Evaluates both sets of polars
        at each of the Reynolds numbers, then blends at each Reynolds number.

        """

        # combine Reynolds numbers
        Relist1 = [p.Re for p in self.polars]
        Relist2 = [p.Re for p in other.polars]
        Relist = np.union1d(Relist1, Relist2)

        # blend polars
        n = len(Relist)
        polars = [0]*n
        for i in range(n):
            p1 = self.getPolar(Relist[i])
            p2 = other.getPolar(Relist[i])
            polars[i] = p1.blend(p2, weight)


        return Airfoil(polars)


    def correction3D(self, r_over_R, chord_over_r, tsr, alpha_max_corr=30,
                     alpha_linear_min=-5, alpha_linear_max=5):
        """apply 3-D rotational corrections to each polar in airfoil

        Parameters
        ----------
        r_over_R : float
            radial position / rotor radius
        chord_over_r : float
            local chord / local radius
        tsr : float
            tip-speed ratio
        alpha_max_corr : float, optional (deg)
            maximum angle of attack to apply full correction
        alpha_linear_min : float, optional (deg)
            angle of attack where linear portion of lift curve slope begins
        alpha_linear_max : float, optional (deg)
            angle of attack where linear portion of lift curve slope ends

        Returns
        -------
        airfoil : Airfoil
            airfoil with 3-D corrections

        See Also
        --------
        Polar.correction3D : apply 3-D corrections for a Polar

        """

        n = len(self.polars)
        polars = [0]*n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.correction3D(r_over_R, chord_over_r, tsr, alpha_max_corr, alpha_linear_min, alpha_linear_max)

        return Airfoil(polars)


    def extrapolate(self, cdmax, AR=None, cdmin=0.001):
        """apply high alpha extensions to each polar in airfoil

        Parameters
        ----------
        cdmax : float
            maximum drag coefficient
        AR : float, optional
            blade aspect ratio (rotor radius / chord at 75% radius).  if included
            it is used to estimate cdmax
        cdmin: minimum drag coefficient

        Returns
        -------
        airfoil : Airfoil
            airfoil with +/-180 degree extensions

        See Also
        --------
        Polar.extrapolate : extrapolate a Polar to high angles of attack

        """

        n = len(self.polars)
        polars = [0]*n
        for idx, p in enumerate(self.polars):
            polars[idx] = p.extrapolate(cdmax, AR, cdmin)

        return Airfoil(polars)



    def interpToCommonAlpha(self, alpha=None):
        """Interpolates all polars to a common set of angles of attack

        Parameters
        ----------
        alpha : ndarray, optional
            common set of angles of attack to use.  If None a union of
            all angles of attack in the polars is used.

        """

        if alpha is None:
            # union of angle of attacks
            alpha = []
            for p in self.polars:
                alpha = np.union1d(alpha, p.alpha)

        # interpolate each polar to new alpha
        n = len(self.polars)
        polars = [0]*n
        if n == 1:
            polars[0] = self.polar_type(p.Re, alpha, p.cl, p.cd, p.cm)
            return Airfoil(polars)
        for idx, p in enumerate(self.polars):
            cl = np.interp(alpha, p.alpha, p.cl)
            cd = np.interp(alpha, p.alpha, p.cd)
            cm = np.interp(alpha, p.alpha, p.cm)
            polars[idx] = self.polar_type(p.Re, alpha, cl, cd, cm)

        return Airfoil(polars)





    def writeToAerodynFile(self, filename):
        """Write the airfoil section data to a file using AeroDyn input file style.

        Parameters
        ----------
        filename : str
            name (+ relative path) of where to write file

        """

        # aerodyn and wtperf require common set of angles of attack
        af = self.interpToCommonAlpha()

        f = open(filename, 'w')

        print >> f, 'AeroDyn airfoil file.'
        print >> f, 'Compatible with AeroDyn v13.0.'
        print >> f, 'Generated by airfoilprep_free.py'
        print >> f, '{0:<10d}\t\t{1:40}'.format(len(af.polars), 'Number of airfoil tables in this file')
        for p in af.polars:
            print >> f, '{0:<10f}\t{1:40}'.format(p.Re/1e6, 'Reynolds number in millions.')
            param = p.unsteadyparam()
            print >> f, '{0:<10f}\t{1:40}'.format(param[0], 'Control setting')
            print >> f, '{0:<10f}\t{1:40}'.format(param[1], 'Stall angle (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[2], 'Angle of attack for zero Cn for linear Cn curve (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[3], 'Cn slope for zero lift for linear Cn curve (1/rad)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[4], 'Cn at stall value for positive angle of attack for linear Cn curve')
            print >> f, '{0:<10f}\t{1:40}'.format(param[5], 'Cn at stall value for negative angle of attack for linear Cn curve')
            print >> f, '{0:<10f}\t{1:40}'.format(param[6], 'Angle of attack for minimum CD (deg)')
            print >> f, '{0:<10f}\t{1:40}'.format(param[7], 'Minimum CD value')
            for a, cl, cd, cm in zip(p.alpha, p.cl, p.cd, p.cm):
                print >> f, '{:<10f}\t{:<10f}\t{:<10f}\t{:<10f}'.format(a, cl, cd, cm)
            print >> f, 'EOT'
        f.close()


    def createDataGrid(self):
        """interpolate airfoil data onto uniform alpha-Re grid.

        Returns
        -------
        alpha : ndarray (deg)
            a common set of angles of attack (union of all polars)
        Re : ndarray
            all Reynolds numbers defined in the polars
        cl : ndarray
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : ndarray
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """

        af = self.interpToCommonAlpha()
        polarList = af.polars

        # angle of attack is already same for each polar
        alpha = polarList[0].alpha

        # all Reynolds numbers
        Re = [p.Re for p in polarList]

        # fill in cl, cd grid
        cl = np.zeros((len(alpha), len(Re)))
        cd = np.zeros((len(alpha), len(Re)))
        cm = np.zeros((len(alpha), len(Re)))

        for (idx, p) in enumerate(polarList):
            cl[:, idx] = p.cl
            cd[:, idx] = p.cd
            cm[:, idx] = p.cm


        return alpha, Re, cl, cd, cm

    def __ClassShape(self, w, x, N1, N2, dz):

            # Class function; taking input of N1 and N2
            C = np.zeros(len(x))
            for i in range(len(x)):
                C[i] = x[i]**N1*((1-x[i])**N2)

            # Shape function; using Bernstein Polynomials
            n = len(w) - 1  # Order of Bernstein polynomials

            K = np.zeros(n+1)
            for i in range(0, n+1):
                K[i] = factorial(n)/(factorial(i)*(factorial((n)-(i))))

            S = np.zeros(len(x))
            for i in range(len(x)):
                S[i] = 0
                for j in range(0, n+1):
                    S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

            # Calculate y output
            y = np.zeros(len(x))
            for i in range(len(y)):
                y[i] = C[i] * S[i] + x[i] * dz

            return y

    def __ClassShapeComplex(self, w, x, N1, N2, dz):

            # Class function; taking input of N1 and N2
            C = np.zeros(len(x), dtype=complex)
            for i in range(len(x)):
                C[i] = x[i]**N1*((1-x[i])**N2)

            # Shape function; using Bernstein Polynomials
            n = len(w) - 1  # Order of Bernstein polynomials

            K = np.zeros(n+1, dtype=complex)
            for i in range(0, n+1):
                K[i] = mpmath.factorial(n)/(mpmath.factorial(i)*(mpmath.factorial((n)-(i))))

            S = np.zeros(len(x), dtype=complex)
            for i in range(len(x)):
                S[i] = 0
                for j in range(0, n+1):
                    S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

            # Calculate y output
            y = np.zeros(len(x), dtype=complex)
            for i in range(len(y)):
                y[i] = C[i] * S[i] + x[i] * dz

            return y

    @classmethod
    def xfoilFlowGradients(self, CST, alpha, Re):

        step_size = 1e-20
        cs_step = complex(0, step_size)

        # read in coordinate file
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
        airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates.dat'
        # TODO: Check to make sure the right file is being read
        with suppress_stdout_stderr():
            airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file)
        airfoil.re = Re
        airfoil.mach = 0.00
        airfoil.iter = 1000

        # cl, cd, cm, lexitflag = airfoil.solveAlpha(alpha)
        # if lexitflag:
        #     cl = -10.0
        #     cd = 0.0
        cl_alpha, cd_alpha, cm, lexitflag = airfoil.solveAlphaComplex(alpha+cs_step)
        if lexitflag:
            cl_alpha = -10.0
            cd_alpha = 0.0
        # TODO: CHECK EQUATION FOR COMPLEX STEP
        dcl_dalpha = np.imag(cl_alpha)/np.imag(cs_step)
        dcd_dalpha = np.imag(cd_alpha)/np.imag(cs_step)

        airfoil.re = Re[0][0] + cs_step
        cl_Re, cd_Re, cm, lexitflag = airfoil.solveAlphaComplex(alpha)
        if lexitflag:
            cl_Re = -10.0
            cd_Re = 0.0
        dcl_dRe = np.imag(cl_Re)/np.imag(cs_step)
        dcd_dRe = np.imag(cd_Re)/np.imag(cs_step)

        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    @classmethod
    def xfoilGradients(self, CST, alpha, Re, FDorCS):

        def cstComplex(alpha, Re, wl, wu, N, dz):
            # wl = self.wl
            # wu = self.wu
            N = N
            dz = dz

            # Populate x coordinates
            x = np.ones((N, 1), dtype=complex)
            zeta = np.zeros((N, 1)) #, dtype=complex)
            for z in range(0, N):
                zeta[z] = 2.0 * pi / N * z
                if z == N - 1:
                    zeta[z] = 2.0 * pi
                x[z] = 0.5*(cmath.cos(zeta[z])+1.0)

            # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
            N1 = 0.5
            N2 = 1

            try:
                zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
                zerind = zerind[0][0]
            except:
                zerind = N/2

            xl = np.zeros(zerind, dtype=complex)
            xu = np.zeros(N-zerind, dtype=complex)

            for z in range(len(xl)):
                xl[z] = x[z][0]        # Lower surface x-coordinates
            for z in range(len(xu)):
                xu[z] = x[z + zerind][0]   # Upper surface x-coordinates

            yl = __ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
            yu = __ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

            y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
            y = y[::-1]
            # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
            # coord = [x, y]
            x1 = np.zeros(len(x), dtype=complex)
            for k in range(len(x)):
                x1[k] = x[k][0]
            x = x1

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
            airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates_complex.dat'

            coord_file = open(airfoil_shape_file, 'w')

            print >> coord_file, 'CST'
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

            coord_file.close()

            # read in coordinate file
            with suppress_stdout_stderr():
                airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file)
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 1000

            angle = alpha
            cl, cd, cm, lexitflag = airfoil.solveAlphaComplex(angle)
            if lexitflag:
                cl = -10.0
                cd = 0.0
            return cl, cd
            # error handling in case of XFOIL failure
            # for k in range(len(cl)):
            #     if cl[k] == -10.0:
            #         if k == 0:
            #             cl[k] = cl[k+1] - cl[k+2] + cl[k+1]
            #             cd[k] = cd[k+1] - cd[k+2] + cd[k+1]
            #         elif k == len(cl)-1:
            #             cl[k] = cl[k-1] - cl[k-2] + cl[k-1]
            #             cd[k] = cd[k-1] - cd[k-2] + cd[k-1]
            #         else:
            #             cl[k] = (cl[k+1] - cl[k-1])/2.0 + cl[k-1]
            #             cd[k] = (cd[k+1] - cd[k-1])/2.0 + cd[k-1]
            #     if cl[k] == -10.0 or cl[k] < -2. or cl[k] > 2. or cd[k] < 0.00001 or cd[k] > 0.5 or not np.isfinite(cd[k]) or not np.isfinite(cl[k]):
            #         to_delete = np.append(to_delete, k)
            # cl = np.delete(cl, to_delete)
            # cd = np.delete(cd, to_delete)
            # alphas = np.delete(alphas, to_delete)

            # polars.append(polarType(Re, alphas, cl, cd, cm))
        def __ClassShape(w, x, N1, N2, dz):

            # Class function; taking input of N1 and N2
            C = np.zeros(len(x))
            for i in range(len(x)):
                C[i] = x[i]**N1*((1-x[i])**N2)

            # Shape function; using Bernstein Polynomials
            n = len(w) - 1  # Order of Bernstein polynomials

            K = np.zeros(n+1)
            for i in range(0, n+1):
                K[i] = factorial(n)/(factorial(i)*(factorial((n)-(i))))

            S = np.zeros(len(x))
            for i in range(len(x)):
                S[i] = 0
                for j in range(0, n+1):
                    S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

            # Calculate y output
            y = np.zeros(len(x))
            for i in range(len(y)):
                y[i] = C[i] * S[i] + x[i] * dz

            return y

        def __ClassShapeComplex(w, x, N1, N2, dz):

            # Class function; taking input of N1 and N2
            C = np.zeros(len(x), dtype=complex)
            for i in range(len(x)):
                C[i] = x[i]**N1*((1-x[i])**N2)

            # Shape function; using Bernstein Polynomials
            n = len(w) - 1  # Order of Bernstein polynomials

            K = np.zeros(n+1, dtype=complex)
            for i in range(0, n+1):
                K[i] = mpmath.factorial(n)/(mpmath.factorial(i)*(mpmath.factorial((n)-(i))))

            S = np.zeros(len(x), dtype=complex)
            for i in range(len(x)):
                S[i] = 0
                for j in range(0, n+1):
                    S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

            # Calculate y output
            y = np.zeros(len(x), dtype=complex)
            for i in range(len(y)):
                y[i] = C[i] * S[i] + x[i] * dz

            return y

        def cstReal(alpha, Re, wl, wu, N, dz):

            # Populate x coordinates
            x = np.ones((N, 1))
            zeta = np.zeros((N, 1))
            for z in range(0, N):
                zeta[z] = 2 * pi / N * z
                if z == N - 1:
                    zeta[z] = 2.0 * pi
                x[z] = 0.5*(cos(zeta[z])+1.0)

            # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
            N1 = 0.5
            N2 = 1

            try:
                zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
                zerind = zerind[0][0]
            except:
                zerind = N/2

            xl = np.zeros(zerind)
            xu = np.zeros(N-zerind)

            for z in range(len(xl)):
                xl[z] = x[z]        # Lower surface x-coordinates
            for z in range(len(xu)):
                xu[z] = x[z + zerind]   # Upper surface x-coordinates

            yl = __ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
            yu = __ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

            y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
            y = y[::-1]
            # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
            # coord = [x, y]
            x1 = np.zeros(len(x))
            for k in range(len(x)):
                x1[k] = x[k][0]
            x = x1

            basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
            airfoil_shape_file = basepath + os.path.sep + 'cst_coordinates.dat'

            coord_file = open(airfoil_shape_file, 'w')

            print >> coord_file, 'CST'
            for i in range(len(x)):
                print >> coord_file, '{:<10f}\t{:<10f}'.format(x[i], y[i])

            coord_file.close()

            # read in coordinate file
            with suppress_stdout_stderr():
                airfoil = pyXLIGHT.xfoilAnalysis(airfoil_shape_file)
            airfoil.re = Re
            airfoil.mach = 0.00
            airfoil.iter = 1000

            angle = alpha
            cl, cd, cm, lexitflag = airfoil.solveAlpha(angle)
            if lexitflag:
                cl = -10.0
                cd = 0.0
            return cl, cd



        n2 = 1
        n1 = len(CST)/2
        CST = np.array([CST])
        for i in range(n2):
            wu = np.zeros(n1, dtype=complex)
            wl = np.zeros(n1, dtype=complex)
            for j in range(n1):
                wu[j] = CST[i][j]
                wl[j] = CST[i][j + n1]
            # wu, wl = np.split(af_parameters[i], 2)
            w1 = np.average(wl)
            w2 = np.average(wu)
            if w1 < w2:
                pass
            else:
                higher = wl
                lower = wu
                wl = lower
                wu = higher
            N = 120
            dz = 0.
        self.wl = wl
        self.wu = wu
        self.N = N
        self.dz = dz

        step_size = 1e-20
        cs_step = complex(0, step_size)
        dcl_dcst, dcd_dcst = np.zeros(8), np.zeros(8)
        cl, cd = cstReal(alpha, Re, wl, wu, N, dz)

        for i in range(len(wl)):
            wl_complex = wl.copy()
            wl_complex[i] += cs_step
            cl_complex, cd_complex = cstComplex(alpha, Re, wl_complex, wu, N, dz)
            dcl_dcst[i] = np.imag(cl_complex)/np.imag(cs_step)
            dcd_dcst[i] = np.imag(cd_complex)/np.imag(cs_step)
            wu_complex = wu.copy()
            wu_complex[i] += cs_step
            cl_complex, cd_complex = cstComplex(alpha, Re, wl, wu_complex, N, dz)
            dcl_dcst[i+4] = np.imag(cl_complex)/np.imag(cs_step)
            dcd_dcst[i+4] = np.imag(cd_complex)/np.imag(cs_step)

        return cl, cd, dcl_dcst, dcd_dcst




    @classmethod
    def cfdGradients(self, CST, alpha, Re, iterations, processors, FDorCS):

        import os, sys, shutil, copy
        sys.path.append(os.environ['SU2_RUN'])
        import SU2

        # filename = 'free_form_config.cfg'
        basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CoordinatesFiles')
        filename = basepath + os.path.sep + 'inv_NACA0012.cfg'
        # filename = 'inv_NACA0012.cfg'
        partitions = processors
        compute = True
        step = 1e-4
        iterations = iterations

        # Config and state
        config = SU2.io.Config(filename)
        state  = SU2.io.State()
        config.NUMBER_PART = partitions
        config.EXT_ITER    = iterations

        # find solution files if they exist
        state.find_files(config)

        konfig = copy.deepcopy(config)
        ztate  = copy.deepcopy(state)

        konfig.AoA = np.degrees(alpha)
        konfig.MACH_NUMBER = 0.05
        #TODO : ADD REYNOLDS NUMBER

        cd = SU2.eval.func('DRAG', konfig, ztate)
        cl = SU2.eval.func('LIFT', konfig, ztate)
        cm = SU2.eval.func('MOMENT_Z', konfig, ztate)

        # # check for existing files
        # if not compute:
        #     config.RESTART_SOL = 'YES'
        #     state.find_files(config)
        # else:
        #     state.FILES.MESH = config.MESH_FILENAME
        #
        # # Direct Solution
        # if compute:
        #     info = SU2.run.direct(config)
        #     state.update(info)
        #     SU2.io.restart2solution(config,state)
        #
        # # Adjoint Solution
        # if compute:
        #     info = SU2.run.adjoint(config)
        #     state.update(info)
        #     #SU2.io.restart2solution(config,state)
        #
        # # Gradient Projection
        # info = SU2.run.projection(config,step)
        # state.update(info)
        #
        # get_gradients = info.get('GRADIENTS')
        # adjoint_gradient_cd = get_gradients.get('DRAG')
        # adjoint_gradient_cl = get_gradients.get('LIFT')

        # Adjoint Solution
        dcl_dcst = np.zeros(8)
        dcd_dcst = np.zeros(8)

        info = SU2.run.adjoint(config)
        state.update(info)
        #SU2.io.restart2solution(config,state)

        # Gradient Projection
        info = SU2.run.projection(config,step)
        state.update(info)

        get_gradients = info.get('GRADIENTS')
        dcl_dx = get_gradients.get('LIFT')
        dcd_dx = get_gradients.get('DRAG')


        # return state

        n = len(CST)
        m = len(dcd_dx)
        dcst_dx = np.zeros((n, m))
        fd_step = 1e-6

        wl_original = [-0.5, -0.5, -0.5, -0.5]
        wu_original = [0.5, 0.5, 0.5, 0.5]
        dz = 0.0
        N = 200
        coord_old = cst_to_coordinates(wl_original, wu_original, N, dz)

        design = [85, 79, 74, 70, 67, 63, 60, 56, 53, 50, 47, 43, 40, 37, 33, 29, 25, 21, 14, 115, 121, 126, 130, 133, 137, 140, 144, 147, 150, 153, 157, 160, 163, 167, 171, 175, 179, 186]

        # Gradients
        if FDorCS == 'FD':

            for i in range(0, n):
                wl_new = deepcopy(wl_original)
                wu_new = deepcopy(wu_original)
                if i < n/2:
                    wl_new[i] += fd_step
                else:
                    wu_new[i-4] += fd_step
                coor_new = cst_to_coordinates(wl_new, wu_new, N, dz)
                j = 0
                for coor_d in design:
                    if (coor_new[1][coor_d] - coord_old[1][coor_d]).real == 0:
                        dcst_dx[i][j] = 0
                    else:
                        dcst_dx[i][j] = 1/((coor_new[1][coor_d] - coord_old[1][coor_d]).real / fd_step)
                    j += 1

        elif FDorCS == 'CS':
            step_size = 1e-20
            cs_step = complex(0, step_size)

            for i in range(0, n):
                wl_new = deepcopy(wl_original)
                wu_new = deepcopy(wu_original)
                if i >= n/2:
                    wl_new[i] += cs_step
                else:
                    wu_new[i+4] += fd_step
                coor_new = cst_to_coordinates(wl_new, wu_new, N, dz)
                j = 0
                for coor_d in design:
                    if coor_new[1][coor_d].imag == 0:
                        dcst_dx[i][j] = 0
                    else:
                        dcst_dx[i][j] = 1/(coor_new[1][coor_d].imag / step_size)
                    j += 1
        dcl_dx = np.zeros((38)) #,1))

        dcst_dx = np.matrix(dcst_dx)
        dcl_dx = np.matrix(dcl_dx)
        dcd_dx = np.matrix(dcd_dx)

        dcl_dcst = dcst_dx * dcl_dx.T
        dcd_dcst = dcst_dx * dcd_dx.T

        return cl, cd, dcl_dcst, dcd_dcst

    def plot(self, single_figure=True):
        """plot cl/cd/cm polars

        Parameters
        ----------
        single_figure : bool
            True  : plot all cl on the same figure (same for cd,cm)
            False : plot all cl/cd/cm on separate figures

        Returns
        -------
        figs : list of figure handles

        """

        import matplotlib.pyplot as plt

        figs = []

        # if in single figure mode (default)
        if single_figure:
            # generate figure handles
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            figs.append(fig1)

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            figs.append(fig2)

            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111)
            figs.append(fig3)

            # loop through polars and plot
            for p in self.polars:
                # plot cl
                ax1.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
                ax1.set_xlabel('angle of attack (deg)')
                ax1.set_ylabel('lift coefficient')
                ax1.legend(loc='best')

                # plot cd
                ax2.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
                ax2.set_xlabel('angle of attack (deg)')
                ax2.set_ylabel('drag coefficient')
                ax2.legend(loc='best')

                # plot cm
                ax3.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
                ax3.set_xlabel('angle of attack (deg)')
                ax3.set_ylabel('moment coefficient')
                ax3.legend(loc='best')

        # otherwise, multi figure mode -- plot all on separate figures
        else:
            for p in self.polars:
                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cl, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('lift coefficient')
                ax.legend(loc='best')

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cd, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('drag coefficient')
                ax.legend(loc='best')

                fig = plt.figure()
                figs.append(fig)
                ax = fig.add_subplot(111)
                ax.plot(p.alpha, p.cm, label='Re = ' + str(p.Re/1e6) + ' million')
                ax.set_xlabel('angle of attack (deg)')
                ax.set_ylabel('moment coefficient')
                ax.legend(loc='best')
        plt.show()
        return figs

def cst_to_coordinates(wl, wu, N, dz):
    x = np.ones((N, 1))
    zeta = np.zeros((N, 1))
    for z in range(0, N):
        zeta[z] = 2 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind)
    xu = np.zeros(N-zerind)

    for z in range(len(xl)):
        xl[z] = x[z]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind]   # Upper surface x-coordinates

    yl = __ClassShape(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = __ClassShape(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x))
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1
    return [x, y]

def cst_to_coordinates_complex(wl, wu, N, dz):
    # Populate x coordinates
    x = np.ones((N, 1), dtype=complex)
    zeta = np.zeros((N, 1)) #, dtype=complex)
    for z in range(0, N):
        zeta[z] = 2.0 * pi / N * z
        if z == N - 1:
            zeta[z] = 2.0 * pi
        x[z] = 0.5*(cmath.cos(zeta[z])+1.0)

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    try:
        zerind = np.where(x == 0)  # Used to separate upper and lower surfaces
        zerind = zerind[0][0]
    except:
        zerind = N/2

    xl = np.zeros(zerind, dtype=complex)
    xu = np.zeros(N-zerind, dtype=complex)

    for z in range(len(xl)):
        xl[z] = x[z][0]        # Lower surface x-coordinates
    for z in range(len(xu)):
        xu[z] = x[z + zerind][0]   # Upper surface x-coordinates

    yl = __ClassShapeComplex(wl, xl, N1, N2, -dz) # Call ClassShape function to determine lower surface y-coordinates
    yu = __ClassShapeComplex(wu, xu, N1, N2, dz)  # Call ClassShape function to determine upper surface y-coordinates

    y = np.concatenate([yl, yu])  # Combine upper and lower y coordinates
    y = y[::-1]
    # coord_split = [xl, yl, xu, yu]  # Combine x and y into single output
    # coord = [x, y]
    x1 = np.zeros(len(x), dtype=complex)
    for k in range(len(x)):
        x1[k] = x[k][0]
    x = x1
    return [x, y]

def __ClassShape(w, x, N1, N2, dz):

    # Class function; taking input of N1 and N2
    C = np.zeros(len(x))
    for i in range(len(x)):
        C[i] = x[i]**N1*((1-x[i])**N2)

    # Shape function; using Bernstein Polynomials
    n = len(w) - 1  # Order of Bernstein polynomials

    K = np.zeros(n+1)
    for i in range(0, n+1):
        K[i] = factorial(n)/(factorial(i)*(factorial((n)-(i))))

    S = np.zeros(len(x))
    for i in range(len(x)):
        S[i] = 0
        for j in range(0, n+1):
            S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

    # Calculate y output
    y = np.zeros(len(x))
    for i in range(len(y)):
        y[i] = C[i] * S[i] + x[i] * dz

    return y

def __ClassShapeComplex(w, x, N1, N2, dz):

    # Class function; taking input of N1 and N2
    C = np.zeros(len(x), dtype=complex)
    for i in range(len(x)):
        C[i] = x[i]**N1*((1-x[i])**N2)

    # Shape function; using Bernstein Polynomials
    n = len(w) - 1  # Order of Bernstein polynomials

    K = np.zeros(n+1, dtype=complex)
    for i in range(0, n+1):
        K[i] = mpmath.factorial(n)/(mpmath.factorial(i)*(mpmath.factorial((n)-(i))))

    S = np.zeros(len(x), dtype=complex)
    for i in range(len(x)):
        S[i] = 0
        for j in range(0, n+1):
            S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

    # Calculate y output
    y = np.zeros(len(x), dtype=complex)
    for i in range(len(y)):
        y[i] = C[i] * S[i] + x[i] * dz

    return y

    # def evaluate(self, alpha, Re):
    #     """Get lift/drag coefficient at the specified angle of attack and Reynolds number

    #     Parameters
    #     ----------
    #     alpha : float (rad)
    #         angle of attack (in Radians!)
    #     Re : float
    #         Reynolds number

    #     Returns
    #     -------
    #     cl : float
    #         lift coefficient
    #     cd : float
    #         drag coefficient

    #     Notes
    #     -----
    #     Uses a spline so that output is continuously differentiable
    #     also uses a small amount of smoothing to help remove spurious multiple solutions

    #     """

    #     # setup spline if necessary
    #     if self.need_to_setup_spline:
    #         alpha_v, Re_v, cl_M, cd_M = self.createDataGrid()
    #         alpha_v = np.radians(alpha_v)

    #         # special case if zero or one Reynolds number (need at least two for bivariate spline)
    #         if len(Re_v) < 2:
    #             Re_v = [1e1, 1e15]
    #             cl_M = np.c_[cl_M, cl_M]
    #             cd_M = np.c_[cd_M, cd_M]

    #         kx = min(len(alpha_v)-1, 3)
    #         ky = min(len(Re_v)-1, 3)

    #         self.cl_spline = RectBivariateSpline(alpha_v, Re_v, cl_M, kx=kx, ky=ky, s=0.1)
    #         self.cd_spline = RectBivariateSpline(alpha_v, Re_v, cd_M, kx=kx, ky=ky, s=0.001)
    #         self.need_to_setup_spline = False

    #     # evaluate spline --- index to make scalar

    #     cl = self.cl_spline.ev(alpha, Re)[0]
    #     cd = self.cd_spline.ev(alpha, Re)[0]

    #     return cl, cd






if __name__ == '__main__':

    import os
    from argparse import ArgumentParser, RawTextHelpFormatter

    # setup command line arguments
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                            description='Preprocessing airfoil data for wind turbine applications.')
    parser.add_argument('src_file', type=str, help='source file')
    parser.add_argument('--stall3D', type=str, nargs=3, metavar=('r/R', 'c/r', 'tsr'), help='2D data -> apply 3D corrections')
    parser.add_argument('--extrap', type=str, nargs=1, metavar=('cdmax'), help='3D data -> high alpha extrapolations')
    parser.add_argument('--blend', type=str, nargs=2, metavar=('otherfile', 'weight'), help='blend 2 files weight 0: sourcefile, weight 1: otherfile')
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--plot', action='store_true', help='plot data using matplotlib')
    parser.add_argument('--common', action='store_true', help='interpolate the data at different Reynolds numbers to a common set of angles of attack')


    # parse command line arguments
    args = parser.parse_args()
    fileOut = args.out

    if args.plot:
        import matplotlib.pyplot as plt

    # perform actions
    if args.stall3D is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_3D' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)
        floats = [float(var) for var in args.stall3D]
        af3D = af.correction3D(*floats)

        if args.common:
            af3D = af3D.interpToCommonAlpha()

        af3D.writeToAerodynFile(fileOut)

        if args.plot:

            for p, p3D in zip(af.polars, af3D.polars):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                plt.plot(p.alpha, p.cl, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cl, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend(loc='lower right')

                # plt.subplot(122)
                plt.figure()
                plt.plot(p.alpha, p.cd, 'k', label='2D')
                plt.plot(p3D.alpha, p3D.cd, 'r', label='3D')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend(loc='upper center')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/stall3d.pdf')

            plt.show()


    elif args.extrap is not None:

        if fileOut is None:
            name, ext = os.path.splitext(args.src_file)
            fileOut = name + '_extrap' + ext

        af = Airfoil.initFromAerodynFile(args.src_file)

        afext = af.extrapolate(float(args.extrap[0]))

        if args.common:
            afext = afext.interpToCommonAlpha()

        afext.writeToAerodynFile(fileOut)

        if args.plot:

            for p, pext in zip(af.polars, afext.polars):
                # plt.figure(figsize=(6.0, 2.6))
                # plt.subplot(121)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cl, 'r')
                p2, = plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='upper right')

                # plt.subplot(122)
                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cd, 'r')
                p2, = plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='lower right')

                plt.figure()
                p1, = plt.plot(pext.alpha, pext.cm, 'r')
                p2, = plt.plot(p.alpha, p.cm, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('moment coefficient')
                plt.legend([p2, p1], ['orig', 'extrap'], loc='upper right')

                # plt.tight_layout()
                # plt.savefig('/Users/sning/Dropbox/NREL/SysEng/airfoilpreppy/docs/images/extrap.pdf')

            plt.show()


    elif args.blend is not None:

        if fileOut is None:
            name1, ext = os.path.splitext(args.src_file)
            name2, ext = os.path.splitext(os.path.basename(args.blend[0]))
            fileOut = name1 + '+' + name2 + '_blend' + args.blend[1] + ext

        af1 = Airfoil.initFromAerodynFile(args.src_file)
        af2 = Airfoil.initFromAerodynFile(args.blend[0])
        afOut = af1.blend(af2, float(args.blend[1]))

        if args.common:
            afOut = afOut.interpToCommonAlpha()

        afOut.writeToAerodynFile(fileOut)



        if args.plot:

            for p in afOut.polars:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cl, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('lift coefficient')
                plt.text(0.6, 0.2, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cd, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('drag coefficient')
                plt.text(0.2, 0.8, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(p.alpha, p.cm, 'k')
                plt.xlabel('angle of attack (deg)')
                plt.ylabel('moment coefficient')
                plt.text(0.2, 0.8, 'Re = ' + str(p.Re/1e6) + ' million', transform=ax.transAxes)

            plt.show()
# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])