__author__ = 'ryanbarr'


import warnings
from math import cos, sin, pi
import numpy as np
import _bem
from openmdao.api import Component
from openmdao.api import ExecComp, IndepVarComp, Group, Problem
from scipy.optimize import brentq
from zope.interface import Interface, implements
from scipy.interpolate import RectBivariateSpline, bisplev
from airfoilprep import Airfoil

class AirfoilInterface(Interface):
    """Interface for airfoil aerodynamic analysis."""

    def evaluate(alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        Any implementation can be used, but to keep the smooth properties
        of CCBlade, the implementation should be C1 continuous.

        """

class CCAirfoil:
    """A helper class to evaluate airfoil data using a continuously
    differentiable cubic spline"""
    implements(AirfoilInterface)


    def __init__(self, alpha, Re, cl, cd):
        """Setup CCAirfoil from raw airfoil data on a grid.

        Parameters
        ----------
        alpha : array_like (deg)
            angles of attack where airfoil data are defined
            (should be defined from -180 to +180 degrees)
        Re : array_like
            Reynolds numbers where airfoil data are defined
            (can be empty or of length one if not Reynolds number dependent)
        cl : array_like
            lift coefficient 2-D array with shape (alpha.size, Re.size)
            cl[i, j] is the lift coefficient at alpha[i] and Re[j]
        cd : array_like
            drag coefficient 2-D array with shape (alpha.size, Re.size)
            cd[i, j] is the drag coefficient at alpha[i] and Re[j]

        """

        alpha = np.radians(alpha)
        self.one_Re = False

        # special case if zero or one Reynolds number (need at least two for bivariate spline)
        if len(Re) < 2:
            Re = [1e1, 1e15]
            cl = np.c_[cl, cl]
            cd = np.c_[cd, cd]
            self.one_Re = True

        kx = min(len(alpha)-1, 3)
        ky = min(len(Re)-1, 3)

        # a small amount of smoothing is used to prevent spurious multiple solutions
        self.cl_spline = RectBivariateSpline(alpha, Re, cl, kx=kx, ky=ky, s=0.1)
        self.cd_spline = RectBivariateSpline(alpha, Re, cd, kx=kx, ky=ky, s=0.001)


    @classmethod
    def initFromAerodynFile(cls, aerodynFile):
        """convenience method for initializing with AeroDyn formatted files

        Parameters
        ----------
        aerodynFile : str
            location of AeroDyn style airfoiil file

        Returns
        -------
        af : CCAirfoil
            a constructed CCAirfoil object

        """

        af = Airfoil.initFromAerodynFile(aerodynFile)
        alpha, Re, cl, cd = af.createDataGrid()
        return cls(alpha, Re, cl, cd)


    def evaluate(self, alpha, Re):
        """Get lift/drag coefficient at the specified angle of attack and Reynolds number.

        Parameters
        ----------
        alpha : float (rad)
            angle of attack
        Re : float
            Reynolds number

        Returns
        -------
        cl : float
            lift coefficient
        cd : float
            drag coefficient

        Notes
        -----
        This method uses a spline so that the output is continuously differentiable, and
        also uses a small amount of smoothing to help remove spurious multiple solutions.

        """

        cl = self.cl_spline.ev(alpha, Re)
        cd = self.cd_spline.ev(alpha, Re)

        return cl, cd


    def derivatives(self, alpha, Re):

        # note: direct call to bisplev will be unnecessary with latest scipy update (add derivative method)
        tck_cl = self.cl_spline.tck[:3] + self.cl_spline.degrees  # concatenate lists
        tck_cd = self.cd_spline.tck[:3] + self.cd_spline.degrees

        dcl_dalpha = bisplev(alpha, Re, tck_cl, dx=1, dy=0)
        dcd_dalpha = bisplev(alpha, Re, tck_cd, dx=1, dy=0)

        if self.one_Re:
            dcl_dRe = 0.0
            dcd_dRe = 0.0
        else:
            dcl_dRe = bisplev(alpha, Re, tck_cl, dx=0, dy=1)
            dcd_dRe = bisplev(alpha, Re, tck_cd, dx=0, dy=1)

        return dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe


class CCInit(Component):

    """
    CCInit

    Initializes all the inputs and performs a few simple calculations as preparation for analysis

    """

    def __init__(self, nSector):
        super(CCInit, self).__init__()

        self.add_param('Rtip', val=0.0)
        self.add_param('precone', val=0.0)
        self.add_param('tilt', val=0.0)
        self.add_param('yaw', val=0.0)
        self.add_param('shearExp', val=0.0)
        self.add_param('precurveTip', val=0.0)
        self.add_param('presweepTip', val=0.0)
        self.add_param('Uinf', val=0.0)
        self.add_param('tsr', val=0.0)

        self.add_output('rotorR', shape=1)
        self.add_output('nSector', shape=1)
        self.add_output('Omega', shape=1)

        self.nSector = nSector
        self.fd_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):

        precone = params['precone']
        tilt = params['tilt']
        yaw = params['yaw']
        shearExp = params['shearExp']
        Rtip = params['Rtip']
        precurveTip = params['precurveTip']
        nSector = self.nSector

        # rotor radius
        unknowns['rotorR'] = Rtip*cos(precone) + precurveTip*sin(precone)

        # azimuthal discretization
        if tilt == 0.0 and yaw == 0.0 and shearExp == 0.0:
            nSector = 1  # no more are necessary
        else:
            nSector = max(4, nSector)  # at least 4 are necessary

        unknowns['nSector'] = nSector

        Uinf = params['Uinf']
        Rtip = params['Rtip']
        unknowns['Omega'] = Uinf*params['tsr']/Rtip * 30.0/pi

    def jacobian(self, params, unknowns, resids):

        J = {}

        precone = params['precone']
        precurveTip = params['precurveTip']
        Rtip = params['Rtip']
        tsr = params['tsr']
        Uinf = params['Uinf']

        J['rotorR', 'precone'] = -Rtip*sin(precone) + precurveTip*cos(precone)
        J['rotorR', 'precurveTip'] = sin(precone)
        J['rotorR', 'Rtip'] = cos(precone)
        J['nSector', 'nSector_in'] = 1
        J['Omega', 'Uinf'] = tsr/Rtip * 30.0 / pi
        J['Omega', 'tsr'] = Uinf/Rtip * 30.0 / pi
        J['Omega', 'Rtip'] = -Uinf*tsr / (Rtip**2) * 30.0 / pi

        return J

class WindComponents(Component):
    """
    WindComponents

    Outputs: Vx, Vy

    """

    def __init__(self, n):
        super(WindComponents, self).__init__()
        self.add_param('r', val=np.zeros(n))
        self.add_param('precurve', val=np.zeros(n))
        self.add_param('presweep', shape=n)
        self.add_param('Uinf', shape=1)
        self.add_param('precone', shape=1)
        self.add_param('azimuth', val=0.0)
        self.add_param('tilt', shape=1)
        self.add_param('yaw', shape=1)
        self.add_param('Omega', shape=1)
        self.add_param('shearExp', shape=1)
        self.add_param('hubHt', shape=1)

        self.add_output('Vx', shape=n)
        self.add_output('Vy', shape=n)

        self.fd_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):

        r = params['r']
        precurve = params['precurve']
        presweep = params['presweep']
        precone = params['precone']
        yaw = params['yaw']
        tilt = params['tilt']
        Uinf = params['Uinf']
        Omega = params['Omega']
        hubHt = params['hubHt']
        shearExp = params['shearExp']
        azimuth = params['azimuth']

        Vx, Vy = _bem.windcomponents(r, precurve, presweep, precone, yaw, tilt, azimuth, Uinf, Omega, hubHt, shearExp)

        unknowns['Vx'] = Vx
        unknowns['Vy'] = Vy


    def jacobian(self, params, unknowns, resids):
        J = {}

        r = params['r']
        precurve = params['precurve']
        presweep = params['presweep']
        precone = params['precone']
        yaw = params['yaw']
        tilt = params['tilt']
        azimuth = params['azimuth']
        Uinf = params['Uinf']
        Omega = params['Omega']
        hubHt = params['hubHt']
        shearExp = params['shearExp']

        # y = [r, precurve, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]  (derivative order)
        n = len(r)
        dy_dy = np.eye(3*n+7)

        _, Vxd, _, Vyd = _bem.windcomponents_dv(r, dy_dy[:, :n], precurve, dy_dy[:, n:2*n],
            presweep, dy_dy[:, 2*n:3*n], precone, dy_dy[:, 3*n], yaw, dy_dy[:, 3*n+3],
            tilt, dy_dy[:, 3*n+1], azimuth, dy_dy[:, 3*n+4], Uinf, dy_dy[:, 3*n+5],
            Omega, dy_dy[:, 3*n+6], hubHt, dy_dy[:, 3*n+2], shearExp)

        dVx_dr = np.diag(Vxd[:n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dr = np.diag(Vyd[:n, :])

        dVx_dcurve = Vxd[n:2*n, :].T  # tri-diagonal  (note: dVx_j / dcurve_i  i==row)
        dVy_dcurve = Vyd[n:2*n, :].T  # off-diagonal are actually all zero, but leave for convenience

        dVx_dsweep = np.diag(Vxd[2*n:3*n, :])  # off-diagonal terms are known to be zero and not needed
        dVy_dsweep = np.diag(Vyd[2*n:3*n, :])

        # w = [r, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]
        dVx_dw = np.vstack((dVx_dr, dVx_dsweep, Vxd[3*n:, :]))
        dVy_dw = np.vstack((dVy_dr, dVy_dsweep, Vyd[3*n:, :]))

        J['Vx', 'r'] = Vxd[:n, :]
        J['Vy', 'r'] = Vyd[:n, :]
        J['Vx', 'presweep'] = Vxd[2*n:3*n, :]
        J['Vy', 'presweep'] = Vyd[2*n:3*n, :]
        J['Vx', 'precone'] = dVx_dw[2]
        J['Vy', 'precone'] = dVy_dw[2]
        J['Vx', 'tilt'] = dVx_dw[3]
        J['Vy', 'tilt'] = dVy_dw[3]
        J['Vx', 'hubHt'] = dVx_dw[4]
        J['Vy', 'hubHt'] = dVy_dw[4]
        J['Vx', 'yaw'] = dVx_dw[5]
        J['Vy', 'yaw'] = dVy_dw[5]
        J['Vx', 'azimuth'] = dVx_dw[6]
        J['Vy', 'azimuth'] = dVy_dw[6]
        J['Vx', 'Uinf'] = dVx_dw[7]
        J['Vy', 'Uinf'] = dVy_dw[7]
        J['Vx', 'Omega'] = dVx_dw[8]
        J['Vy', 'Omega'] = dVy_dw[8]
        J['Vx', 'precurve'] = dVx_dcurve
        J['Vy', 'precurve'] = dVy_dcurve

        return J

class Angles(Component):
    def __init__(self, af, bemoptions, n):
        super(Angles, self).__init__()

        self.add_param('pitch', shape=1)
        self.add_param('Rtip', shape=1)
        self.add_param('Vx', shape=n)
        self.add_param('Vy', shape=n)
        self.add_param('Omega', shape=1)
        self.add_param('r', val=np.zeros(n))
        self.add_param('chord', shape=n)
        self.add_param('theta', shape=n)
        self.add_param('rho', shape=1)
        self.add_param('mu', shape=1)
        self.add_param('Rhub', shape=1)
        self.add_param('B', val=3, pass_by_obj=True)

        self.add_output('phi', val=np.zeros(n))

        self.af = af
        self.bemoptions = bemoptions
        self.fd_options['form'] = 'central'
        self.fd_options['force_fd'] = True

    def __errorFunction(self, phi, r, chord, theta, af, Vx, Vy, iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions):
        """strip other outputs leaving only residual for Brent's method"""

        fzero, a, ap = self.__runBEM(phi, r, chord, theta, af, Vx, Vy, iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions)

        return fzero

    def __runBEM(self, phi, r, chord, theta, af, Vx, Vy, iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions):
        """residual of BEM method and other corresponding variables"""

        a = 0.0
        ap = 0.0

        if iterRe == 0.0:
            iterRe = int(iterRe)
        for i in range(iterRe):

            alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, pitch,
                                             chord, theta, rho, mu)
            cl, cd = af.evaluate(alpha, Re)

            fzero, a, ap = _bem.inductionfactors(r, chord, Rhub, Rtip, phi,
                                                 cl, cd, B, Vx, Vy, **bemoptions)

        return fzero, a, ap

    def __residualDerivatives(self, phi, r, chord, theta, af, Vx, Vy, iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions):
        """derivatives of fzero, a, ap"""

        if iterRe != 1:
            ValueError('Analytic derivatives not supplied for case with iterRe > 1')

        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)

        # alpha, Re (analytic derivaives)
        a = 0.0
        ap = 0.0
        alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, pitch,
                                         chord, theta, rho, mu)

        dalpha_dx = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        dRe_dx = np.array([0.0, Re/chord, 0.0, Re*Vx/W**2, Re*Vy/W**2, 0.0, 0.0, 0.0, 0.0])

        # cl, cd (spline derivatives)
        cl, cd = af.evaluate(alpha, Re)
        dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af.derivatives(alpha, Re)

        # chain rule
        dcl_dx = dcl_dalpha*dalpha_dx + dcl_dRe*dRe_dx
        dcd_dx = dcd_dalpha*dalpha_dx + dcd_dRe*dRe_dx

        # residual, a, ap (Tapenade)
        dx_dx = np.eye(9)

        fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(r, chord, Rhub, Rtip,
            phi, cl, cd, B, Vx, Vy, dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
            dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **bemoptions)

        return dR_dx, da_dx, dap_dx


    def solve_nonlinear(self, params, unknowns, resids):

        Vx = params['Vx']
        Vy = params['Vy']
        Omega = params['Omega']
        r = params['r']
        chord = params['chord']
        theta = params['theta']
        pitch = params['pitch']
        rho = params['rho']
        mu = params['mu']
        Rhub = params['Rhub']
        Rtip = params['Rtip']

        af = self.af
        iterRe = 1
        B = params['B']
        bemoptions = self.bemoptions

        n = len(r)
        errf = self.__errorFunction
        rotating = (Omega != 0)

        self.dalpha_dx = np.zeros((n, 9))
        self.W = np.zeros(n)
        self.dW_dx = np.zeros((n, 9))
        self.dRe_dx = np.zeros((n, 9))

        phi_t = np.zeros(n)
        phi_dx_t = np.zeros((n, 9))

        for i in range(n):

            # index dependent arguments
            args = (r[i], chord[i], theta[i], af[i], Vx[i], Vy[i], iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions)

            if not rotating:  # non-rotating

                phi_star = pi/2.0

            else:

                # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------

                # set standard limits
                epsilon = 1e-6
                phi_lower = epsilon
                phi_upper = pi/2

                if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # an uncommon but possible case

                    if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
                        phi_lower = -pi/4
                        phi_upper = -epsilon
                    else:
                        phi_lower = pi/2
                        phi_upper = pi - epsilon

                try:
                    phi_star = brentq(errf, phi_lower, phi_upper, args=args)

                except ValueError:

                    warnings.warn('error.  check input values.')
                    phi_star = 0.0

            phi = phi_star
            if rotating:
                _, a, ap = self.__runBEM(phi, *args)
            else:
                a = 0.0
                ap = 0.0

            # derivative of residual function
            if rotating:
                dR_dx, da_dx, dap_dx = self.__residualDerivatives(phi, *args)
                dphi_dx = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                dR_dx = np.zeros(9)
                dR_dx[0] = 1.0  # just to prevent divide by zero
                da_dx = np.zeros(9)
                dap_dx = np.zeros(9)
                dphi_dx = np.zeros(9)

            # ssss = phi*(1/a * da_dx)
            # sss2 = phi*(1/ap * dap_dx)
            # if a != 0 and ap != 0:
            #     phi_dx = phi*(da_dx[0]**-1 * da_dx / a  + dap_dx[0]**-1 * dap_dx / ap)
            # elif a != 0:
            #     phi_dx = phi*(da_dx[0]**-1 * da_dx / a)
            # elif ap != 0:
            #     phi_dx = phi*(dap_dx[0]**-1 * dap_dx / ap)
            # else:
            #     phi_dx = np.zeros(9)

            # if a != 0 and ap != 0:
            #     phi_dx = (da_dx[0]**-1 * da_dx / a  + dap_dx[0]**-1 * dap_dx / ap)
            # elif a != 0:
            #     phi_dx = (da_dx[0]**-1 * da_dx / a)
            # elif ap != 0:
            #     phi_dx = (dap_dx[0]**-1 * dap_dx / ap)
            # else:
            #     phi_dx = np.zeros(9)

            # if a != 0 and ap != 0:
            #     phi_dx = (da_dx[0]**-1 * a  * da_dx + dap_dx[0]**-1 * dap_dx * ap )/2.0
            # elif a != 0:
            #     phi_dx = da_dx[0]**-1 * da_dx
            # elif ap != 0:
            #     phi_dx = dap_dx[0]**-1 * dap_dx
            # else:
            #     phi_dx = np.zeros(9)

            # a_dx[i] = da_dx
            # ap_dx[i] = dap_dx
            # if a != 0 and ap != 0:
            #     phi_dx = (da_dx  + dap_dx)/2.0
            # elif a != 0:
            #     phi_dx = da_dx
            # elif ap != 0:
            #     phi_dx = dap_dx
            # else:
            #     phi_dx = np.zeros(9)

            phi_dx_t[i] = dphi_dx
            phi_t[i] = phi_star
        unknowns['phi'] = phi_t
        self.phi_dx_t = phi_dx_t


    def jacobian(self, params, unknowns, resids):

        J = {}
        phi = unknowns['phi']
        dphi_dx = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        phi_dx = self.phi_dx_t

        # TODO check gradients for phi
        J['phi', 'chord'] = np.diag(phi_dx[:, 1])
        J['phi', 'theta'] = np.diag(phi_dx[:, 2])
        J['phi', 'Vx'] = np.diag(phi_dx[:, 3])
        J['phi', 'Vy'] = np.diag(phi_dx[:, 4])
        J['phi', 'r'] = np.diag(phi_dx[:, 5])
        J['phi', 'Rhub'] = phi_dx[:, 6]
        J['phi', 'Rtip'] = phi_dx[:, 7]
        J['phi', 'pitch'] = phi_dx[:, 8]

        return J


class Flow(Component):
    def __init__(self, af, bemoptions, n):
        super(Flow, self).__init__()

        self.add_param('pitch', shape=1)
        self.add_param('Rtip', shape=1)
        self.add_param('Vx', shape=n)
        self.add_param('Vy', shape=n)
        self.add_param('Omega', shape=1)
        self.add_param('r', val=np.zeros(n))
        self.add_param('chord', shape=n)
        self.add_param('theta', shape=n)
        self.add_param('rho', shape=1)
        self.add_param('mu', shape=1)
        self.add_param('Rhub', shape=1)
        self.add_param('phi', val=np.zeros(n))
        self.add_param('B', val=3, pass_by_obj=True)

        self.add_output('alpha', val=np.zeros(n))
        self.add_output('Re', val=np.zeros(n))
        self.add_output('W', val=np.zeros(n))

        self.af = af
        self.bemoptions = bemoptions
        self.fd_options['form'] = 'central'

    def __runBEM(self, phi, r, chord, theta, af, Vx, Vy, iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions):
        """residual of BEM method and other corresponding variables"""

        a = 0.0
        ap = 0.0

        if iterRe == 0.0:
            iterRe = int(iterRe)
        for i in range(iterRe):

            alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, pitch,
                                             chord, theta, rho, mu)
            cl, cd = af.evaluate(alpha, Re)

            fzero, a, ap = _bem.inductionfactors(r, chord, Rhub, Rtip, phi,
                                                 cl, cd, B, Vx, Vy, **bemoptions)

        return fzero, a, ap

    def __residualDerivatives(self, phi, r, chord, theta, af, Vx, Vy, iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions):
        """derivatives of fzero, a, ap"""

        if iterRe != 1:
            ValueError('Analytic derivatives not supplied for case with iterRe > 1')

        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)

        # alpha, Re (analytic derivaives)
        a = 0.0
        ap = 0.0
        alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, pitch,
                                         chord, theta, rho, mu)

        dalpha_dx = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        dRe_dx = np.array([0.0, Re/chord, 0.0, Re*Vx/W**2, Re*Vy/W**2, 0.0, 0.0, 0.0, 0.0])

        # cl, cd (spline derivatives)
        cl, cd = af.evaluate(alpha, Re)
        dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af.derivatives(alpha, Re)

        # chain rule
        dcl_dx = dcl_dalpha*dalpha_dx + dcl_dRe*dRe_dx
        dcd_dx = dcd_dalpha*dalpha_dx + dcd_dRe*dRe_dx

        # residual, a, ap (Tapenade)
        dx_dx = np.eye(9)

        fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(r, chord, Rhub, Rtip,
            phi, cl, cd, B, Vx, Vy, dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
            dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **bemoptions)

        return dR_dx, da_dx, dap_dx


    def solve_nonlinear(self, params, unknowns, resids):

        Vx = params['Vx']
        Vy = params['Vy']
        Omega = params['Omega']
        r = params['r']
        chord = params['chord']
        theta = params['theta']
        pitch = params['pitch']
        rho = params['rho']
        mu = params['mu']
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        B = params['B']

        af = self.af
        iterRe = 1
        bemoptions = self.bemoptions

        n = len(r)
        rotating = (Omega != 0)

        self.dalpha_dx = np.zeros((n, 9))
        self.W = np.zeros(n)
        self.dW_dx = np.zeros((n, 9))
        self.dRe_dx = np.zeros((n, 9))

        alpha_total = np.zeros(n)
        Re_total = np.zeros(n)

        for i in range(n):

            # index dependent arguments
            args = (r[i], chord[i], theta[i], af[i], Vx[i], Vy[i], iterRe, pitch, rho, mu, Rhub, Rtip, B, bemoptions)

            if not rotating:  # non-rotating

                phi = pi/2.0

            else:

                phi = params['phi'][i]

            if rotating:
                _, a, ap = self.__runBEM(phi, *args)
            else:
                a = 0.0
                ap = 0.0

            alpha, W, Re = _bem.relativewind(phi, a, ap, Vx[i], Vy[i], pitch,
                                             chord[i], theta[i], rho, mu)

            # derivative of residual function
            if rotating:
                dR_dx, da_dx, dap_dx = self.__residualDerivatives(phi, *args)
                dphi_dx = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                dR_dx = np.zeros(9)
                dR_dx[0] = 1.0  # just to prevent divide by zero
                da_dx = np.zeros(9)
                dap_dx = np.zeros(9)

            # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
            dx_dx = np.eye(9)

            # alpha, W, Re (Tapenade)

            alpha, dalpha_dx, W, dW_dx, Re, dRe_dx = _bem.relativewind_dv(phi, dx_dx[0, :],
                a, da_dx, ap, dap_dx, Vx[i], dx_dx[3, :], Vy[i], dx_dx[4, :],
                pitch, dx_dx[8, :], chord[i], dx_dx[1, :], theta[i], dx_dx[2, :],
                rho, mu)

            self.dalpha_dx[i] = dalpha_dx
            self.W[i] = W
            self.dW_dx[i] = dW_dx
            self.dRe_dx[i] = dRe_dx
            alpha_total[i] = alpha
            Re_total[i] = Re

        unknowns['alpha'] = alpha_total
        unknowns['Re'] = Re_total
        unknowns['W'] = self.W


        #         # stack
        # # z = [r, chord, theta, Rhub, Rtip, pitch]
        # # w = [r, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega]
        # # X = [r, chord, theta, Rhub, Rtip, presweep, precone, tilt, hubHt, yaw, azimuth, Uinf, Omega, pitch]
        # dNp_dz[0, :] += dNp_dw[0, :]  # add partial w.r.t. r
        # dTp_dz[0, :] += dTp_dw[0, :]
        #
        # dNp_dX = np.vstack((dNp_dz[:-1, :], dNp_dw[1:, :], dNp_dz[-1, :]))
        # dTp_dX = np.vstack((dTp_dz[:-1, :], dTp_dw[1:, :], dTp_dz[-1, :]))
        #
        # # add chain rule for conversion to radians
        # ridx = [2, 6, 7, 9, 10, 13]
        # dNp_dX[ridx, :] *= pi/180.0
        # dTp_dX[ridx, :] *= pi/180.0

    def jacobian(self, params, unknowns, resids):
        J = {}

        dalpha_dx = self.dalpha_dx
        dW_dx = self.dW_dx
        dRe_dx = self.dRe_dx
        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)

        J['alpha', 'phi'] = np.diag(dalpha_dx[:,0])
        J['alpha', 'chord'] = np.diag(dalpha_dx[:,1])
        J['alpha', 'theta'] = np.diag(dalpha_dx[:,2])
        J['alpha', 'Vx'] = np.diag(dalpha_dx[:,3])
        J['alpha', 'Vy'] = np.diag(dalpha_dx[:,4])
        J['alpha', 'r'] = np.diag(dalpha_dx[:,5])
        J['alpha', 'Rhub'] = dalpha_dx[:,6]
        J['alpha', 'Rtip'] = dalpha_dx[:,7]
        J['alpha', 'pitch'] = dalpha_dx[:,8]

        J['Re', 'phi'] = np.diag(dRe_dx[:,0])
        J['Re', 'chord'] = np.diag(dRe_dx[:,1])
        J['Re', 'theta'] = np.diag(dRe_dx[:,2])
        J['Re', 'Vx'] = np.diag(dRe_dx[:,3])
        J['Re', 'Vy'] = np.diag(dRe_dx[:,4])
        J['Re', 'r'] = np.diag(dRe_dx[:,5])
        J['Re', 'Rhub'] = dRe_dx[:,6]
        J['Re', 'Rtip'] = dRe_dx[:,7]
        J['Re', 'pitch'] = dRe_dx[:,8]

        J['W', 'phi'] = np.diag(dW_dx[:,0])
        J['W', 'chord'] = np.diag(dW_dx[:,1])
        J['W', 'theta'] = np.diag(dW_dx[:,2])
        J['W', 'Vx'] = np.diag(dW_dx[:,3])
        J['W', 'Vy'] = np.diag(dW_dx[:,4])
        J['W', 'r'] = np.diag(dW_dx[:,5])
        J['W', 'Rhub'] = dW_dx[:,6]
        J['W', 'Rtip'] = dW_dx[:,7]
        J['W', 'pitch'] = dW_dx[:,8]

        return J


class Airfoils(Component):
    """ Lift and drag coefficients
    """
    def __init__(self, af, n):
        super(Airfoils, self).__init__()
        self.add_param('alpha', val=np.zeros(n))
        self.add_param('Re', val=np.zeros(n))

        self.add_output('cl', val=np.zeros(n))
        self.add_output('cd', val=np.zeros(n))

        self.af = af
        self.fd_options['form'] = 'central'


    def solve_nonlinear(self, params, unknowns, resids):

        af = self.af

        alpha = params['alpha']
        Re = params['Re']

        cl_total = np.zeros(len(af))
        cd_total = np.zeros(len(af))

        for i in range(len(af)):

            cl, cd = af[i].evaluate(alpha[i], Re[i])
            cl_total[i] = cl
            cd_total[i] = cd

        unknowns['cl'] = cl_total
        unknowns['cd'] = cd_total

    def jacobian(self, params, unknowns, resids):

        J = {}

        af = self.af
        # af = params['af']
        alpha = params['alpha']
        Re = params['Re']

        dcl_dalpha = np.zeros(len(af))
        dcl_dRe = np.zeros(len(af))
        dcd_dalpha = np.zeros(len(af))
        dcd_dRe = np.zeros(len(af))

        for i in range(len(af)):
            # cl, cd (spline derivatives)
            dcl_dalpha[i], dcl_dRe[i], dcd_dalpha[i], dcd_dRe[i] = af[i].derivatives(alpha[i], Re[i])


        J['cl', 'alpha'] = np.diag(dcl_dalpha)
        J['cl', 'Re'] = np.diag(dcl_dRe)
        J['cd', 'alpha'] = np.diag(dcd_dalpha)
        J['cd', 'Re'] = np.diag(dcl_dRe)

        return J


class DistributedAeroLoads(Component):
    def __init__(self, af, n):
        super(DistributedAeroLoads, self).__init__()

        self.add_param('chord', shape=n)
        self.add_param('rho', shape=1)
        self.add_param('phi', val=np.zeros(n))
        self.add_param('cl', val=np.zeros(n))
        self.add_param('cd', val=np.zeros(n))
        self.add_param('W', val=np.zeros(n))

        self.add_output('Np', shape=n)
        self.add_output('Tp', shape=n)

        self.af = af
        self.fd_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):

        chord = params['chord']
        rho = params['rho']
        phi = params['phi']
        cl = params['cl']
        cd = params['cd']
        W = params['W']
        af = self.af

        n = len(af)

        Np = np.zeros(n)
        Tp = np.zeros(n)

        dNp_dcl = np.zeros(n)
        dTp_dcl = np.zeros(n)
        dNp_dcd = np.zeros(n)
        dTp_dcd = np.zeros(n)
        dNp_dphi = np.zeros(n)
        dTp_dphi = np.zeros(n)
        dNp_drho = np.zeros(n)
        dTp_drho = np.zeros(n)
        dNp_dW = np.zeros(n)
        dTp_dW = np.zeros(n)
        dNp_dchord = np.zeros(n)
        dTp_dchord = np.zeros(n)

        for i in range(n):
            cphi = cos(phi[i])
            sphi = sin(phi[i])

            cn = cl[i]*cphi + cd[i]*sphi  # these expressions should always contain drag
            ct = cl[i]*sphi - cd[i]*cphi

            q = 0.5*rho*W[i]**2
            Np[i] = cn*q*chord[i]
            Tp[i] = ct*q*chord[i]

            dNp_dcl[i] = cphi*q*chord[i]
            dTp_dcl[i] = sphi*q*chord[i]
            dNp_dcd[i] = sphi*q*chord[i]
            dTp_dcd[i] = -cphi*q*chord[i]
            dNp_dphi[i] = (-cl[i]*sphi + cd[i]*cphi)*q*chord[i]
            dTp_dphi[i] = (cl[i]*cphi + cd[i]*sphi)*q*chord[i]
            dNp_drho[i] = cn*q/rho*chord[i]
            dTp_drho[i] = ct*q/rho*chord[i]
            dNp_dW[i] = cn*0.5*rho*2*W[i]*chord[i]
            dTp_dW[i] = ct*0.5*rho*2*W[i]*chord[i]
            dNp_dchord[i] = cn*q
            dTp_dchord[i] = ct*q


        self.dNp_dcl = dNp_dcl
        self.dTp_dcl = dTp_dcl
        self.dNp_dcd = dNp_dcd
        self.dTp_dcd = dTp_dcd
        self.dNp_dphi = dNp_dphi
        self.dTp_dphi = dTp_dphi
        self.dNp_drho = dNp_drho
        self.dTp_drho = dTp_drho
        self.dNp_dW = dNp_dW
        self.dTp_dW = dTp_dW
        self.dNp_dchord = dNp_dchord
        self.dTp_dchord = dTp_dchord

        unknowns['Np'] = Np
        unknowns['Tp'] = Tp


    def jacobian(self, params, unknowns, resids):

        J = {}

        dNp_dcl = self.dNp_dcl
        dTp_dcl = self.dTp_dcl
        dNp_dcd = self.dNp_dcd
        dTp_dcd = self.dTp_dcd
        dNp_dphi = self.dNp_dphi
        dTp_dphi = self.dTp_dphi
        dNp_drho = self.dNp_drho
        dTp_drho = self.dTp_drho
        dNp_dW = self.dNp_dW
        dTp_dW = self.dTp_dW
        dNp_dchord = self.dNp_dchord
        dTp_dchord = self.dTp_dchord

        # # add chain rule for conversion to radians
        ## TODO: Check radian conversion
        # ridx = [2, 6, 7, 9, 10, 13]
        # dNp_dz[ridx, :] *= pi/180.0
        # dTp_dz[ridx, :] *= pi/180.0

        J['Np', 'cl'] = np.diag(dNp_dcl)
        J['Tp', 'cl'] = np.diag(dTp_dcl)
        J['Np', 'cd'] = np.diag(dNp_dcd)
        J['Tp', 'cd'] = np.diag(dTp_dcd)
        J['Np', 'phi'] = np.diag(dNp_dphi)
        J['Tp', 'phi'] = np.diag(dTp_dphi)
        J['Np', 'rho'] = dNp_drho
        J['Tp', 'rho'] = dTp_drho
        J['Np', 'W'] = np.diag(dNp_dW) # rho*W*cn*chord
        J['Tp', 'W'] = np.diag(dTp_dW)
        J['Np', 'chord'] = np.diag(dNp_dchord)
        J['Tp', 'chord'] = np.diag(dTp_dchord)

        return J

class CCEvaluate(Component):
    def __init__(self, af, n):
        super(CCEvaluate, self).__init__()

        self.add_param('Uinf', val=10.0)
        self.add_param('Rtip', val=63.)
        self.add_param('Omega', shape=1)
        self.add_param('r', val=np.zeros(n))
        self.add_param('B', val=3)
        self.add_param('precurve', shape=n)
        self.add_param('presweep', shape=n)
        self.add_param('presweepTip', shape=1)
        self.add_param('precurveTip', shape=1)
        self.add_param('rho', shape=1)
        self.add_param('precone', shape=1)
        self.add_param('Rhub', shape=1)
        self.add_param('nSector', shape=1)
        self.add_param('rotorR', shape=1)

        for i in range(8):
            self.add_param('Np'+str(i+1), val=np.zeros(n))
            self.add_param('Tp'+str(i+1), val=np.zeros(n))

        self.add_output('P', val=0.5)
        self.add_output('T', val=0.5)
        self.add_output('Q', val=0.5)
        self.add_output('CP', val=0.5)
        self.add_output('CT', val=0.5)
        self.add_output('CQ', val=0.5)

        self.af = af
        self.fd_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):
        r = params['r']
        precurve = params['precurve']
        presweep = params['presweep']
        precone = params['precone']
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        precurveTip = params['precurveTip']
        presweepTip = params['presweepTip']
        nSector = params['nSector']
        Uinf = params['Uinf']
        Omega = params['Omega']
        B = params['B']
        rho = params['rho']
        rotorR = params['rotorR']

        nsec = int(nSector)

        npts = 1 #len(Uinf)
        T = np.zeros(npts)
        Q = np.zeros(npts)

        n = len(r)

        dT_dr = np.zeros((1, n))
        dQ_dr = np.zeros((1, n))
        dP_dr = np.zeros((1, n))
        dT_dprecurve = np.zeros((1, n))
        dQ_dprecurve = np.zeros((1, n))
        dP_dprecurve = np.zeros((1, n))
        dT_dpresweep = np.zeros((1, n))
        dQ_dpresweep = np.zeros((1, n))
        dP_dpresweep = np.zeros((1, n))
        dT_dprecone = 0
        dQ_dprecone = 0
        dP_dprecone = 0
        dT_dRhub = 0
        dQ_dRhub = 0
        dP_dRhub = 0
        dT_dRtip = 0
        dQ_dRtip = 0
        dP_dRtip = 0
        dT_dprecurvetip = 0
        dQ_dprecurvetip = 0
        dP_dprecurvetip = 0
        dT_dpresweeptip = 0
        dQ_dpresweeptip = 0
        dP_dpresweeptip = 0

        Np = {}
        Tp = {}

        dT_dNp_t = {}
        dQ_dNp_t = {}
        dP_dNp_t = {}
        dT_dTp_t = {}
        dQ_dTp_t = {}
        dP_dTp_t = {}

        for i in range(8):
            Np['Np' + str(i+1)] = params['Np' + str(i+1)]
            Tp['Tp' + str(i+1)] = params['Tp' + str(i+1)]

        args = (r, precurve, presweep, precone,
            Rhub, Rtip, precurveTip, presweepTip)

        for i in range(npts):  # iterate across conditions

            for j in range(nsec):  # integrate across azimuth
                azimuth = 360.0*float(j)/nsec

                Np1 = Np['Np'+str(j+1)]
                Tp1 = Tp['Tp'+str(j+1)]

                # Np, Tp, dNp, dTp = self.distributedAeroLoads(Uinf[i], Omega[i], pitch[i], azimuth)

                Tsub, Qsub = _bem.thrusttorque(Np1, Tp1, *args)

                T[i] += B * Tsub / nsec
                Q[i] += B * Qsub / nsec

                Tb = np.array([1, 0])
                Qb = np.array([0, 1])
                Npb, Tpb, rb, precurveb, presweepb, preconeb, Rhubb, Rtipb, precurvetipb, presweeptipb = \
                    _bem.thrusttorque_bv(Np1, Tp1, r, precurve, presweep, precone, Rhub, Rtip, precurveTip, presweepTip, Tb, Qb)

                dT_dNp1 = Npb[0, :]
                dQ_dNp1 = Npb[1, :]
                dT_dTp1 = Tpb[0, :]
                dQ_dTp1 = Tpb[1, :]
                dP_dTp1 = Tpb[1, :] * Omega * pi / 30.0
                dP_dNp1 = Npb[1, :] * Omega * pi / 30.0
                dT_dr1 = rb[0, :]
                dQ_dr1 = rb[1, :]
                dP_dr1 = rb[1, :] * Omega * pi / 30.0
                dT_dprecurve1 = precurveb[0, :]
                dQ_dprecurve1 = precurveb[1, :]
                dP_dprecurve1 = precurveb[1, :] * Omega * pi / 30.0
                dT_dpresweep1 = presweepb[0, :]
                dQ_dpresweep1 = presweepb[1, :]
                dP_dpresweep1 = presweepb[1, :] * Omega * pi / 30.0
                dT_dprecone1 = preconeb[0]
                dQ_dprecone1 = preconeb[1]
                dP_dprecone1 = preconeb[1] * Omega * pi / 30.0
                dT_dRhub1 = Rhubb[0]
                dQ_dRhub1 = Rhubb[1]
                dP_dRhub1 = Rhubb[1] * Omega * pi / 30.0
                dT_dRtip1 = Rtipb[0]
                dQ_dRtip1 = Rtipb[1]
                dP_dRtip1 = Rtipb[1] * Omega * pi / 30.0
                dT_dprecurvetip1 = precurvetipb[0]
                dQ_dprecurvetip1 = precurvetipb[1]
                dP_dprecurvetip1 = precurvetipb[1] * Omega * pi / 30.0
                dT_dpresweeptip1 = presweeptipb[0]
                dQ_dpresweeptip1 = presweeptipb[1]
                dP_dpresweeptip1 = presweeptipb[1] * Omega * pi / 30.0

                dT_dNp = np.zeros((1, n))
                dT_dTp = np.zeros((1, n))
                dQ_dNp = np.zeros((1, n))
                dQ_dTp = np.zeros((1, n))
                dP_dNp = np.zeros((1, n))
                dP_dTp = np.zeros((1, n))

                dT_dNp += [x * B / nsec for x in dT_dNp1] # B * dT_dNp / nsec
                dT_dNp_t['Np'+str(j+1)] = dT_dNp
                dQ_dNp += [x * B / nsec for x in dQ_dNp1]
                dQ_dNp_t['Np'+str(j+1)] = dQ_dNp
                dP_dNp += [x * B / nsec for x in dP_dNp1]
                dP_dNp_t['Np'+str(j+1)] = dP_dNp
                dT_dTp += [x * B / nsec for x in dT_dTp1]
                dT_dTp_t['Tp'+str(j+1)] = dT_dTp
                dQ_dTp += [x * B / nsec for x in dQ_dTp1]
                dQ_dTp_t['Tp'+str(j+1)] = dQ_dTp
                dP_dTp += [x * B / nsec for x in dP_dTp1]
                dP_dTp_t['Tp'+str(j+1)] = dP_dTp
                dT_dr += [x * B / nsec for x in dT_dr1]
                dQ_dr += [x * B / nsec for x in dQ_dr1]
                dP_dr += [x * B / nsec for x in dP_dr1]
                dT_dprecurve += [x * B / nsec for x in dT_dprecurve1]
                dQ_dprecurve += [x * B / nsec for x in dQ_dprecurve1]
                dP_dprecurve += [x * B / nsec for x in dP_dprecurve1]
                dT_dpresweep += [x * B / nsec for x in dT_dpresweep1]
                dQ_dpresweep += [x * B / nsec for x in dQ_dpresweep1]
                dP_dpresweep += [x * B / nsec for x in dP_dpresweep1]
                dT_dprecone += dT_dprecone1 * B / nsec
                dQ_dprecone += dQ_dprecone1 * B / nsec
                dP_dprecone += dP_dprecone1 * B / nsec
                dT_dRhub += dT_dRhub1 * B / nsec
                dQ_dRhub += dQ_dRhub1 * B / nsec
                dP_dRhub += dP_dRhub1 * B / nsec
                dT_dRtip += dT_dRtip1 * B / nsec
                dQ_dRtip += dQ_dRtip1 * B / nsec
                dP_dRtip += dP_dRtip1 * B / nsec
                dT_dprecurvetip += dT_dprecurvetip1 * B / nsec
                dQ_dprecurvetip += dQ_dprecurvetip1 * B / nsec
                dP_dprecurvetip += dP_dprecurvetip1 * B / nsec
                dT_dpresweeptip += dT_dpresweeptip1 * B / nsec
                dQ_dpresweeptip += dQ_dpresweeptip1 * B / nsec
                dP_dpresweeptip += dP_dpresweeptip1 * B / nsec

        # Power
        P = Q * Omega*pi/30.0  # RPM to rad/s

        # normalize if necessary
        q = 0.5 * rho * Uinf**2
        A = pi * rotorR**2
        CP = P / (q * A * Uinf)
        CT = T / (q * A)
        CQ = Q / (q * rotorR * A)

        dCP_drho = CP * rho
        dCT_drho = CT * rho
        dCQ_drho = CQ * rho
        dCP_drotorR = -2 * P / (q * pi * rotorR**3 * Uinf)
        dCT_drotorR = -2 * T / (q * pi * rotorR**3)
        dCQ_drotorR = -3 * Q / (q * pi * rotorR**4)
        dCP_dUinf = -3 * P / (0.5 * rho * Uinf**4 * A)
        dCT_dUinf = -2 * T / (0.5 * rho * Uinf**3 * A)
        dCQ_dUinf = -2 * Q / (0.5 * rho * Uinf**3 * rotorR * A)
        dP_dOmega = Q * pi / 30.0

        self.dCP_drho = dCP_drho
        self.dCT_drho = dCT_drho
        self.dCQ_drho = dCQ_drho
        self.dCP_drotorR = dCP_drotorR
        self.dCT_drotorR = dCT_drotorR
        self.dCQ_drotorR = dCQ_drotorR
        self.dCP_dUinf = dCP_dUinf
        self.dCT_dUinf = dCT_dUinf
        self.dCQ_dUinf = dCQ_dUinf
        self.dP_dOmega = dP_dOmega
        self.dT_dNp_t = dT_dNp_t
        self.dT_dTp_t = dT_dTp_t
        self.dQ_dNp_t = dQ_dNp_t
        self.dQ_dTp_t = dQ_dTp_t
        self.dP_dNp_t = dP_dNp_t
        self.dP_dTp_t = dP_dTp_t
        self.dT_dr = dT_dr
        self.dQ_dr = dQ_dr
        self.dP_dr = dP_dr
        self.dT_dprecurve = dT_dprecurve
        self.dQ_dprecurve = dQ_dprecurve
        self.dP_dprecurve = dP_dprecurve
        self.dT_dpresweep = dT_dpresweep
        self.dQ_dpresweep = dQ_dpresweep
        self.dP_dpresweep = dP_dpresweep
        self.dT_dprecone = dT_dprecone
        self.dQ_dprecone = dQ_dprecone
        self.dP_dprecone = dP_dprecone
        self.dT_dRhub = dT_dRhub
        self.dQ_dRhub = dQ_dRhub
        self.dP_dRhub = dP_dRhub
        self.dT_dRtip = dT_dRtip
        self.dQ_dRtip = dQ_dRtip
        self.dP_dRtip = dP_dRtip
        self.dT_dprecurvetip = dT_dprecurvetip
        self.dQ_dprecurvetip = dQ_dprecurvetip
        self.dP_dprecurvetip = dP_dprecurvetip
        self.dT_dpresweeptip = dT_dpresweeptip
        self.dQ_dpresweeptip = dQ_dpresweeptip
        self.dP_dpresweeptip = dP_dpresweeptip


        unknowns['CP'] = CP[0]
        unknowns['CT'] = CT[0]
        unknowns['CQ'] = CQ[0]
        unknowns['P'] = P[0]
        unknowns['T'] = T[0]
        unknowns['Q'] = Q[0]

    def jacobian(self, params, unknowns, resids):
        J = {}

        dCP_drho = self.dCP_drho
        dCT_drho = self.dCT_drho
        dCQ_drho = self.dCQ_drho
        dCP_drotorR = self.dCP_drotorR
        dCT_drotorR = self.dCT_drotorR
        dCQ_drotorR = self.dCQ_drotorR

        dCP_dUinf = self.dCP_dUinf
        dCT_dUinf = self.dCT_dUinf
        dCQ_dUinf = self.dCQ_dUinf

        dP_dOmega = self.dP_dOmega

        rho = params['rho']
        Uinf = params['Uinf']
        rotorR = params['rotorR']

        q = 0.5 * rho * Uinf**2
        A = pi * rotorR**2

        nSector = int(params['nSector'])

        for i in range(nSector):
            J['CP', 'Np'+str(i+1)] = self.dP_dNp_t['Np'+str(i+1)] / (q * A * Uinf)
            J['CP', 'Tp'+str(i+1)] = self.dP_dTp_t['Tp'+str(i+1)] / (q * A * Uinf)
            J['CT', 'Np'+str(i+1)] = self.dT_dNp_t['Np'+str(i+1)] / (q * A)
            J['CT', 'Tp'+str(i+1)] = self.dT_dTp_t['Tp'+str(i+1)] / (q * A)
            J['CQ', 'Np'+str(i+1)] = self.dQ_dNp_t['Np'+str(i+1)] / (q * rotorR * A)
            J['CQ', 'Tp'+str(i+1)] = self.dQ_dTp_t['Tp'+str(i+1)] / (q * rotorR * A)
            J['P', 'Np'+str(i+1)] = self.dP_dNp_t['Np'+str(i+1)]
            J['P', 'Tp'+str(i+1)] = self.dP_dTp_t['Tp'+str(i+1)]
            J['T', 'Np'+str(i+1)] = self.dT_dNp_t['Np'+str(i+1)]
            J['T', 'Tp'+str(i+1)] = self.dT_dTp_t['Tp'+str(i+1)]
            J['Q', 'Np'+str(i+1)] = self.dQ_dNp_t['Np'+str(i+1)]
            J['Q', 'Tp'+str(i+1)] = self.dQ_dTp_t['Tp'+str(i+1)]

        J['CP', 'Uinf'] = dCP_dUinf
        J['CP', 'Rtip'] = self.dP_dRtip / (q * A * Uinf)
        J['CP', 'Omega'] = dP_dOmega / (q * A * Uinf)
        J['CP', 'r'] = self.dP_dr / (q * A * Uinf)
        J['CP', 'precurve'] = self.dP_dprecurve / (q * A * Uinf)
        J['CP', 'presweep'] = self.dP_dpresweep / (q * A * Uinf)
        J['CP', 'presweepTip'] = self.dP_dpresweeptip / (q * A * Uinf)
        J['CP', 'precurveTip'] = self.dP_dprecurvetip / (q * A * Uinf)
        J['CP', 'precone'] = self.dP_dprecone / (q * A * Uinf)
        J['CP', 'rho'] = dCP_drho
        J['CP', 'Rhub'] = self.dP_dRhub / (q * A * Uinf)
        J['CP', 'rotorR'] = dCP_drotorR

        J['CT', 'Uinf'] = dCT_dUinf
        J['CT', 'Rtip'] = self.dT_dRtip / (q * A)
        J['CT', 'Omega'] = 0
        J['CT', 'r'] = (self.dT_dr / (q * A))
        J['CT', 'precurve'] = self.dT_dprecurve / (q * A)
        J['CT', 'presweep'] = self.dT_dpresweep / (q * A)
        J['CT', 'presweepTip'] = self.dT_dpresweeptip / (q * A)
        J['CT', 'precurveTip'] = self.dT_dprecurvetip / (q * A)
        J['CT', 'precone'] = self.dT_dprecone / (q * A)
        J['CT', 'rho'] = dCT_drho
        J['CT', 'Rhub'] = self.dT_dRhub / (q * A)
        J['CT', 'rotorR'] = dCT_drotorR


        J['CQ', 'Uinf'] = dCQ_dUinf
        J['CQ', 'Rtip'] = self.dQ_dRtip / (q * rotorR * A)
        J['CQ', 'Omega'] = 0
        J['CQ', 'r'] = self.dQ_dr /  (q * rotorR * A)
        J['CQ', 'precurve'] = self.dQ_dprecurve /  (q * rotorR * A)
        J['CQ', 'presweep'] = self.dQ_dpresweep/  (q * rotorR * A)
        J['CQ', 'presweepTip'] = self.dQ_dpresweeptip /  (q * rotorR * A)
        J['CQ', 'precurveTip'] = self.dQ_dprecurvetip / (q * rotorR * A)
        J['CQ', 'precone'] = self.dQ_dprecone / (q * rotorR * A)
        J['CQ', 'rho'] = dCQ_drho
        J['CQ', 'Rhub'] = self.dQ_dRhub /  (q * rotorR * A)
        J['CQ', 'rotorR'] = dCQ_drotorR



        # J['P', 'Uinf'] =
        J['P', 'Rtip'] = self.dP_dRtip
        J['P', 'Omega'] = dP_dOmega
        J['P', 'r'] = self.dP_dr
        J['P', 'precurve'] = self.dP_dprecurve
        J['P', 'presweep'] = self.dP_dpresweep
        J['P', 'presweepTip'] = self.dP_dpresweeptip
        J['P', 'precurveTip'] = self.dP_dprecurvetip
        J['P', 'precone'] = self.dP_dprecone
        # J['P', 'rho'] = 1
        J['P', 'Rhub'] = self.dP_dRhub
        # J['P', 'rotorR'] = 1


        # J['T', 'Uinf'] = 1
        J['T', 'Rtip'] = self.dT_dRtip
        # J['T', 'Omega'] = 1
        J['T', 'r'] = self.dT_dr
        J['T', 'precurve'] = self.dT_dprecurve
        J['T', 'presweep'] = self.dT_dpresweep
        J['T', 'presweepTip'] = self.dT_dpresweeptip
        J['T', 'precurveTip'] = self.dT_dprecurvetip
        J['T', 'precone'] = self.dT_dprecone
        # J['T', 'rho'] = 1
        J['T', 'Rhub'] = self.dT_dRhub
        J['T', 'rotorR'] = 0


        # J['Q', 'Uinf'] = 1
        J['Q', 'Rtip'] = self.dQ_dRtip
        # J['Q', 'Omega'] = 1
        J['Q', 'r'] = self.dQ_dr
        J['Q', 'precurve'] = self.dQ_dprecurve
        J['Q', 'presweep'] = self.dQ_dpresweep
        J['Q', 'presweepTip'] = self.dQ_dpresweeptip
        J['Q', 'precurveTip'] = self.dQ_dprecurvetip
        J['Q', 'precone'] = self.dQ_dprecone
        # J['Q', 'rho'] = 1
        J['Q', 'Rhub'] = self.dQ_dRhub
        J['Q', 'rotorR'] = 0

        return J


class Sweep(Group):
    def __init__(self, azimuth, n, af, bemoptions):
        super(Sweep, self).__init__()

        self.add('azimuth', IndepVarComp('azimuth', azimuth), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('angles', Angles(af, bemoptions, n), promotes=['*'])
        self.add('flow', Flow(af, bemoptions, n), promotes=['*'])
        self.add('bem', Airfoils(af, n), promotes=['*'])
        self.add('loads', DistributedAeroLoads(af, n), promotes=['chord', 'rho', 'phi', 'cl', 'cd', 'W'])

class SweepGroup(Group):
    def __init__(self, af, nSector, bemoptions):
        super(SweepGroup, self).__init__()

        r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                      28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                      56.1667, 58.9000, 61.6333])

        n = len(r)
        self.add('r', IndepVarComp('r', np.zeros(17)), promotes=['*'])
        self.add('chord', IndepVarComp('chord', np.zeros(17)), promotes=['*'])
        self.add('theta', IndepVarComp('theta', np.zeros(17)), promotes=['*'])
        self.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        self.add('Uinf', IndepVarComp('Uinf', 0.0), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', 0.0), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', np.zeros(17)), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', np.zeros(17)), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0), promotes=['*'])

        self.add('init', CCInit(nSector), promotes=['*'])

        for i in range(nSector):
            azimuth = pi/180.0*360.0*float(i)/nSector
            self.add('group'+str(i+1), Sweep(azimuth, n, af, bemoptions), promotes=['Uinf', 'pitch', 'Rtip', 'Omega', 'r', 'chord', 'theta', 'rho', 'mu', 'Rhub', 'hubHt', 'precurve', 'presweep', 'precone', 'tilt', 'yaw', 'pitch', 'shearExp', 'B'])

class CCBlade(Group):

    def __init__(self, af, nSector, bemoptions):
        super(CCBlade, self).__init__()

        n = len(af)

        self.add('load_group', SweepGroup(af, nSector, bemoptions), promotes=['Uinf', 'tsr', 'pitch', 'Rtip', 'Omega', 'r', 'chord', 'theta', 'rho', 'mu', 'Rhub', 'nSector', 'rotorR', 'precurve', 'presweep', 'precurveTip', 'presweepTip', 'precone', 'tilt', 'yaw', 'pitch', 'shearExp', 'hubHt', 'B'])
        self.add('eval', CCEvaluate(af, n), promotes=['Uinf', 'Rtip', 'Omega', 'r', 'Rhub', 'B', 'precurve', 'presweep', 'presweepTip', 'precurveTip', 'precone', 'nSector', 'rotorR', 'rho', 'CP', 'CT', 'CQ', 'P', 'T', 'Q'])

        for i in range(8):
            self.connect('load_group.group' + str(i+1) + '.loads.Np', 'eval.Np' + str(i+1))
            self.connect('load_group.group' + str(i+1) + '.loads.Tp', 'eval.Tp' + str(i+1))

        self.add('obj_cmp', ExecComp('obj = -CP', CP=1.0), promotes=['*'])

class LoadsGroup(Group):
    def __init__(self, af, nSector):
        super(LoadsGroup, self).__init__()

        n = len(af)

        bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)

        self.add('r', IndepVarComp('r', np.zeros(17)), promotes=['*'])
        self.add('chord', IndepVarComp('chord', np.zeros(17)), promotes=['*'])
        self.add('theta', IndepVarComp('theta', np.zeros(17)), promotes=['*'])
        self.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        self.add('Uinf', IndepVarComp('Uinf', 0.0), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', 0.0), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', np.zeros(17)), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', np.zeros(17)), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0), promotes=['*'])
        self.add('azimuth', IndepVarComp('azimuth', 0.0), promotes=['*'])

        self.add('init', CCInit(nSector), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('angles', Angles(af, bemoptions, n), promotes=['*'])
        self.add('flow', Flow(af, bemoptions, n), promotes=['*'])
        self.add('bem', Airfoils(af, n), promotes=['*'])
        self.add('loads', DistributedAeroLoads(af, n), promotes=['*'])

# class WindGroup(Group):
#     def __init__(self, azimuth, n):
#         super(WindGroup, self).__init__()
#         self.add('wind', WindComponents(azimuth, n), promotes=['*'])

if __name__ == "__main__":

    # geometry
    Rhub = 1.5
    Rtip = 63.0

    r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
                  28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
                  56.1667, 58.9000, 61.6333])
    chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                      3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
    theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                      6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
    B = 3  # number of blades
    iterRe = 1
    bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)

    # atmosphere
    rho = 1.225
    mu = 1.81206e-5

    import os
    afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
    basepath = '5MW_AFFiles' + os.path.sep

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
    airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
    airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
    airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
    airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
    airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    af = [0]*len(r)
    for i in range(len(r)):
        af[i] = airfoil_types[af_idx[i]]


    tilt = -5.0
    precone = 2.5
    yaw = 0.0
    shearExp = 0.2
    hubHt = 80.0
    nSector = 8

    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 90.

    ccblade = Problem()
    root = ccblade.root = CCBlade(af, nSector, bemoptions)

    ### SETUP OPTIMIZATION
    # CCBlade.driver = pyOptSparseDriver()
    # CCBlade.driver.options['optimizer'] = 'SNOPT' #'SLSQP'
    # # CCBlade.driver.options['tol'] = 1.0e-8
    #
    # CCBlade.driver.add_desvar('tsr', low=1.5,
    #          high=14.0)
    #
    # CCBlade.driver.add_objective('obj')
    #
    # recorder = SqliteRecorder('para')
    # recorder.options['record_params'] = True
    # recorder.options['record_metadata'] = True
    # CCBlade.driver.add_recorder(recorder)

    ccblade.setup()

    ccblade['Rhub'] = Rhub
    ccblade['Rtip'] = Rtip
    ccblade['r'] = r
    ccblade['chord'] = chord
    ccblade['theta'] = np.radians(theta)
    ccblade['B'] = B
    ccblade['rho'] = rho
    ccblade['mu'] = mu
    ccblade['tilt'] = np.radians(tilt)
    ccblade['precone'] = np.radians(precone)
    ccblade['yaw'] = np.radians(yaw)
    ccblade['shearExp'] = shearExp
    ccblade['hubHt'] = hubHt
    ccblade['nSector'] = nSector
    ccblade['Uinf'] = Uinf
    ccblade['tsr'] = tsr
    ccblade['pitch'] = np.radians(pitch)

    ccblade.run()

    # test_grad = open('partial_test_grad.txt', 'w')
    # power_gradients = ccblade.check_total_derivatives_modified2(out_stream=test_grad)
    # power_partial = ccblade.check_partial_derivatives(out_stream=test_grad)

    print 'CP', root.eval.unknowns['CP']
