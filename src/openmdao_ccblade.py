__author__ = 'ryanbarr'

import warnings
from math import cos, sin, pi
import numpy as np
import _bem
from openmdao.api import Component, ExecComp, IndepVarComp, Group, Problem, SqliteRecorder, ScipyGMRES
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from zope.interface import Interface, implements
from scipy.interpolate import RectBivariateSpline, bisplev
from airfoilprep import Airfoil
from brent import Brent

class CCInit(Component):

    """
    CCInit

    Initializes all the inputs and performs a few simple calculations as preparation for analysis

    """

    def __init__(self):
        super(CCInit, self).__init__()

        self.add_param('Rtip', val=0.0)
        self.add_param('precone', val=0.0)
        self.add_param('precurveTip', val=0.0)
        self.add_param('Uinf', val=0.0)
        self.add_param('tsr', val=0.0)

        self.add_output('rotorR', shape=1)
        self.add_output('Omega', shape=1)
        # self.add_output('rotating', val=True, pass_by_obj=True)

        self.fd_options['form'] = 'central'

    def solve_nonlinear(self, params, unknowns, resids):

        precone = params['precone']
        Rtip = params['Rtip']
        precurveTip = params['precurveTip']

        # rotor radius
        unknowns['rotorR'] = Rtip*cos(precone) + precurveTip*sin(precone)

        # tip speed ratio
        Uinf = params['Uinf']
        Rtip = params['Rtip']
        unknowns['Omega'] = Uinf*params['tsr']/Rtip * 30.0/pi

        # rotating
        # unknowns['rotating'] = (unknowns['Omega'] != 0)

    def linearize(self, params, unknowns, resids):

        J = {}

        precone = params['precone']
        precurveTip = params['precurveTip']
        Rtip = params['Rtip']
        tsr = params['tsr']
        Uinf = params['Uinf']

        J['rotorR', 'precone'] = -Rtip*sin(precone) + precurveTip*cos(precone)
        J['rotorR', 'precurveTip'] = sin(precone)
        J['rotorR', 'Rtip'] = cos(precone)
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

    def linearize(self, params, unknowns, resids):
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

class FlowCondition(Component):
    def __init__(self):
        super(FlowCondition, self).__init__()
        self.add_param('pitch', shape=1)
        self.add_param('Vx', shape=1)
        self.add_param('Vy', shape=1)
        self.add_param('chord', shape=1)
        self.add_param('theta', shape=1)
        self.add_param('rho', shape=1)
        self.add_param('mu', shape=1)
        self.add_param('a_sub', shape=1)
        self.add_param('phi_sub', shape=1)
        self.add_param('ap_sub', shape=1)

        self.add_output('alpha_sub', shape=1)
        self.add_output('Re_sub', shape=1)
        self.add_output('W_sub', shape=1)


    def solve_nonlinear(self, params, unknowns, resids):
        Vx = params['Vx']
        Vy = params['Vy']
        chord = params['chord']
        theta = params['theta']
        pitch = params['pitch']
        rho = params['rho']
        mu = params['mu']
        phi = params['phi_sub']
        a = params['a_sub']
        ap = params['ap_sub']

        alpha, W, Re = _bem.relativewind(phi, a, ap, Vx, Vy, pitch, chord, theta, rho, mu)

        dalpha_dx = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        dRe_dx = np.array([0.0, Re/chord, 0.0, Re*Vx/W**2, Re*Vy/W**2, 0.0, 0.0, 0.0, 0.0])
        dW_dx = np.zeros(len(dRe_dx))

        self.dalpha_dx = dalpha_dx
        self.dW_dx = dW_dx
        self.dRe_dx = dRe_dx

        unknowns['alpha_sub'] = alpha
        unknowns['W_sub'] = W
        unknowns['Re_sub'] = Re

    def linearize(self, params, unknowns, resids):

        J = {}

        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
        dalpha_dx = self.dalpha_dx
        dW_dx = self.dW_dx
        dRe_dx = self.dRe_dx

        J['alpha_sub', 'phi_sub'] = dalpha_dx[0]
        J['alpha_sub', 'chord'] = dalpha_dx[1]
        J['alpha_sub', 'theta'] = dalpha_dx[2]
        J['alpha_sub', 'Vx'] = dalpha_dx[3]
        J['alpha_sub', 'Vy'] = dalpha_dx[4]
        J['alpha_sub', 'r'] = dalpha_dx[5]
        J['alpha_sub', 'Rhub'] = dalpha_dx[6]
        J['alpha_sub', 'Rtip'] = dalpha_dx[7]
        J['alpha_sub', 'pitch'] = dalpha_dx[8]

        J['Re_sub', 'phi_sub'] = dRe_dx[0]
        J['Re_sub', 'chord'] = dRe_dx[1]
        J['Re_sub', 'theta'] = dRe_dx[2]
        J['Re_sub', 'Vx'] = dRe_dx[3]
        J['Re_sub', 'Vy'] = dRe_dx[4]
        J['Re_sub', 'r'] = dRe_dx[5]
        J['Re_sub', 'Rhub'] = dRe_dx[6]
        J['Re_sub', 'Rtip'] = dRe_dx[7]
        J['Re_sub', 'pitch'] = dRe_dx[8]

        J['W_sub', 'phi_sub'] = dW_dx[0]
        J['W_sub', 'chord'] = dW_dx[1]
        J['W_sub', 'theta'] = dW_dx[2]
        J['W_sub', 'Vx'] = dW_dx[3]
        J['W_sub', 'Vy'] = dW_dx[4]
        J['W_sub', 'r'] = dW_dx[5]
        J['W_sub', 'Rhub'] = dW_dx[6]
        J['W_sub', 'Rtip'] = dW_dx[7]
        J['W_sub', 'pitch'] = dW_dx[8]

        return J

class AirfoilComp(Component):
    """ Lift and drag coefficients
    """
    def __init__(self, n, i):
        super(AirfoilComp, self).__init__()
        self.add_param('alpha_sub', shape=1)
        self.add_param('Re_sub', shape=1)
        self.add_param('af', val=np.zeros(n), pass_by_obj=True)

        self.add_output('cl_sub', shape=1)
        self.add_output('cd_sub', shape=1)

        self.fd_options['form'] = 'central'
        self.i = i

    def solve_nonlinear(self, params, unknowns, resids):

        alpha = params['alpha_sub']
        Re = params['Re_sub']
        af = params['af']

        cl, cd = af[self.i].evaluate(alpha, Re)

        unknowns['cl_sub'] = cl
        unknowns['cd_sub'] = cd

    def linearize(self, params, unknowns, resids):

        J = {}

        af = params['af']
        alpha = params['alpha_sub']
        Re = params['Re_sub']

        dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af[self.i].derivatives(alpha, Re)

        J['cl_sub', 'alpha_sub'] = dcl_dalpha
        J['cl_sub', 'Re_sub'] = dcl_dRe
        J['cd_sub', 'alpha_sub'] = dcd_dalpha
        J['cd_sub', 'Re_sub'] = dcl_dRe

        return J

class BEM(Component):
    def __init__(self, n, i):
        super(BEM, self).__init__()

        self.add_param('pitch', shape=1)
        self.add_param('Rtip', shape=1)
        self.add_param('Vx', shape=1)
        self.add_param('Vy', shape=1)
        self.add_param('Omega', shape=1)
        self.add_param('r', shape=1)
        self.add_param('chord', shape=1)
        self.add_param('theta', shape=1)
        self.add_param('rho', shape=1)
        self.add_param('mu', shape=1)
        self.add_param('Rhub', shape=1)
        self.add_param('alpha_sub', shape=1)
        self.add_param('Re_sub', shape=1)
        self.add_param('W_sub', shape=1)
        self.add_param('cl_sub', val=1.0)
        self.add_param('cd_sub', shape=1)
        self.add_param('B', val=3, pass_by_obj=True)
        self.add_param('af', val=np.zeros(n), pass_by_obj=True)
        self.add_param('bemoptions', val={}, pass_by_obj=True)
        self.add_output('a_sub', shape=1)
        self.add_output('ap_sub', shape=1)

        self.add_state('phi_sub', shape=1)

        self.fd_options['form'] = 'central'
        self.i = i

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def apply_nonlinear(self, params, unknowns, resids):

        Vx = params['Vx']
        Vy = params['Vy']
        Omega = params['Omega']
        r = params['r']
        chord = params['chord']
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        B = params['B']
        bemoptions = params['bemoptions']
        cl = params['cl_sub']
        cd = params['cd_sub']
        Re = params['Re_sub']
        alpha = params['alpha_sub']
        W = params['W_sub']
        af = params['af']

        rotating = (Omega != 0)

        dalpha_dx = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        dRe_dx = np.array([0.0, Re/chord, 0.0, Re*Vx/W**2, Re*Vy/W**2, 0.0, 0.0, 0.0, 0.0])

        dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = af[self.i].derivatives(alpha, Re)

        # chain rule
        dcl_dx = dcl_dalpha*dalpha_dx + dcl_dRe*dRe_dx
        dcd_dx = dcd_dalpha*dalpha_dx + dcd_dRe*dRe_dx

        # residual, a, ap (Tapenade)
        dx_dx = np.eye(9)

        fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(r, chord, Rhub, Rtip,
            unknowns['phi_sub'], cl, cd, B, Vx, Vy, dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
            dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **bemoptions)

        fzero, a, ap = _bem.inductionfactors(r, chord, Rhub, Rtip, unknowns['phi_sub'],
                                                 cl, cd, B, Vx, Vy, **bemoptions)
        resids['phi_sub'] = fzero
        unknowns['a_sub'] = a
        unknowns['ap_sub'] = ap

        if not rotating:
            dR_dx = np.zeros(9)
            dR_dx[0] = 1.0  # just to prevent divide by zero

        self.dR_dx = dR_dx
        self.da_dx = da_dx
        self.dap_dx = dap_dx

    def linearize(self, params, unknowns, resids):
        J = {}

        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
        dR_dx = self.dR_dx
        da_dx = self.da_dx
        dap_dx = self.dap_dx

        J['phi_sub', 'phi_sub'] = dR_dx[0]
        J['phi_sub', 'chord'] = dR_dx[1]
        J['phi_sub', 'theta'] = dR_dx[2]
        J['phi_sub', 'Vx'] = dR_dx[3]
        J['phi_sub', 'Vy'] = dR_dx[4]
        J['phi_sub', 'r'] = dR_dx[5]
        J['phi_sub', 'Rhub'] = dR_dx[6]
        J['phi_sub', 'Rtip'] = dR_dx[7]
        J['phi_sub', 'pitch'] = dR_dx[8]

        J['a_sub', 'phi_sub'] = da_dx[0]
        J['a_sub', 'chord'] = da_dx[1]
        J['a_sub', 'theta'] = da_dx[2]
        J['a_sub', 'Vx'] = da_dx[3]
        J['a_sub', 'Vy'] = da_dx[4]
        J['a_sub', 'r'] = da_dx[5]
        J['a_sub', 'Rhub'] = da_dx[6]
        J['a_sub', 'Rtip'] = da_dx[7]
        J['a_sub', 'pitch'] = da_dx[8]

        J['ap_sub', 'phi_sub'] = dap_dx[0]
        J['ap_sub', 'chord'] = dap_dx[1]
        J['ap_sub', 'theta'] = dap_dx[2]
        J['ap_sub', 'Vx'] = dap_dx[3]
        J['ap_sub', 'Vy'] = dap_dx[4]
        J['ap_sub', 'r'] = dap_dx[5]
        J['ap_sub', 'Rhub'] = dap_dx[6]
        J['ap_sub', 'Rtip'] = dap_dx[7]
        J['ap_sub', 'pitch'] = dap_dx[8]

        return J

class MUX(Component):
    def __init__(self, n):
        super(MUX, self).__init__()
        for i in range(n):
            self.add_param('phi'+str(i+1), val=0.0)
            self.add_param('cl'+str(i+1), val=0.0)
            self.add_param('cd'+str(i+1), val=0.0)
            self.add_param('W'+str(i+1), val=0.0)
        self.add_output('phi', val=np.zeros(n))
        self.add_output('cl', val=np.zeros(n))
        self.add_output('cd', val=np.zeros(n))
        self.add_output('W', val=np.zeros(n))
        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):
        n = self.n
        for i in range(n):
            unknowns['phi'][i] = params['phi'+str(i+1)]
            unknowns['cl'][i] = params['cl'+str(i+1)]
            unknowns['cd'][i] = params['cd'+str(i+1)]
            unknowns['W'][i] = params['W'+str(i+1)]

    def linearize(self, params, unknowns, resids):
        n = self.n
        J = {}
        for i in range(n):
            zeros = np.zeros(n)
            zeros[i] = 1
            J['phi', 'phi'+str(i+1)] = zeros
            J['cl', 'cl'+str(i+1)] = zeros
            J['cd', 'cd'+str(i+1)] = zeros
            J['W', 'W'+str(i+1)] = zeros

        return J

class DistributedAeroLoads(Component):
    def __init__(self, n):
        super(DistributedAeroLoads, self).__init__()

        self.add_param('chord', shape=n)
        self.add_param('rho', shape=1)
        self.add_param('phi', val=np.zeros(n))
        self.add_param('cl', val=np.zeros(n))
        self.add_param('cd', val=np.zeros(n))
        self.add_param('W', val=np.zeros(n))

        self.add_output('Np', shape=n)
        self.add_output('Tp', shape=n)

        self.fd_options['form'] = 'central'
        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):

        chord = params['chord']
        rho = params['rho']
        phi = params['phi']
        cl = params['cl']
        cd = params['cd']
        W = params['W']
        n = self.n

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

    def linearize(self, params, unknowns, resids):

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
    def __init__(self, n):
        super(CCEvaluate, self).__init__()

        self.add_param('Uinf', val=10.0)
        self.add_param('Rtip', val=63.)
        self.add_param('Omega', shape=1)
        self.add_param('r', shape=n)
        self.add_param('B', val=3, pass_by_obj=True)
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

        self.fd_options['form'] = 'central'
        self.n = n

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

        n = self.n

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
                Np1 = Np['Np'+str(j+1)]
                Tp1 = Tp['Tp'+str(j+1)]

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

    def linearize(self, params, unknowns, resids):
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
        J['CT', 'Omega'] = 0.0
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
        J['CQ', 'Omega'] = 0.0
        J['CQ', 'r'] = self.dQ_dr / (q * rotorR * A)
        J['CQ', 'precurve'] = self.dQ_dprecurve / (q * rotorR * A)
        J['CQ', 'presweep'] = self.dQ_dpresweep/ (q * rotorR * A)
        J['CQ', 'presweepTip'] = self.dQ_dpresweeptip / (q * rotorR * A)
        J['CQ', 'precurveTip'] = self.dQ_dprecurvetip / (q * rotorR * A)
        J['CQ', 'precone'] = self.dQ_dprecone / (q * rotorR * A)
        J['CQ', 'rho'] = dCQ_drho
        J['CQ', 'Rhub'] = self.dQ_dRhub / (q * rotorR * A)
        J['CQ', 'rotorR'] = dCQ_drotorR

        J['P', 'Uinf'] = 0.0
        J['P', 'Rtip'] = self.dP_dRtip
        J['P', 'Omega'] = dP_dOmega
        J['P', 'r'] = self.dP_dr
        J['P', 'precurve'] = self.dP_dprecurve
        J['P', 'presweep'] = self.dP_dpresweep
        J['P', 'presweepTip'] = self.dP_dpresweeptip
        J['P', 'precurveTip'] = self.dP_dprecurvetip
        J['P', 'precone'] = self.dP_dprecone
        J['P', 'rho'] = 0.0
        J['P', 'Rhub'] = self.dP_dRhub
        J['P', 'rotorR'] = 0.0

        J['T', 'Uinf'] = 0.0
        J['T', 'Rtip'] = self.dT_dRtip
        J['T', 'Omega'] = 0.0
        J['T', 'r'] = self.dT_dr
        J['T', 'precurve'] = self.dT_dprecurve
        J['T', 'presweep'] = self.dT_dpresweep
        J['T', 'presweepTip'] = self.dT_dpresweeptip
        J['T', 'precurveTip'] = self.dT_dprecurvetip
        J['T', 'precone'] = self.dT_dprecone
        J['T', 'rho'] = 0.0
        J['T', 'Rhub'] = self.dT_dRhub
        J['T', 'rotorR'] = 0

        J['Q', 'Uinf'] = 0.0
        J['Q', 'Rtip'] = self.dQ_dRtip
        J['Q', 'Omega'] = 0.0
        J['Q', 'r'] = self.dQ_dr
        J['Q', 'precurve'] = self.dQ_dprecurve
        J['Q', 'presweep'] = self.dQ_dpresweep
        J['Q', 'presweepTip'] = self.dQ_dpresweeptip
        J['Q', 'precurveTip'] = self.dQ_dprecurvetip
        J['Q', 'precone'] = self.dQ_dprecone
        J['Q', 'rho'] = 0.0
        J['Q', 'Rhub'] = self.dQ_dRhub
        J['Q', 'rotorR'] = 0

        return J


class BrentGroup(Group):
    def __init__(self, n, i):
        super(BrentGroup, self).__init__()

        self.add('flow', FlowCondition(), promotes=['*'])
        self.add('airfoils', AirfoilComp(n, i), promotes=['*'])
        sub = self.add('sub', Group(), promotes=['*'])
        sub.add('bem', BEM(n, i), promotes=['*'])

        sub.ln_solver = ScipyGMRES()
        self.ln_solver = ScipyGMRES()
        self.nl_solver = Brent()
        epsilon = 1e-6
        phi_lower = epsilon
        phi_upper = pi/2
        self.nl_solver.options['lower_bound'] = phi_lower
        self.nl_solver.options['upper_bound'] = phi_upper
        self.nl_solver.options['state_var'] = 'phi_sub'

class LoadsGroup(Group):
    def __init__(self, n):
        super(LoadsGroup, self).__init__()
        self.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        self.add('Uinf', IndepVarComp('Uinf', 0.0), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', 0.0), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0), promotes=['*'])
        self.add('azimuth', IndepVarComp('azimuth', 0.0), promotes=['*'])
        self.add('tsr', IndepVarComp('tsr', 0.0), promotes=['*'])
        self.add('r', IndepVarComp('r', val=np.zeros(n)), promotes=['*'])
        self.add('chord', IndepVarComp('chord', val=np.zeros(n)), promotes=['*'])
        self.add('theta', IndepVarComp('theta', val=np.zeros(n)), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', val=np.zeros(n)), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', val=np.zeros(n)), promotes=['*'])
        self.add('af', IndepVarComp('af', val=np.zeros(n), pass_by_obj=True), promotes=['*'])
        self.add('bemoptions', IndepVarComp('bemoptions', {}, pass_by_obj=True), promotes=['*'])
        self.add('init', CCInit(), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('mux', MUX(n), promotes=['*'])
        for i in range(n):
            self.add('brent'+str(i+1), BrentGroup(n, i), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'af', 'bemoptions'])
            self.connect('r', 'brent'+str(i+1)+'.r', src_indices=[i])
            self.connect('chord', 'brent'+str(i+1)+'.chord', src_indices=[i])
            self.connect('theta', 'brent'+str(i+1)+'.theta', src_indices=[i])
            self.connect('Vx', 'brent'+str(i+1)+'.Vx', src_indices=[i])
            self.connect('Vy', 'brent'+str(i+1)+'.Vy', src_indices=[i])
            self.connect('brent'+str(i+1)+'.phi_sub', 'phi'+str(i+1))
            self.connect('brent'+str(i+1)+'.cl_sub', 'cl'+str(i+1))
            self.connect('brent'+str(i+1)+'.cd_sub', 'cd'+str(i+1))
            self.connect('brent'+str(i+1)+'.W_sub', 'W'+str(i+1))
        self.add('loads', DistributedAeroLoads(n), promotes=['*'])

class Sweep(Group):
    def __init__(self, azimuth, n):
        super(Sweep, self).__init__()

        self.add('azimuth', IndepVarComp('azimuth', azimuth), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('mux', MUX(n), promotes=['*'])
        for i in range(n):
            self.add('brent'+str(i+1), BrentGroup(n, i), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'bemoptions', 'af'])
            self.connect('Vx', 'brent'+str(i+1)+'.Vx', src_indices=[i])
            self.connect('Vy', 'brent'+str(i+1)+'.Vy', src_indices=[i])
            self.connect('brent'+str(i+1)+'.phi_sub', 'phi'+str(i+1))
            self.connect('brent'+str(i+1)+'.cl_sub', 'cl'+str(i+1))
            self.connect('brent'+str(i+1)+'.cd_sub', 'cd'+str(i+1))
            self.connect('brent'+str(i+1)+'.W_sub', 'W'+str(i+1))
        self.add('loads', DistributedAeroLoads(n), promotes=['chord', 'rho', 'phi', 'cl', 'cd', 'W'])

class SweepGroup(Group):
    def __init__(self, nSector):
        super(SweepGroup, self).__init__()
        n = len(af)
        self.add('r', IndepVarComp('r', np.zeros(n)), promotes=['*'])
        self.add('chord', IndepVarComp('chord', np.zeros(n)), promotes=['*'])
        self.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        self.add('theta', IndepVarComp('theta', np.zeros(n)), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        self.add('Uinf', IndepVarComp('Uinf', 0.0), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', 0.0), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', np.zeros(n)), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', np.zeros(n)), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0), promotes=['*'])
        self.add('tsr', IndepVarComp('tsr', 0.0), promotes=['*'])
        self.add('af', IndepVarComp('af', np.zeros(n), pass_by_obj=True), promotes=['*'])
        self.add('bemoptions', IndepVarComp('bemoptions', {}, pass_by_obj=True), promotes=['*'])
        self.add('init', CCInit(), promotes=['*'])

        for i in range(nSector):
            azimuth = pi/180.0*360.0*float(i)/nSector
            self.add('group'+str(i+1), Sweep(azimuth, n), promotes=['r', 'Uinf', 'pitch', 'Rtip', 'Omega', 'chord', 'rho', 'mu', 'Rhub', 'hubHt', 'precurve', 'presweep', 'precone', 'tilt', 'yaw', 'pitch', 'shearExp', 'B', 'bemoptions', 'af'])
            # self.connect('af', 'group'+str(i+1)+'.af')
            for j in range(n):
                self.connect('theta', 'group'+str(i+1)+'.brent'+str(j+1)+'.theta', src_indices=[j])
                self.connect('chord', 'group'+str(i+1)+'.brent'+str(j+1)+'.chord', src_indices=[j])
                self.connect('r', 'group'+str(i+1)+'.brent'+str(j+1)+'.r', src_indices=[j])

class CCBlade(Group):

    def __init__(self, nSector, n):
        super(CCBlade, self).__init__()

        self.add('load_group', SweepGroup(nSector), promotes=['Uinf', 'tsr', 'pitch', 'Rtip', 'Omega', 'r', 'chord', 'theta', 'rho', 'mu', 'Rhub', 'rotorR', 'precurve', 'presweep', 'precurveTip', 'presweepTip', 'precone', 'tilt', 'yaw', 'pitch', 'shearExp', 'hubHt', 'B', 'af', 'bemoptions'])
        self.add('eval', CCEvaluate(n), promotes=['Uinf', 'Rtip', 'Omega', 'r', 'Rhub', 'B', 'precurve', 'presweep', 'presweepTip', 'precurveTip', 'precone', 'nSector', 'rotorR', 'rho', 'CP', 'CT', 'CQ', 'P', 'T', 'Q'])

        for i in range(nSector):
            self.connect('load_group.group' + str(i+1) + '.loads.Np', 'eval.Np' + str(i+1))
            self.connect('load_group.group' + str(i+1) + '.loads.Tp', 'eval.Tp' + str(i+1))

        self.add('obj_cmp', ExecComp('obj = -CP', CP=1.0), promotes=['*'])

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
    n = 17

    # Testing
    # test_bracket(n, af, bemoptions)

    ## Test LoadsGroup
    loads = Problem()
    root = loads.root = LoadsGroup(n)
    loads.setup()

    loads['Rhub'] = Rhub
    loads['Rtip'] = Rtip
    loads['r'] = r
    loads['chord'] = chord
    loads['theta'] = np.radians(theta)
    loads['rho'] = rho
    loads['mu'] = mu
    loads['tilt'] = np.radians(tilt)
    loads['precone'] = np.radians(precone)
    loads['yaw'] = np.radians(yaw)
    loads['shearExp'] = shearExp
    loads['hubHt'] = hubHt
    loads['Uinf'] = Uinf
    loads['tsr'] = Omega * loads['Rtip'] * pi / (30.0 * Uinf)
    loads['pitch'] = np.radians(pitch)
    loads['azimuth'] = np.radians(azimuth)
    loads['af'] = af
    loads['bemoptions'] = bemoptions

    loads.run()

    print 'phi', loads['phi']
    print 'Np', loads['Np']
    print 'Tp', loads['Tp']
    test_grad = open('partial_test_grad2.txt', 'w')
    power_gradients = loads.check_total_derivatives(out_stream=test_grad, unknown_list=['Np', 'Tp'])
    # power_partial = loads.check_partial_derivatives(out_stream=test_grad)
    # print "gradients calculated"

    ## Test CCBlade
    ccblade = Problem()
    ccblade.root = CCBlade(nSector, n)

    ### SETUP OPTIMIZATION
    # ccblade.driver = pyOptSparseDriver()
    # ccblade.driver.options['optimizer'] = 'SNOPT' #'SLSQP'
    # ccblade.driver.add_desvar('tsr', lower=1.5, upper=14.0)
    # ccblade.driver.add_objective('obj')
    # recorder = SqliteRecorder('recorder')
    # recorder.options['record_params'] = True
    # recorder.options['record_metadata'] = True
    # ccblade.driver.add_recorder(recorder)

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
    ccblade['af'] = af
    ccblade['bemoptions'] = bemoptions

    ccblade.run()

    ccblade.root.load_group.group1.brent1.list_connections()

    # test_grad = open('partial_test_grad.txt', 'w')
    # power_gradients = ccblade.check_total_derivatives_modified2(out_stream=test_grad)
    # power_partial = ccblade.check_partial_derivatives(out_stream=test_grad)

    print 'CP', ccblade['CP']
    print 'CT', ccblade['CT']
    print 'CQ', ccblade['CQ']
