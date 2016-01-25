__author__ = 'ryanbarr'

import warnings
from math import cos, sin, pi, sqrt, acos, exp, factorial
import numpy as np
import _bem
from openmdao.api import Component, ExecComp, IndepVarComp, Group, Problem, ScipyGMRES, NLGaussSeidel, ParallelGroup, DumpRecorder
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from openmdao.core.mpi_wrap import MPI
# from zope.interface import Interface, implements
from scipy.interpolate import RectBivariateSpline, bisplev
from airfoilprep_free import Airfoil, CCAirfoil
from brent import Brent

class CCInit(Component):
    """
    CCInit

    Calculates rotorR

    """
    def __init__(self):
        super(CCInit, self).__init__()
        self.add_param('Rtip', val=0.0, units='m')
        self.add_param('precone', val=0.0, units='deg')
        self.add_param('precurveTip', val=0.0, units='m')

        self.add_output('rotorR', shape=1, units='m')

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['rotorR'] = params['Rtip']*cos(params['precone']) + params['precurveTip']*sin(params['precone'])

    def linearize(self, params, unknowns, resids):
        J = {}
        J['rotorR', 'precone'] = -params['Rtip']*sin(params['precone']) + params['precurveTip']*cos(params['precone'])
        J['rotorR', 'precurveTip'] = sin(params['precone'])
        J['rotorR', 'Rtip'] = cos(params['precone'])
        return J

class WindComponents(Component):
    """
    WindComponents

    Inputs: n - Number of sections analyzed across blade

    Outputs: Vx, Vy

    """
    def __init__(self, n):
        super(WindComponents, self).__init__()
        self.add_param('r', val=np.zeros(n), units='m')
        self.add_param('precurve', val=np.zeros(n), units='m')
        self.add_param('presweep', shape=n, units='m')
        self.add_param('Uinf', shape=1, units='m/s')
        self.add_param('precone', shape=1, units='deg')
        self.add_param('azimuth', val=0.0, units='deg')
        self.add_param('tilt', shape=1, units='deg')
        self.add_param('yaw', shape=1, units='deg')
        self.add_param('Omega', shape=1, units='rpm')
        self.add_param('shearExp', shape=1) #, pass_by_obj=True)
        self.add_param('hubHt', shape=1, units='m')

        self.add_output('Vx', shape=n, units='m/s')
        self.add_output('Vy', shape=n, units='m/s')

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
        self.add_param('theta', val=np.zeros(n), units='deg')

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['Vx'], unknowns['Vy'] = _bem.windcomponents(params['r'], params['precurve'], params['presweep'], params['precone'], params['yaw'], params['tilt'], params['azimuth'], params['Uinf'], params['Omega'], params['hubHt'], params['shearExp'])
    def list_deriv_vars(self):

        inputs = ('r', 'precurve', 'presweep', 'Uinf', 'precone', 'azimuth', 'tilt', 'yaw', 'Omega', 'hubHt')
        outputs = ('Vx', 'Vy')

        return inputs, outputs

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
    """
    FlowCondition

    Outputs: alpha, Re, W

    """
    def __init__(self):
        super(FlowCondition, self).__init__()
        self.add_param('pitch', shape=1, units='deg')
        self.add_param('Vx', shape=1, units='m/s')
        self.add_param('Vy', shape=1, units='m/s')
        self.add_param('chord', shape=1, units='m')
        self.add_param('theta', shape=1, units='deg')
        self.add_param('rho', shape=1, units='kg/m**3')
        self.add_param('mu', shape=1, units='kg/(m*s)')
        self.add_param('a_sub', val=0.0841641)
        self.add_param('ap_sub', val=-0.0841641)
        self.add_param('phi_sub', shape=1, units='deg')
        self.add_param('da_dx', val=np.zeros(9))
        self.add_param('dap_dx', val=np.zeros(9))

        self.add_output('alpha_sub', shape=1, units='deg')
        self.add_output('Re_sub', shape=1)
        self.add_output('W_sub', shape=1, units='m/s')
        self.add_output('dalpha_dx', shape=9)
        self.add_output('dRe_dx', shape=9)

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.i = i
        # self.fd_options['force_fd'] = True

    def solve_nonlinear(self, params, unknowns, resids):

        alpha, W, Re = _bem.relativewind(params['phi_sub'], params['a_sub'], params['ap_sub'], params['Vx'], params['Vy'], params['pitch'], params['chord'], params['theta'], params['rho'], params['mu'])
        unknowns['dalpha_dx'] = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        unknowns['dRe_dx'] = np.array([0.0, Re/params['chord'], 0.0, Re*params['Vx']/W**2, Re*params['Vy']/W**2, 0.0, 0.0, 0.0, 0.0])
        unknowns['alpha_sub'] = alpha
        unknowns['W_sub'] = W
        unknowns['Re_sub'] = Re

    def list_deriv_vars(self):

        inputs = ('Vx', 'Vy', 'theta', 'pitch', 'rho', 'mu', 'phi_sub', 'a_sub', 'ap_sub', 'chord')
        outputs = ('alpha_sub', 'W_sub', 'Re_sub')

        return inputs, outputs

    def linearize(self, params, unknowns, resids):

        J = {}

        # x = [phi, chord, theta, Vx, Vy, r, Rhub, Rtip, pitch]  (derivative order)
        dx_dx = np.eye(9)

        alpha, dalpha_dx, W, dW_dx, Re, dRe_dx = _bem.relativewind_dv(params['phi_sub'], dx_dx[0, :],
            params['a_sub'], params['da_dx'], params['ap_sub'], params['dap_dx'], params['Vx'], dx_dx[3, :], params['Vy'], dx_dx[4, :],
            params['pitch'], dx_dx[8, :], params['chord'], dx_dx[1, :], params['theta'], dx_dx[2, :],
            params['rho'], params['mu'])

        Vx = params['Vx']
        Vy = params['Vy']
        if abs(params['a_sub']) > 10:
            dW_da = 0.0
            dW_dap = Vy / cos(params['phi_sub'])
            dW_dVx = 0.0
            dW_dVy = (1+params['ap_sub'])/cos(params['phi_sub'])
        elif abs(params['ap_sub']) > 10:
            dW_da = -params['Vx'] / sin(params['phi_sub'])
            dW_dap = 0.0
            dW_dVx = (1-params['a_sub'])/sin(params['phi_sub'])
            dW_dVy = 0.0
        else:
            dW_da = Vx**2*(params['a_sub'] - 1) / (sqrt((Vx*(1 - params['a_sub']))**2 + (Vy*(1+params['ap_sub']))**2))
            dW_dap = Vy**2*(params['ap_sub'] + 1) / (sqrt((Vx*(1 - params['a_sub']))**2 + (Vy*(1+params['ap_sub']))**2))
            dW_dVx = Vx*((1-params['a_sub'])**2) / (sqrt((Vx*(1 - params['a_sub']))**2 + (Vy*(1+params['ap_sub']))**2))
            dW_dVy = Vy*((1 + params['ap_sub'])**2) / (sqrt((Vx*(1 - params['a_sub']))**2 + (Vy*(1+params['ap_sub']))**2))

        J['alpha_sub', 'phi_sub'] = 1.0
        J['alpha_sub', 'theta'] = -1.0
        J['alpha_sub', 'pitch'] = -1.0

        J['Re_sub', 'chord'] = Re/params['chord']
        J['Re_sub', 'Vx'] = params['rho'] * dW_dVx * params['chord'] / params['mu']
        J['Re_sub', 'Vy'] = params['rho'] * dW_dVy * params['chord'] / params['mu']
        J['Re_sub', 'a_sub'] = params['rho'] * dW_da * params['chord'] / params['mu']
        J['Re_sub', 'ap_sub'] = params['rho'] * dW_dap * params['chord'] / params['mu']
        J['Re_sub', 'rho'] = Re / params['rho']
        J['Re_sub', 'mu'] = -Re / params['mu'] #**2

        J['W_sub', 'Vx'] = dW_dVx
        J['W_sub', 'Vy'] = dW_dVy
        J['W_sub', 'a_sub'] = dW_da
        J['W_sub', 'ap_sub'] = dW_dap

        # J['W_sub', 'phi_sub'] = dW_dx[0]
        # J['W_sub', 'chord'] = dW_dx[1]
        # J['W_sub', 'theta'] = dW_dx[2]
        # J['W_sub', 'Vx'] = dW_dx[3]
        # J['W_sub', 'Vy'] = dW_dx[4]
        # J['W_sub', 'r'] = dW_dx[5]
        # J['W_sub', 'Rhub'] = dW_dx[6]
        # J['W_sub', 'Rtip'] = dW_dx[7]
        # J['W_sub', 'pitch'] = dW_dx[8]
        # J['W_sub', 'a_sub'] = dW_da
        # J['W_sub', 'ap_sub'] = dW_dap

        return J

class deMUX(Component):
    def __init__(self, n, i):
        super(deMUX, self).__init__()
        self.add_param('airfoil_parameterization', val=np.zeros((n, 8)))
        for i in range(n):
            self.add_output('airfoil'+str(i), val=np.zeros(8))
        self.n = n
    def solve_nonlinear(self, params, unknowns, resids):
        for i in range(self.n):
            unknowns['airfoil'+str(i)] = params['airfoil_parameterization'][i]

    def linearize(self, params, unknowns, resids):
        J ={}
        for i in range(self.n):
            J['airfoil'+str(i), 'airfoil_parameterization'] = n
        return J

class AirfoilComp(Component):
    """
    AirfoilComp

    Inputs: n - Number of sections analyzed across blade, i - Section number

    Outputs: cl, cd

    """
    def __init__(self, n, i):
        super(AirfoilComp, self).__init__()
        self.add_param('alpha_sub', shape=1, units='deg')
        self.add_param('Re_sub', shape=1)
        self.add_param('airfoil_parameterization', val=np.zeros((n, 8))) #, pass_by_obj=True)
        self.add_param('airfoil_analysis_options', val={}, pass_by_obj=True)
        self.add_param('af', val=np.zeros(n), pass_by_obj=True)

        self.add_output('cl_sub', shape=1)
        self.add_output('cd_sub', shape=1)
        self.add_output('dcl_dalpha', shape=1)
        self.add_output('dcl_dRe', shape=1)
        self.add_output('dcd_dalpha', shape=1)
        self.add_output('dcd_dRe', shape=1)

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
        self.i = i
        self.n = n

    def airfoil_parameterization_methodology(self, CST, alpha, Re, FDorCS):
        # self.af = CCAirfoil.initFromCST(CST) #, [alpha], [Re])
        # alpha, Re, cl, cd, cm = self.af.createDataGrid()

        cl, cd = self.af.evaluate(alpha, Re)
        dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe = self.af.derivatives(alpha, Re) #xfoilFlowGradients(CST, alpha, Re, FDorCS)
        return cl, cd, dcl_dalpha, dcl_dRe, dcd_dalpha, dcd_dRe

    def airfoil_parameterization_methodology_dv(self, CST, alpha, Re, CFDorXFOIL, iterations, processors, FDorCS):
        # if CFDorXFOIL == 'CFD':
        #     cl, cd, dcl_dcst, dcd_dcst = Airfoil.cfdGradients(CST, alpha, Re, iterations, processors, FDorCS, Uinf=10.0)
        # else:
        cl, cd, dcl_dcst, dcd_dcst = Airfoil.xfoilGradients(CST, alpha, Re, FDorCS)
        return cl, cd, dcl_dcst, dcd_dcst

    def solve_nonlinear(self, params, unknowns, resids):
        FDorCS = params['airfoil_analysis_options']['FDorCS']
        if self.i == 0:
            unknowns['cl_sub'], unknowns['cd_sub'], unknowns['dcl_dalpha'], unknowns['dcl_dRe'], unknowns['dcd_dalpha'], unknowns['dcd_dRe'] = 0.0, 0.50, 0.0, 0.0, 0.0, 0.0
        elif self.i == 1 or self.i == 2:
            unknowns['cl_sub'], unknowns['cd_sub'], unknowns['dcl_dalpha'], unknowns['dcl_dRe'], unknowns['dcd_dalpha'], unknowns['dcd_dRe'] = 0.0, 0.35, 0.0, 0.0, 0.0, 0.0
        else:
            self.af = params['af'][self.i]
            unknowns['cl_sub'], unknowns['cd_sub'], unknowns['dcl_dalpha'], unknowns['dcl_dRe'], unknowns['dcd_dalpha'], unknowns['dcd_dRe'] = self.airfoil_parameterization_methodology(params['airfoil_parameterization'][self.i], params['alpha_sub'], params['Re_sub'], FDorCS)

    def list_deriv_vars(self):
        inputs = ('alpha_sub', 'Re_sub')
        outputs = ('cl_sub', 'cd_sub')
        return inputs, outputs

    def linearize(self, params, unknowns, resids):
        J = {}
        airfoil_analysis = params['airfoil_analysis_options']
        if self.i == 0:
            cl, cd, dcl_dcst, dcd_dcst = 0.0, 0.50, np.zeros((8, 1)), np.zeros((8, 1))
        elif self.i == 1 or self.i == 2:
            cl, cd, dcl_dcst, dcd_dcst = 0.0, 0.35, np.zeros((8, 1)), np.zeros((8, 1))
        else:
            cl, cd, dcl_dcst, dcd_dcst = self.airfoil_parameterization_methodology_dv(params['airfoil_parameterization'][self.i], params['alpha_sub'], params['Re_sub'], airfoil_analysis['CFDorXFOIL'], airfoil_analysis['iterations'], airfoil_analysis['processors'], airfoil_analysis['FDorCS'])
        # unknowns['cl_sub'], unknowns['cd_sub'] = cl, cd
        zero_cl = np.zeros((1, self.n * 8))
        zero_cd = np.zeros((1, self.n * 8))
        for i in range(8):
            zero_cl[0][i+self.i*8] = dcl_dcst[i]
            zero_cd[0][i+self.i*8] = dcd_dcst[i]

        J['cl_sub', 'alpha_sub'] = unknowns['dcl_dalpha']
        J['cl_sub', 'Re_sub'] = unknowns['dcl_dRe']
        J['cl_sub', 'airfoil_parameterization'] = zero_cl
        J['cd_sub', 'alpha_sub'] = unknowns['dcd_dalpha']
        J['cd_sub', 'Re_sub'] = unknowns['dcd_dRe']
        J['cd_sub', 'airfoil_parameterization'] = zero_cd
        return J

class BEM(Component):
    """
    BEM

    Inputs: n - Number of sections analyzed across blade, i - Section number

    Outputs: phi, a, ap

    """
    def __init__(self, n, i):
        super(BEM, self).__init__()

        self.add_param('pitch', shape=1, units='deg')
        self.add_param('Rtip', shape=1, units='m')
        self.add_param('Vx', shape=1, units='m/s')
        self.add_param('Vy', shape=1, units='m/s')
        self.add_param('Omega', shape=1, units='rpm')
        self.add_param('r', shape=1, units='m')
        self.add_param('chord', shape=1, units='m')
        self.add_param('theta', shape=1, units='deg')
        self.add_param('rho', shape=1, units='kg/m**3')
        self.add_param('mu', shape=1, units='kg/(m*s)')
        self.add_param('Rhub', shape=1, units='m')
        self.add_param('alpha_sub', shape=1, units='deg')
        self.add_param('cl_sub', shape=1)
        self.add_param('cd_sub', shape=1)
        self.add_param('B', val=3, pass_by_obj=True)
        self.add_param('bemoptions', val={}, pass_by_obj=True)
        self.add_param('dcl_dalpha', shape=1)
        self.add_param('dcd_dalpha', shape=1)
        self.add_param('dalpha_dx', shape=9)
        self.add_param('dcl_dRe', shape=1)
        self.add_param('dcd_dRe', shape=1)
        self.add_param('dRe_dx', shape=9)

        self.add_output('a_sub', shape=1)
        self.add_output('ap_sub', shape=1)
        self.add_output('da_dx', val=np.zeros(9))
        self.add_output('dap_dx', val=np.zeros(9))
        self.add_state('phi_sub', shape=1, units='deg')

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.i = i
        # self.fd_options['force_fd'] = True

    def solve_nonlinear(self, params, unknowns, resids):
        r = params['r']
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        Vx = params['Vx']
        Vy = params['Vy']
        bemoptions = params['bemoptions']
        chord = params['chord']

        dx_dx = np.eye(9)
        # chain rule
        dcl_dx = params['dcl_dalpha']*params['dalpha_dx'] + params['dcl_dRe']*params['dRe_dx']
        dcd_dx = params['dcd_dalpha']*params['dalpha_dx'] + params['dcd_dRe']*params['dRe_dx']

        if not (params['Omega'] != 0):
            dR_dx = np.zeros(9)
            dR_dx[0] = 1.0  # just to prevent divide by zero
            da_dx = np.zeros(9)
            dap_dx = np.zeros(9)

        else:
            # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------
            fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(r, chord, Rhub, Rtip,
             unknowns['phi_sub'], params['cl_sub'], params['cd_sub'], params['B'], Vx, Vy, dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
             dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **bemoptions)
            # resids['phi_sub'] = fzero
            unknowns['a_sub'] = a
            unknowns['ap_sub'] = ap
            unknowns['da_dx'] = da_dx
            unknowns['dap_dx'] = dap_dx

            # self.fzero = fzero
            # self.a = a
            # self.ap = ap

        self.dR_dx = dR_dx
        self.da_dx = da_dx
        self.dap_dx = dap_dx

    def apply_nonlinear(self, params, unknowns, resids):
        r = params['r']
        Rhub = params['Rhub']
        Rtip = params['Rtip']
        Vx = params['Vx']
        Vy = params['Vy']
        bemoptions = params['bemoptions']
        chord = params['chord']

        dx_dx = np.eye(9)
        # chain rule
        dcl_dx = params['dcl_dalpha']*params['dalpha_dx'] + params['dcl_dRe']*params['dRe_dx']
        dcd_dx = params['dcd_dalpha']*params['dalpha_dx'] + params['dcd_dRe']*params['dRe_dx']

        if not (params['Omega'] != 0):
            dR_dx = np.zeros(9)
            dR_dx[0] = 1.0  # just to prevent divide by zero
            da_dx = np.zeros(9)
            dap_dx = np.zeros(9)

        else:
            # ------ BEM solution method see (Ning, doi:10.1002/we.1636) ------
            fzero, a, ap, dR_dx, da_dx, dap_dx = _bem.inductionfactors_dv(r, chord, Rhub, Rtip,
             unknowns['phi_sub'], params['cl_sub'], params['cd_sub'], params['B'], Vx, Vy, dx_dx[5, :], dx_dx[1, :], dx_dx[6, :], dx_dx[7, :],
             dx_dx[0, :], dcl_dx, dcd_dx, dx_dx[3, :], dx_dx[4, :], **bemoptions)
            resids['phi_sub'] = fzero
            resids['a_sub'] = a - unknowns['a_sub']
            resids['ap_sub'] = ap - unknowns['ap_sub']
            # if self.i == 11:
            #     pass
            # self.fzero = fzero
            # self.a = a
            # self.ap = ap

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
        # J['phi_sub', 'theta'] = dR_dx[2]
        J['phi_sub', 'Vx'] = dR_dx[3]
        J['phi_sub', 'Vy'] = dR_dx[4]
        J['phi_sub', 'r'] = dR_dx[5]
        J['phi_sub', 'Rhub'] = dR_dx[6]
        J['phi_sub', 'Rtip'] = dR_dx[7]
        # J['phi_sub', 'pitch'] = dR_dx[8]

        Vx = params['Vx']
        Vy = params['Vy']
        sigma_p = params['B']/2.0/pi*params['chord']/params['r']
        # cl = params['cl_sub']
        # cd = params['cd_sub']
        cphi = cos(unknowns['phi_sub'])
        sphi = sin(unknowns['phi_sub'])
        B = params['B']
        Rtip = params['Rtip']
        r = params['r']
        Rhub = params['Rhub']
        factortip = B/2.0*(Rtip - r)/(r*abs(sphi))
        Ftip = 2.0/pi*acos(exp(-factortip))

        factorhub = B/2.0*(r - Rhub)/(Rhub*abs(sphi))
        Fhub = 2.0/pi*acos(exp(-factorhub))

        F = Ftip * Fhub

        # dR_dcl = 1/(Vy/Vx)*sigma_p/4.0/F
        # dR_dcd = -1/(Vy/Vx)*sigma_p*cphi/4.0/F/sphi

        dR_dcl = cphi/(Vy/Vx)*(sigma_p*(sphi)/4.0/F/sphi/cphi) # 1/(Vy/Vx)*sigma_p/4.0/F
        dR_dcd = cphi/(Vy/Vx)*(sigma_p*(-cphi)/4.0/F/sphi/cphi) # -1/(Vy/Vx)*sigma_p*cphi/4.0/F/sphi

        # J['phi_sub', 'cl_sub'] = dR_dcl
        # J['phi_sub', 'cd_sub'] = dR_dcd
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


        J['phi_sub', 'cl_sub'] = 0.0
        return J

class MUX(Component):
    """
    MUX - Combines all the sections into single variables of arrays

    Inputs: n - Number of sections analyzed across blade

    Outputs: phi, cl, cd, W

    """
    def __init__(self, n):
        super(MUX, self).__init__()
        for i in range(n):
            self.add_param('phi'+str(i+1), val=0.0, units='deg')
            self.add_param('cl'+str(i+1), val=0.0)
            self.add_param('cd'+str(i+1), val=0.0)
            self.add_param('W'+str(i+1), val=0.0, units='m/s')
            self.add_param('alpha'+str(i+1), val=0.0, units='deg')
        self.add_param('airfoil_parameterization', val=np.zeros((n, 8))) #, pass_by_obj=True)
        self.add_param('airfoil_analysis_options', val={}, pass_by_obj=True)

        self.add_output('phi', val=np.zeros(n), units='deg')
        self.add_output('cl', val=np.zeros(n))
        self.add_output('cd', val=np.zeros(n))
        self.add_output('W', val=np.zeros(n), units='m/s')

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):
        for i in range(self.n):
            CST = params['airfoil_parameterization'][i]
            alpha = params['alpha'+str(i+1)]
            Re = 1e7
            CFDorXFOIL = params['airfoil_analysis_options']['CFDorXFOIL']
            CFDorXFOIL = 'XFOIL'
            # if CFDorXFOIL == 'CFD':
            #     cl, cd,  = Airfoil.cfdGradients(CST, alpha, Re, params['airfoil_analysis_options']['iterations'], params['airfoil_analysis_options']['processors'], params['airfoil_analysis_options']['FDorCS'], Uinf=10.0, ComputeGradients=False)
            #     unknowns['cl'][i] = cl
            #     unknowns['cd'][i] = cd
            # else:
            #     # cl, cd, self.dcl_dcst, self.dcd_dcst = Airfoil.xfoilGradients(CST, alpha, Re, params['airfoil_analysis_options']['FDorCS'])
            unknowns['cl'][i] = params['cl'+str(i+1)]
            unknowns['cd'][i] = params['cd'+str(i+1)]
            unknowns['phi'][i] = params['phi'+str(i+1)]
            unknowns['W'][i] = params['W'+str(i+1)]

    def linearize(self, params, unknowns, resids):
        n = self.n
        J = {}
        Re = 1e7
        dcl_dcst_total = np.zeros((1, self.n * 8))
        dcd_dcst_total = np.zeros((1, self.n * 8))

        for i in range(n):
            # CST = params['airfoil_parameterization'][i]
            # alpha = params['alpha'+str(i+1)]
            # if params['airfoil_analysis_options']['CFDorXFOIL'] == 'CFD':
            #     if i < 5:
            #         cl, cd, dcl_dcst, dcd_dcst = 0.0, 0.0, np.zeros(8), np.zeros(8)
            #     else:
            #         cl, cd, dcl_dcst, dcd_dcst = Airfoil.cfdGradients(CST, alpha, Re, params['airfoil_analysis_options']['iterations'], params['airfoil_analysis_options']['processors'], params['airfoil_analysis_options']['FDorCS'], Uinf=10.0, ComputeGradients=True)
            # else:
            #     cl, cd, dcl_dcst, dcd_dcst = Airfoil.xfoilGradients(CST, alpha, Re, params['airfoil_analysis_options']['FDorCS'])
            # if i == 5:
            #     pass
            # for j in range(len(dcd_dcst)):
            #     dcl_dcst_total[0][j+i*8] = dcl_dcst[j]
            #     dcd_dcst_total[0][j+i*8] = dcd_dcst[j]
            zeros = np.zeros(n)
            zeros[i] = 1
            J['phi', 'phi'+str(i+1)] = zeros
            J['cl', 'cl'+str(i+1)] = zeros
            J['cd', 'cd'+str(i+1)] = zeros
            J['W', 'W'+str(i+1)] = zeros
            # J['cd', 'cd'+str(i+1)] = zeros
            # J['W', 'W'+str(i+1)] = zeros
        # J['cl', 'airfoil_parameterization'] = dcl_dcst_total
        # J['cd', 'airfoil_parameterization'] = dcd_dcst_total
        return J

    # def get_req_procs(self):
    #     """
    #     min/max number of procs that this component can use
    #     """
    #     processors = 2
    #     return (1, processors)

class MUX_POWER(Component):
    """
    MUX_POWER - Combines all the sections into single variables of arrays

    Inputs: n2 - Number of Uinfs

    Outputs: CT, CQ, CP, T, Q, P

    """
    def __init__(self, n2):
        super(MUX_POWER, self).__init__()
        for i in range(n2):
            self.add_param('CT'+str(i+1), val=0.0)
            self.add_param('CQ'+str(i+1), val=0.0)
            self.add_param('CP'+str(i+1), val=0.0)
            self.add_param('T'+str(i+1), val=0.0)
            self.add_param('Q'+str(i+1), val=0.0)
            self.add_param('P'+str(i+1), val=0.0)

        self.add_output('CT', val=np.zeros(n2))
        self.add_output('CQ', val=np.zeros(n2))
        self.add_output('CP', val=np.zeros(n2))
        self.add_output('T', val=np.zeros(n2), units='N')
        self.add_output('Q', val=np.zeros(n2), units='N*m')
        self.add_output('P', val=np.zeros(n2), units='W')

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        self.n2 = n2

        self.add_param('theta', val=np.zeros(17), units='deg')
        self.add_param('Uinf', val=np.zeros(n2), units='m/s')
        self.add_param('pitch', val=np.zeros(n2), units='deg')
        self.add_param('Omega', val=np.zeros(n2), units='rpm')

    def solve_nonlinear(self, params, unknowns, resids):
        for i in range(self.n2):
            unknowns['CT'][i] = params['CT'+str(i+1)]
            unknowns['CQ'][i] = params['CQ'+str(i+1)]
            unknowns['CP'][i] = params['CP'+str(i+1)]
            unknowns['T'][i] = params['T'+str(i+1)]
            unknowns['Q'][i] = params['Q'+str(i+1)]
            unknowns['P'][i] = params['P'+str(i+1)]

    def linearize(self, params, unknowns, resids):
        n2 = self.n2
        J = {}
        for i in range(n2):
            zeros = np.zeros(n2)
            zeros[i] = 1
            J['CT', 'CT'+str(i+1)] = zeros
            J['CQ', 'CQ'+str(i+1)] = zeros
            J['CP', 'CP'+str(i+1)] = zeros
            J['T', 'T'+str(i+1)] = zeros
            J['Q', 'Q'+str(i+1)] = zeros
            J['P', 'P'+str(i+1)] = zeros
        return J

class DistributedAeroLoads(Component):
    """
    DistributedAeroLoads

    Inputs: n - Number of sections analyzed across blade

    Outputs: Np, Tp

    """
    def __init__(self, n):
        super(DistributedAeroLoads, self).__init__()
        self.add_param('chord', shape=n, units='m')
        self.add_param('rho', shape=1, units='kg/m**3')
        self.add_param('phi', val=np.zeros(n), units='deg')
        self.add_param('cl', val=np.zeros(n))
        self.add_param('cd', val=np.zeros(n))
        self.add_param('W', val=np.zeros(n), units='m/s')

        self.add_output('Np', shape=n, units='N/m')
        self.add_output('Tp', shape=n, units='N/m')

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
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

        self.dNp_dcl, self.dTp_dcl, self.dNp_dcd, self.dTp_dcd, self.dNp_dphi, self.dTp_dphi, self.dNp_drho, self.dTp_drho, self.dNp_dW, self.dTp_dW, self.dNp_dchord, self.dTp_dchord = \
            np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

        for i in range(n):
            cphi = cos(phi[i])
            sphi = sin(phi[i])

            cn = cl[i]*cphi + cd[i]*sphi  # these expressions should always contain drag
            ct = cl[i]*sphi - cd[i]*cphi
            q = 0.5*rho*W[i]**2
            Np[i] = cn*q*chord[i]
            Tp[i] = ct*q*chord[i]

            self.dNp_dcl[i] = cphi*q*chord[i]
            self.dTp_dcl[i] = sphi*q*chord[i]
            self.dNp_dcd[i] = sphi*q*chord[i]
            self.dTp_dcd[i] = -cphi*q*chord[i]
            self.dNp_dphi[i] = (-cl[i]*sphi + cd[i]*cphi)*q*chord[i]
            self.dTp_dphi[i] = (cl[i]*cphi + cd[i]*sphi)*q*chord[i]
            self.dNp_drho[i] = cn*q/rho*chord[i]
            self.dTp_drho[i] = ct*q/rho*chord[i]
            self.dNp_dW[i] = cn*0.5*rho*2*W[i]*chord[i]
            self.dTp_dW[i] = ct*0.5*rho*2*W[i]*chord[i]
            self.dNp_dchord[i] = cn*q
            self.dTp_dchord[i] = ct*q

        unknowns['Np'] = Np
        unknowns['Tp'] = Tp

    def linearize(self, params, unknowns, resids):

        J = {}
        # # add chain rule for conversion to radians
        ## TODO: Check radian conversion
        # ridx = [2, 6, 7, 9, 10, 13]
        # dNp_dz[ridx, :] *= pi/180.0
        # dTp_dz[ridx, :] *= pi/180.0
        J['Np', 'cl'] = np.diag(self.dNp_dcl)
        J['Tp', 'cl'] = np.diag(self.dTp_dcl)
        J['Np', 'cd'] = np.diag(self.dNp_dcd)
        J['Tp', 'cd'] = np.diag(self.dTp_dcd)
        J['Np', 'phi'] = np.diag(self.dNp_dphi)
        J['Tp', 'phi'] = np.diag(self.dTp_dphi)
        J['Np', 'rho'] = self.dNp_drho
        J['Tp', 'rho'] = self.dTp_drho
        J['Np', 'W'] = np.diag(self.dNp_dW) # rho*W*cn*chord
        J['Tp', 'W'] = np.diag(self.dTp_dW)
        J['Np', 'chord'] = np.diag(self.dNp_dchord)
        J['Tp', 'chord'] = np.diag(self.dTp_dchord)
        return J

class CCEvaluate(Component):
    """
    CCEvaluate

    Inputs: n - Number of sections analyzed across blade, nSector - Number of sweeps

    Outputs: CP, CT, CQ, P, T, Q

    """
    def __init__(self, n, nSector):
        super(CCEvaluate, self).__init__()

        self.add_param('Uinf', val=10.0, units='m/s')
        self.add_param('Rtip', val=63., units='m')
        self.add_param('Omega', shape=1, units='rpm')
        self.add_param('r', shape=n, units='m')
        self.add_param('B', val=3, pass_by_obj=True)
        self.add_param('precurve', shape=n, units='m')
        self.add_param('presweep', shape=n, units='m')
        self.add_param('presweepTip', shape=1, units='m')
        self.add_param('precurveTip', shape=1, units='m')
        self.add_param('rho', shape=1, units='kg/m**3')
        self.add_param('precone', shape=1, units='deg')
        self.add_param('Rhub', shape=1, units='m')
        self.add_param('nSector', val=4, pass_by_obj=True)
        self.add_param('rotorR', shape=1, units='m')
        for i in range(nSector):
            self.add_param('Np'+str(i+1), val=np.zeros(n), units='N/m')
            self.add_param('Tp'+str(i+1), val=np.zeros(n), units='N/m')

        self.add_output('P', val=0.5, units='W')
        self.add_output('T', val=0.5, units='N')
        self.add_output('Q', val=0.5, units='N*m')
        self.add_output('CP', val=0.5)
        self.add_output('CT', val=0.5)
        self.add_output('CQ', val=0.5)

        self.fd_options['form'] = 'central'
        self.fd_options['step_type'] = 'relative'
        # self.fd_options['force_fd'] = True
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
        n = self.n
        Np = {}
        Tp = {}
        for i in range(nSector):
            Np['Np' + str(i+1)] = params['Np' + str(i+1)]
            Tp['Tp' + str(i+1)] = params['Tp' + str(i+1)]
        T = np.zeros(npts)
        Q = np.zeros(npts)


        self.dT_dr, self.dQ_dr, self.dP_dr, self.dT_dprecurve, self.dQ_dprecurve, self.dP_dprecurve, self.dT_dpresweep, self.dQ_dpresweep, self.dP_dpresweep, \
        self.dT_dprecone, self.dQ_dprecone, self.dP_dprecone, self.dT_dRhub, self.dQ_dRhub, self.dP_dRhub, self.dT_dRtip, self.dQ_dRtip, self.dP_dRtip, \
        self.dT_dprecurvetip, self.dQ_dprecurvetip, self.dP_dprecurvetip, self.dT_dpresweeptip, self.dQ_dpresweeptip, self.dP_dpresweeptip = \
            np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), \
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.


        self.dT_dNp_tot, self.dQ_dNp_tot, self.dP_dNp_tot, self.dT_dTp_tot, self.dQ_dTp_tot, self.dP_dTp_tot = {}, {}, {}, {}, {}, {}

        args = (r, precurve, presweep, precone, Rhub, Rtip, precurveTip, presweepTip)

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

                dT_dNp, dT_dTp, dQ_dNp, dQ_dTp, dP_dNp, dP_dTp = np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n)), np.zeros((1, n))

                dT_dNp += [x * B / nsec for x in dT_dNp1]
                self.dT_dNp_tot['Np'+str(j+1)] = dT_dNp
                dQ_dNp += [x * B / nsec for x in dQ_dNp1]
                self.dQ_dNp_tot['Np'+str(j+1)] = dQ_dNp
                dP_dNp += [x * B / nsec for x in dP_dNp1]
                self.dP_dNp_tot['Np'+str(j+1)] = dP_dNp
                dT_dTp += [x * B / nsec for x in dT_dTp1]
                self.dT_dTp_tot['Tp'+str(j+1)] = dT_dTp
                dQ_dTp += [x * B / nsec for x in dQ_dTp1]
                self.dQ_dTp_tot['Tp'+str(j+1)] = dQ_dTp
                dP_dTp += [x * B / nsec for x in dP_dTp1]
                self.dP_dTp_tot['Tp'+str(j+1)] = dP_dTp
                self.dT_dr += [x * B / nsec for x in dT_dr1]
                self.dQ_dr += [x * B / nsec for x in dQ_dr1]
                self.dP_dr += [x * B / nsec for x in dP_dr1]
                self.dT_dprecurve += [x * B / nsec for x in dT_dprecurve1]
                self.dQ_dprecurve += [x * B / nsec for x in dQ_dprecurve1]
                self.dP_dprecurve += [x * B / nsec for x in dP_dprecurve1]
                self.dT_dpresweep += [x * B / nsec for x in dT_dpresweep1]
                self.dQ_dpresweep += [x * B / nsec for x in dQ_dpresweep1]
                self.dP_dpresweep += [x * B / nsec for x in dP_dpresweep1]
                self.dT_dprecone += dT_dprecone1 * B / nsec
                self.dQ_dprecone += dQ_dprecone1 * B / nsec
                self.dP_dprecone += dP_dprecone1 * B / nsec
                self.dT_dRhub += dT_dRhub1 * B / nsec
                self.dQ_dRhub += dQ_dRhub1 * B / nsec
                self.dP_dRhub += dP_dRhub1 * B / nsec
                self.dT_dRtip += dT_dRtip1 * B / nsec
                self.dQ_dRtip += dQ_dRtip1 * B / nsec
                self.dP_dRtip += dP_dRtip1 * B / nsec
                self.dT_dprecurvetip += dT_dprecurvetip1 * B / nsec
                self.dQ_dprecurvetip += dQ_dprecurvetip1 * B / nsec
                self.dP_dprecurvetip += dP_dprecurvetip1 * B / nsec
                self.dT_dpresweeptip += dT_dpresweeptip1 * B / nsec
                self.dQ_dpresweeptip += dQ_dpresweeptip1 * B / nsec
                self.dP_dpresweeptip += dP_dpresweeptip1 * B / nsec

        # Power
        P = Q * Omega*pi/30.0  # RPM to rad/s
        # normalize if necessary
        q = 0.5 * rho * Uinf**2
        A = pi * rotorR**2
        CP = P / (q * A * Uinf)
        CT = T / (q * A)
        CQ = Q / (q * rotorR * A)
        unknowns['CP'] = CP[0]
        unknowns['CT'] = CT[0]
        unknowns['CQ'] = CQ[0]
        unknowns['P'] = P[0]
        unknowns['T'] = T[0]
        unknowns['Q'] = Q[0]

        self.dCP_drho = -CP / rho
        self.dCT_drho = -CT / rho
        self.dCQ_drho = -CQ / rho
        self.dCP_drotorR = -2 * P / (q * pi * rotorR**3 * Uinf)
        self.dCT_drotorR = -2 * T / (q * pi * rotorR**3)
        self.dCQ_drotorR = -3 * Q / (q * pi * rotorR**4)
        self.dCP_dUinf = -3 * P / (0.5 * rho * Uinf**4 * A)
        self.dCT_dUinf = -2 * T / (0.5 * rho * Uinf**3 * A)
        self.dCQ_dUinf = -2 * Q / (0.5 * rho * Uinf**3 * rotorR * A)
        self.dP_dOmega = Q * pi / 30.0

    def linearize(self, params, unknowns, resids):
        J = {}
        Uinf = params['Uinf']
        rotorR = params['rotorR']

        q = 0.5 * params['rho'] * Uinf**2
        A = pi * rotorR**2

        nSector = int(params['nSector'])

        for i in range(nSector):
            J['CP', 'Np'+str(i+1)] = self.dP_dNp_tot['Np'+str(i+1)] / (q * A * Uinf)
            J['CP', 'Tp'+str(i+1)] = self.dP_dTp_tot['Tp'+str(i+1)] / (q * A * Uinf)
            J['CT', 'Np'+str(i+1)] = self.dT_dNp_tot['Np'+str(i+1)] / (q * A)
            J['CT', 'Tp'+str(i+1)] = self.dT_dTp_tot['Tp'+str(i+1)] / (q * A)
            J['CQ', 'Np'+str(i+1)] = self.dQ_dNp_tot['Np'+str(i+1)] / (q * rotorR * A)
            J['CQ', 'Tp'+str(i+1)] = self.dQ_dTp_tot['Tp'+str(i+1)] / (q * rotorR * A)
            J['P', 'Np'+str(i+1)] = self.dP_dNp_tot['Np'+str(i+1)]
            J['P', 'Tp'+str(i+1)] = self.dP_dTp_tot['Tp'+str(i+1)]
            J['T', 'Np'+str(i+1)] = self.dT_dNp_tot['Np'+str(i+1)]
            J['T', 'Tp'+str(i+1)] = self.dT_dTp_tot['Tp'+str(i+1)]
            J['Q', 'Np'+str(i+1)] = self.dQ_dNp_tot['Np'+str(i+1)]
            J['Q', 'Tp'+str(i+1)] = self.dQ_dTp_tot['Tp'+str(i+1)]

        J['CP', 'Uinf'] = self.dCP_dUinf
        J['CP', 'Rtip'] = self.dP_dRtip / (q * A * Uinf)
        J['CP', 'Omega'] = self.dP_dOmega / (q * A * Uinf)
        J['CP', 'r'] = self.dP_dr / (q * A * Uinf)
        J['CP', 'precurve'] = self.dP_dprecurve / (q * A * Uinf)
        J['CP', 'presweep'] = self.dP_dpresweep / (q * A * Uinf)
        J['CP', 'presweepTip'] = self.dP_dpresweeptip / (q * A * Uinf)
        J['CP', 'precurveTip'] = self.dP_dprecurvetip / (q * A * Uinf)
        J['CP', 'precone'] = self.dP_dprecone / (q * A * Uinf)
        J['CP', 'rho'] = self.dCP_drho
        J['CP', 'Rhub'] = self.dP_dRhub / (q * A * Uinf)
        J['CP', 'rotorR'] = self.dCP_drotorR

        J['CT', 'Uinf'] = self.dCT_dUinf
        J['CT', 'Rtip'] = self.dT_dRtip / (q * A)
        J['CT', 'Omega'] = 0.0
        J['CT', 'r'] = (self.dT_dr / (q * A))
        J['CT', 'precurve'] = self.dT_dprecurve / (q * A)
        J['CT', 'presweep'] = self.dT_dpresweep / (q * A)
        J['CT', 'presweepTip'] = self.dT_dpresweeptip / (q * A)
        J['CT', 'precurveTip'] = self.dT_dprecurvetip / (q * A)
        J['CT', 'precone'] = self.dT_dprecone / (q * A)
        J['CT', 'rho'] = self.dCT_drho
        J['CT', 'Rhub'] = self.dT_dRhub / (q * A)
        J['CT', 'rotorR'] = self.dCT_drotorR

        J['CQ', 'Uinf'] = self.dCQ_dUinf
        J['CQ', 'Rtip'] = self.dQ_dRtip / (q * rotorR * A)
        J['CQ', 'Omega'] = 0.0
        J['CQ', 'r'] = self.dQ_dr / (q * rotorR * A)
        J['CQ', 'precurve'] = self.dQ_dprecurve / (q * rotorR * A)
        J['CQ', 'presweep'] = self.dQ_dpresweep/ (q * rotorR * A)
        J['CQ', 'presweepTip'] = self.dQ_dpresweeptip / (q * rotorR * A)
        J['CQ', 'precurveTip'] = self.dQ_dprecurvetip / (q * rotorR * A)
        J['CQ', 'precone'] = self.dQ_dprecone / (q * rotorR * A)
        J['CQ', 'rho'] = self.dCQ_drho
        J['CQ', 'Rhub'] = self.dQ_dRhub / (q * rotorR * A)
        J['CQ', 'rotorR'] = self.dCQ_drotorR

        J['P', 'Uinf'] = 0.0
        J['P', 'Rtip'] = self.dP_dRtip
        J['P', 'Omega'] = self.dP_dOmega
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

class Loads_for_RotorSE(Component):
    def __init__(self, n):
        super(Loads_for_RotorSE, self).__init__()
        self.add_param('Rhub', shape=1, units='m')
        self.add_param('r', val=np.zeros(n), units='m')
        self.add_param('Rtip', shape=1, units='m')
        self.add_param('Np', shape=n, units='N/m')
        self.add_param('Tp', shape=n, units='N/m')
        self.add_param('Uinf', shape=1, units='m/s')
        self.add_param('Omega', shape=1, units='rpm')
        self.add_param('pitch', shape=1, units='deg')
        self.add_param('azimuth', shape=1, units='deg')

        self.add_output('loads:r', shape=n+2, units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads:Px', shape=n+2, units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads:Py', shape=n+2, units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads:Pz', shape=n+2, units='N/m', desc='distributed loads in blade-aligned z-direction')

        # corresponding setting for loads
        self.add_output('loads:V', shape=1, units='m/s', desc='hub height wind speed')
        self.add_output('loads:Omega', shape=1, units='rpm', desc='rotor rotation speed')
        self.add_output('loads:pitch', shape=1, units='deg', desc='pitch angle')
        self.add_output('loads:azimuth', shape=1, units='deg', desc='azimuthal angle')

        self.n = n

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['loads:r'] = np.concatenate([[params['Rhub']], params['r'], [params['Rtip']]])
        Np = np.concatenate([[0.0], params['Np'], [0.0]])
        Tp = np.concatenate([[0.0], params['Tp'], [0.0]])

        # conform to blade-aligned coordinate system
        unknowns['loads:Px'] = Np
        unknowns['loads:Py'] = -Tp
        unknowns['loads:Pz'] = 0*Np

        # return other outputs needed
        unknowns['loads:V'] = params['Uinf']
        unknowns['loads:Omega'] = params['Omega']
        unknowns['loads:pitch'] = params['pitch']
        unknowns['loads:azimuth'] = params['azimuth']

    def linearize(self, params, unknowns, resids):
        n = self.n

        dr_dr = np.vstack([np.zeros(n), np.eye(n), np.zeros(n)])
        dr_dRhub = np.zeros(n+2)
        dr_dRtip = np.zeros(n+2)
        dr_dRhub[0] = 1.0
        dr_dRtip[-1] = 1.0

        dV = np.zeros(4*n+10)
        dV[3*n+6] = 1.0
        dOmega = np.zeros(4*n+10)
        dOmega[3*n+7] = 1.0
        dpitch = np.zeros(4*n+10)
        dpitch[3*n+8] = 1.0
        dazimuth = np.zeros(4*n+10)
        dazimuth[3*n+9] = 1.0

        J = {}
        zero = np.zeros(17)
        J['loads:r', 'r'] = dr_dr
        J['loads:r', 'Rhub'] = dr_dRhub
        J['loads:r', 'Rtip'] = dr_dRtip
        J['loads:Px', 'Np'] = np.vstack([zero, np.eye(n), zero])
        J['loads:Py', 'Tp'] = np.vstack([zero, -np.eye(n), zero])
        J['loads:V', 'Uinf'] = 1.0
        J['loads:Omega', 'Omega'] = 1.0
        J['loads:pitch', 'pitch'] = 1.0
        J['loads:azimuth', 'azimuth'] = 1.0

        return J


class BrentGroup(Group):
    def __init__(self, n, i):
        super(BrentGroup, self).__init__()

        sub = self.add('sub', Group(), promotes=['*'])
        sub.add('flow', FlowCondition(), promotes=['*'])
        sub.add('airfoils', AirfoilComp(n, i), promotes=['*'])
        sub.add('bem', BEM(n, i), promotes=['*'])
        sub.ln_solver = ScipyGMRES()
        sub.nl_solver = Brent()
        self.ln_solver = ScipyGMRES()
        self.nl_solver = NLGaussSeidel()
        # self.nl_solver.options['atol'] = 1e-12 #TODO Check
        # self.nl_solver.options['rtol'] = 1e-12
        # sub.ln_solver = ScipyGMRES()
        # self.ln_solver = ScipyGMRES()
        # self.nl_solver = Brent()
        # set standard limits
        epsilon = 1e-6
        phi_lower = epsilon
        # phi_upper = pi/2
        phi_upper = pi/2
        # if errf(phi_lower, *args)*errf(phi_upper, *args) > 0:  # an uncommon but possible case
        #
        #         if errf(-pi/4, *args) < 0 and errf(-epsilon, *args) > 0:
        #             phi_lower = -pi/4
        #             phi_upper = -epsilon
        #         else:
        #             phi_lower = pi/2
        #             phi_upper = pi - epsilon
        sub.nl_solver.options['lower_bound'] = phi_lower
        sub.nl_solver.options['upper_bound'] = phi_upper
        sub.nl_solver.options['state_var'] = 'phi_sub'
        # self.fd_options['force_fd'] = True

    def list_deriv_vars(self):

        inputs = ('Vx', 'Vy', 'chord', 'theta', 'pitch', 'rho', 'mu')
        outputs = ('phi_sub', 'a_sub', 'ap_sub')

        return inputs, outputs

class Loads(Group):
    def __init__(self, n):
        super(Loads, self).__init__()
        self.add('Rhub', IndepVarComp('Rhub', 0.0, units='m'), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0, units='m'), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0, units='deg'), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0, units='deg'), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0, units='m'), promotes=['*'])
        self.add('Uinf', IndepVarComp('Uinf', 0.0, units='m/s'), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', 0.0, units='deg'), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0, units='deg'), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0, units='m'), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0, units='m'), promotes=['*'])
        self.add('azimuth', IndepVarComp('azimuth', 0.0, units='deg'), promotes=['*'])
        self.add('Omega', IndepVarComp('Omega', 0.0, units='rpm'), promotes=['*'])
        self.add('r', IndepVarComp('r', val=np.zeros(n), units='m'), promotes=['*'])
        self.add('chord', IndepVarComp('chord', val=np.zeros(n), units='m'), promotes=['*'])
        self.add('theta', IndepVarComp('theta', val=np.zeros(n), units='deg'), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', val=np.zeros(n), units='m'), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', val=np.zeros(n), units='m'), promotes=['*'])
        self.add('af', IndepVarComp('af', val=np.zeros(n)), promotes=['*'])
        self.add('airfoil_parameterization', IndepVarComp('airfoil_parameterization', val=np.zeros((n, 8))), promotes=['*'])
        self.add('airfoil_analysis_options', IndepVarComp('airfoil_analysis_options', {}, pass_by_obj=True), promotes=['*'])
        self.add('bemoptions', IndepVarComp('bemoptions', {}, pass_by_obj=True), promotes=['*'])
        self.add('shearExp', IndepVarComp('shearExp', 0.0), promotes=['*'])
        self.add('init', CCInit(), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('mux', MUX(n), promotes=['*'])
        pg = self.add('parallel', ParallelGroup(), promotes=['*'])
        for i in range(n):
            pg.add('brent'+str(i+1), BrentGroup(n, i), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'airfoil_parameterization', 'bemoptions', 'airfoil_analysis_options', 'af'])
            self.connect('r', 'brent'+str(i+1)+'.r', src_indices=[i])
            self.connect('chord', 'brent'+str(i+1)+'.chord', src_indices=[i])
            self.connect('theta', 'brent'+str(i+1)+'.theta', src_indices=[i])
            self.connect('Vx', 'brent'+str(i+1)+'.Vx', src_indices=[i])
            self.connect('Vy', 'brent'+str(i+1)+'.Vy', src_indices=[i])
            self.connect('brent'+str(i+1)+'.phi_sub', 'phi'+str(i+1))
            self.connect('brent'+str(i+1)+'.cl_sub', 'cl'+str(i+1))
            self.connect('brent'+str(i+1)+'.cd_sub', 'cd'+str(i+1))
            self.connect('brent'+str(i+1)+'.W_sub', 'W'+str(i+1))
            self.connect('brent'+str(i+1)+'.alpha_sub', 'alpha'+str(i+1))
        self.add('loads', DistributedAeroLoads(n), promotes=['*'])
        self.add('obj_cmp', ExecComp('obj = -max(Np)', Np=np.zeros(n)), promotes=['*'])

class LoadsGroup_to_RotorSE(Group):
    def __init__(self, n):
        super(LoadsGroup_to_RotorSE, self).__init__()
        self.add('init', CCInit(), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('mux', MUX(n), promotes=['*'])
        for i in range(n):
            self.add('brent'+str(i+1), BrentGroup(n, i), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'airfoil_parameterization', 'bemoptions', 'airfoil_analysis_options', 'af'])
            self.connect('r', 'brent'+str(i+1)+'.r', src_indices=[i])
            self.connect('chord', 'brent'+str(i+1)+'.chord', src_indices=[i])
            # self.connect('theta', 'brent'+str(i+1)+'.theta', src_indices=[i])
            self.connect('Vx', 'brent'+str(i+1)+'.Vx', src_indices=[i])
            self.connect('Vy', 'brent'+str(i+1)+'.Vy', src_indices=[i])
            self.connect('brent'+str(i+1)+'.phi_sub', 'phi'+str(i+1))
            self.connect('brent'+str(i+1)+'.cl_sub', 'cl'+str(i+1))
            self.connect('brent'+str(i+1)+'.cd_sub', 'cd'+str(i+1))
            self.connect('brent'+str(i+1)+'.W_sub', 'W'+str(i+1))
            self.connect('brent'+str(i+1)+'.alpha_sub', 'alpha'+str(i+1))
        self.add('loads', DistributedAeroLoads(n), promotes=['*'])

class Sweep(Group):
    def __init__(self, azimuth, n):
        super(Sweep, self).__init__()
        self.add('azimuth', IndepVarComp('azimuth', azimuth, units='deg'), promotes=['*'])
        self.add('wind', WindComponents(n), promotes=['*'])
        self.add('mux', MUX(n), promotes=['*'])
        for i in range(n):
            self.add('brent'+str(i+1), BrentGroup(n, i), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'bemoptions', 'airfoil_parameterization', 'airfoil_analysis_options', 'af'])
            self.connect('Vx', 'brent'+str(i+1)+'.Vx', src_indices=[i])
            self.connect('Vy', 'brent'+str(i+1)+'.Vy', src_indices=[i])
            self.connect('brent'+str(i+1)+'.phi_sub', 'phi'+str(i+1))
            self.connect('brent'+str(i+1)+'.cl_sub', 'cl'+str(i+1))
            self.connect('brent'+str(i+1)+'.cd_sub', 'cd'+str(i+1))
            self.connect('brent'+str(i+1)+'.W_sub', 'W'+str(i+1))
            self.connect('brent'+str(i+1)+'.alpha_sub', 'alpha'+str(i+1))
        self.add('loads', DistributedAeroLoads(n), promotes=['chord', 'rho', 'phi', 'cl', 'cd', 'W'])

class SweepGroup(Group):
    def __init__(self, nSector, n):
        super(SweepGroup, self).__init__()
        self.add('init', CCInit(), promotes=['*'])
        for i in range(nSector):
            azimuth = pi/180.0*360.0*float(i)/nSector
            self.add('group'+str(i+1), Sweep(azimuth, n), promotes=['r', 'Uinf', 'pitch', 'Rtip', 'Omega', 'chord', 'rho', 'mu', 'Rhub', 'hubHt', 'precurve', 'presweep', 'precone', 'tilt', 'yaw', 'pitch', 'shearExp', 'B', 'bemoptions', 'airfoil_parameterization', 'af', 'airfoil_analysis_options'])

class FlowSweep(Group):
    def __init__(self, nSector, n):
        super(FlowSweep, self).__init__()
        self.add('load_group', SweepGroup(nSector, n), promotes=['Uinf', 'Omega', 'pitch', 'Rtip', 'r', 'chord',  'rho', 'mu', 'Rhub', 'rotorR', 'precurve', 'presweep', 'precurveTip', 'precone', 'tilt', 'yaw', 'shearExp', 'hubHt', 'B', 'airfoil_parameterization', 'airfoil_analysis_options', 'af', 'bemoptions'])
        self.add('eval', CCEvaluate(n, nSector), promotes=['Uinf', 'Rtip', 'Omega', 'r', 'Rhub', 'B', 'precurve', 'presweep', 'presweepTip', 'precurveTip', 'precone', 'nSector', 'rotorR', 'rho', 'CP', 'CT', 'CQ', 'P', 'T', 'Q'])
        for i in range(nSector):
            self.connect('load_group.group' + str(i+1) + '.loads.Np', 'eval.Np' + str(i+1))
            self.connect('load_group.group' + str(i+1) + '.loads.Tp', 'eval.Tp' + str(i+1))

class CCBlade(Group):
    def __init__(self, nSector, n, n2):
        super(CCBlade, self).__init__()
        self.add('Uinf', IndepVarComp('Uinf', np.zeros(n2), units='m/s'), promotes=['*'])
        self.add('pitch', IndepVarComp('pitch', np.zeros(n2), units='deg'), promotes=['*'])
        self.add('Omega', IndepVarComp('Omega', np.zeros(n2), units='rpm'), promotes=['*'])
        self.add('r', IndepVarComp('r', np.zeros(n), units='m'), promotes=['*'])
        self.add('chord', IndepVarComp('chord', np.zeros(n), units='m'), promotes=['*'])
        self.add('Rhub', IndepVarComp('Rhub', 0.0, units='m'), promotes=['*'])
        self.add('Rtip', IndepVarComp('Rtip', 0.0, units='m'), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0, units='deg'), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0, units='deg'), promotes=['*'])
        self.add('theta', IndepVarComp('theta', np.zeros(n), units='deg'), promotes=['*'])
        self.add('hubHt', IndepVarComp('hubHt', 0.0, units='m'), promotes=['*'])
        self.add('precurve', IndepVarComp('precurve', np.zeros(n), units='m'), promotes=['*'])
        self.add('presweep', IndepVarComp('presweep', np.zeros(n), units='m'), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0, units='deg'), promotes=['*'])
        self.add('precurveTip', IndepVarComp('precurveTip', 0.0, units='m'), promotes=['*'])
        self.add('presweepTip', IndepVarComp('presweepTip', 0.0, units='m'), promotes=['*'])
        self.add('mu', IndepVarComp('mu', 0.0, units='kg/(m*s)'), promotes=['*'])
        self.add('rho', IndepVarComp('rho', 0.0, units='kg/m**3'), promotes=['*'])
        self.add('shearExp', IndepVarComp('shearExp', 0.0), promotes=['*'])
        self.add('af', IndepVarComp('af', np.zeros(n), pass_by_obj=True), promotes=['*'])
        self.add('airfoil_parameterization', IndepVarComp('airfoil_parameterization', val=np.zeros((n, 8))), promotes=['*'])
        self.add('airfoil_analysis_options', IndepVarComp('airfoil_analysis_options', {}), promotes=['*'])
        self.add('B', IndepVarComp('B', 3, pass_by_obj=True), promotes=['*'])
        self.add('nSector', IndepVarComp('nSector', 4, pass_by_obj=True), promotes=['*'])
        self.add('bemoptions', IndepVarComp('bemoptions', {}, pass_by_obj=True), promotes=['*'])
        self.add('mux_power', MUX_POWER(n2), promotes=['*'])
        pg = self.add('parallel', ParallelGroup(), promotes=['*'])
        for i in range(n2):
            pg.add('results'+str(i), FlowSweep(nSector, n), promotes=['Rhub', 'Rtip', 'precone', 'tilt', 'hubHt', 'precurve', 'presweep', 'yaw', 'precurveTip', 'presweepTip', 'airfoil_parameterization', 'af','airfoil_analysis_options', 'bemoptions', 'B', 'rho', 'mu', 'shearExp', 'nSector', 'r', 'chord'])
            self.connect('Uinf', 'results'+str(i)+'.Uinf', src_indices=[i])
            self.connect('pitch', 'results'+str(i)+'.pitch', src_indices=[i])
            self.connect('Omega', 'results'+str(i)+'.Omega', src_indices=[i])
            self.connect('results'+str(i)+'.CT', 'CT'+str(i+1))
            self.connect('results'+str(i)+'.CQ', 'CQ'+str(i+1))
            self.connect('results'+str(i)+'.CP', 'CP'+str(i+1))
            self.connect('results'+str(i)+'.T', 'T'+str(i+1))
            self.connect('results'+str(i)+'.Q', 'Q'+str(i+1))
            self.connect('results'+str(i)+'.P', 'P'+str(i+1))
            for k in range(nSector):
                for j in range(n):
                    self.connect('theta', 'results'+str(i)+'.load_group.group'+str(k+1)+'.brent'+str(j+1)+'.theta', src_indices=[j])
                    self.connect('chord', 'results'+str(i)+'.load_group.group'+str(k+1)+'.brent'+str(j+1)+'.chord', src_indices=[j])
                    self.connect('r', 'results'+str(i)+'.load_group.group'+str(k+1)+'.brent'+str(j+1)+'.r', src_indices=[j])
        self.add('obj_cmp', ExecComp('obj = -max(CP)', CP=np.zeros(n2)), promotes=['*'])

class CCBlade_to_RotorSE_connection(Group):

    def __init__(self, run_case, nSector, n, n2):
        super(CCBlade_to_RotorSE_connection, self).__init__()
        if run_case == 'power':
            self.add('mux_power', MUX_POWER(n2), promotes=['*'])
            pg = self.add('parallel', ParallelGroup(), promotes=['*'])
            for i in range(n2):
                pg.add('results'+str(i), FlowSweep(nSector, n), promotes=['Rhub', 'Rtip', 'precone', 'tilt', 'hubHt', 'precurve', 'presweep', 'yaw', 'precurveTip', 'presweepTip', 'airfoil_parameterization', 'af', 'airfoil_analysis_options', 'bemoptions', 'B', 'rho', 'mu', 'shearExp', 'nSector', 'r', 'chord'])
                self.connect('Uinf', 'results'+str(i)+'.Uinf', src_indices=[i])
                self.connect('pitch', 'results'+str(i)+'.pitch', src_indices=[i])
                self.connect('Omega', 'results'+str(i)+'.Omega', src_indices=[i])
                self.connect('results'+str(i)+'.CT', 'CT'+str(i+1))
                self.connect('results'+str(i)+'.CQ', 'CQ'+str(i+1))
                self.connect('results'+str(i)+'.CP', 'CP'+str(i+1))
                self.connect('results'+str(i)+'.T', 'T'+str(i+1))
                self.connect('results'+str(i)+'.Q', 'Q'+str(i+1))
                self.connect('results'+str(i)+'.P', 'P'+str(i+1))
                for k in range(nSector):
                    for j in range(n):
                        self.connect('theta', 'results'+str(i)+'.load_group.group'+str(k+1)+'.brent'+str(j+1)+'.theta', src_indices=[j])
                        self.connect('chord', 'results'+str(i)+'.load_group.group'+str(k+1)+'.brent'+str(j+1)+'.chord', src_indices=[j])
                        self.connect('r', 'results'+str(i)+'.load_group.group'+str(k+1)+'.brent'+str(j+1)+'.r', src_indices=[j])
        elif run_case == 'loads':
            self.add('init', CCInit(), promotes=['*'])
            self.add('wind', WindComponents(n), promotes=['*'])
            self.add('mux', MUX(n), promotes=['*'])
            for i in range(n):
                self.add('brent'+str(i+1), BrentGroup(n, i), promotes=['Rhub', 'Rtip', 'rho', 'mu', 'Omega', 'B', 'pitch', 'airfoil_parameterization', 'bemoptions', 'airfoil_analysis_options', 'af'])
                self.connect('r', 'brent'+str(i+1)+'.r', src_indices=[i])
                self.connect('chord', 'brent'+str(i+1)+'.chord', src_indices=[i])
                self.connect('theta', 'brent'+str(i+1)+'.theta', src_indices=[i])
                self.connect('Vx', 'brent'+str(i+1)+'.Vx', src_indices=[i])
                self.connect('Vy', 'brent'+str(i+1)+'.Vy', src_indices=[i])
                self.connect('brent'+str(i+1)+'.phi_sub', 'phi'+str(i+1))
                self.connect('brent'+str(i+1)+'.cl_sub', 'cl'+str(i+1))
                self.connect('brent'+str(i+1)+'.cd_sub', 'cd'+str(i+1))
                self.connect('brent'+str(i+1)+'.W_sub', 'W'+str(i+1))
                self.connect('brent'+str(i+1)+'.alpha_sub', 'alpha'+str(i+1))
            self.add('loads', DistributedAeroLoads(n), promotes=['*'])
            self.add('loads_rotor', Loads_for_RotorSE(n), promotes=['*'])

        else:
            print 'Warning: run_case needs to be either power or loads.'


if __name__ == "__main__":
    global function_calls_XFOIL
    function_calls_XFOIL = 0

    if MPI: # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl
    else:
        # if you didn't use `mpirun`, then use the numpy data passing
        from openmdao.api import BasicImpl as impl

    def mpi_print(prob, *args):
        """ helper function to only print on rank 0"""
        if prob.root.comm.rank == 0:
            print args


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
    # airfoil_analysis_options = dict(AirfoilParameterization='CST', CFDorXFOIL='XFOIL', FDorCS='CS', iterations=20, processors=0) ### AirfoilParameterization = ('CST', 'Files', 'NACA'), CFDorXFOIL=('XFOIL', 'CFD'), FDorCS=('FD', 'CS'), iterations=20, processors=0)
    # airfoil_analysis_options = dict(AirfoilParameterization='CST', CFDorXFOIL='XFOIL', FDorCS='FD', iterations=20, processors=0) ### AirfoilParameterization = ('CST', 'Files', 'NACA'), CFDorXFOIL=('XFOIL', 'CFD'), FDorCS=('FD', 'CS'), iterations=20, processors=0)
    airfoil_analysis_options = dict(AirfoilParameterization='CST', CFDorXFOIL='XFOIL', FDorCS='CS', iterations=250, processors=0)
    # atmosphere
    rho = 1.225
    mu = 1.81206e-5

    import os
    w0 = [-0.17200255338600826, -0.13744743777735921, -0.24288986290945222, 0.15085289615063024, 0.20650016452789369, 0.35540642522188848, 0.32797634888819488, 0.2592276816645861]
    wl_1 = [-0.17200255338600826, -0.13744743777735921, -0.24288986290945222, 0.15085289615063024, 0.20650016452789369, 0.35540642522188848, 0.32797634888819488, 0.2592276816645861]
    wl_2 = [-0.19600050454371795, -0.28861738331958697, -0.20594891135118523, 0.19143138186871009, 0.22876347660120994, 0.39940768357615447, 0.28896745336793572, 0.29519782561050112]
    wl_3 = [-0.27413320446357803, -0.40701949670950271, -0.29237424992338562, 0.27867844397438357, 0.23582783854698663, 0.43718573158380936, 0.25389099250498309, 0.31090780344061775]
    wl_4 = [-0.29817561716727448, -0.67909473119918973, -0.15737231648880162, 0.12798260780188203, 0.2842322211249545, 0.46026650967959087, 0.21705062978922526, 0.33758303223369945]
    wl_5 = [-0.38027535114760153, -0.75920832612723133, -0.21834261746205941, 0.086359012110824224, 0.38364567865371835, 0.48445264573011815, 0.26999944648962521, 0.34675843509167931]
    wl_6 = [-0.49209940079930325, -0.72861624849999296, -0.38147646962813714, 0.13679205926397994, 0.50396496117640877, 0.54798355691567613, 0.37642896917099616, 0.37017796580840234]

    CST_full = [-0.17200255338600826, -0.13744743777735921, -0.24288986290945222, 0.15085289615063024, 0.20650016452789369, 0.35540642522188848, 0.32797634888819488, 0.2592276816645861,
            -0.19600050454371795, -0.28861738331958697, -0.20594891135118523, 0.19143138186871009, 0.22876347660120994, 0.39940768357615447, 0.28896745336793572, 0.29519782561050112,
            -0.27413320446357803, -0.40701949670950271, -0.29237424992338562, 0.27867844397438357, 0.23582783854698663, 0.43718573158380936, 0.25389099250498309, 0.31090780344061775,
            -0.29817561716727448, -0.67909473119918973, -0.15737231648880162, 0.12798260780188203, 0.2842322211249545, 0.46026650967959087, 0.21705062978922526, 0.33758303223369945,
            -0.38027535114760153, -0.75920832612723133, -0.21834261746205941, 0.086359012110824224, 0.38364567865371835, 0.48445264573011815, 0.26999944648962521, 0.34675843509167931,
            -0.49209940079930325, -0.72861624849999296, -0.38147646962813714, 0.13679205926397994, 0.50396496117640877, 0.54798355691567613, 0.37642896917099616, 0.37017796580840234]

    CST = [[wl_6],[wl_6], [wl_6], [wl_5], [wl_4], [wl_3], [wl_2], [wl_1]]

    basepath = '5MW_AFFiles' + os.path.sep
    afinit = CCAirfoil.initFromAerodynFile
    afinit2 = CCAirfoil.initFromCST  # just for shorthand
    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    # airfoil_types = [0]*8
    # airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    # airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    # airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
    # airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
    # airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
    # airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
    # airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
    # airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')


    for i in range(len(airfoil_types)-2):
        airfoil_types[i+2] = afinit2(CST[i+2], airfoil_analysis_options['CFDorXFOIL'], airfoil_analysis_options['processors'], airfoil_analysis_options['iterations'])

    # airfoil_types = [0]*8
    # airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    # airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')

    # for i in range(len(airfoil_types)-2):
    #     airfoil_types[i+2] = afinit2(CST[i+2])
    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    af = [0]*len(r)
    for i in range(len(r)):
        af[i] = airfoil_types[af_idx[i]]

    # CST_full_2 = np.zeros(len(CST_full))
    # for i in range(len(CST_full_2)):
    #     CST_full_2[i] = CST_full[i]
    CST_full = np.zeros((17, 8))
    for i in range(len(r)):
        for j in range(8):
            CST_full[i][j] = CST[af_idx[i]][0][j]
    CST = CST_full.reshape(17, 1, 8)

    # afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
    # basepath = '5MW_AFFiles' + os.path.sep
    #
    # # load all airfoils
    # airfoil_types = [0]*8
    # airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
    # airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
    # airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
    # airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
    # airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
    # airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
    # airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
    # airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')
    #
    # # place at appropriate radial stations
    # af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
    #
    # af = [0]*len(r)
    # for i in range(len(r)):
    #     af[i] = airfoil_types[af_idx[i]]


    tilt = -5.0
    precone = 2.5
    yaw = 0.0
    shearExp = 0.2
    hubHt = 80.0
    nSector = 4 #8

    # set conditions
    Uinf = 10.0
    tsr = 7.55
    pitch = 0.0
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    azimuth = 90.
    n = len(r)


    #### Test LoadsGroup
    # loads = Problem(impl=impl)
    # root = loads.root = Loads(n)

    # loads.driver = pyOptSparseDriver()
    # loads.driver.options['optimizer'] = 'SNOPT'
    # loads.driver.add_desvar('Omega', lower=1.5, upper=25.0)
    # # loads.driver.add_desvar('airfoil_parameterization', lower=-1.0, upper=1.0)
    # loads.driver.add_objective('obj')
    #
    # recorder = DumpRecorder('optimization2.log')
    # # adding specific includes works, but leaving it out results in a crash
    # # recorder.options['includes'] = ['x']
    # loads.driver.add_recorder(recorder)
    #
    # recorder = SqliteRecorder('recorder')
    # recorder.options['record_params'] = True
    # recorder.options['record_metadata'] = True
    # ccblade.driver.add_recorder(recorder)


    # loads.setup(check=False)
    #
    # loads['Rhub'] = Rhub
    # loads['Rtip'] = Rtip
    # loads['r'] = r
    # loads['chord'] = chord
    # loads['theta'] = np.radians(theta)
    # loads['rho'] = rho
    # loads['mu'] = mu
    # loads['tilt'] = np.radians(tilt)
    # loads['precone'] = np.radians(precone)
    # loads['yaw'] = np.radians(yaw)
    # loads['shearExp'] = shearExp
    # loads['hubHt'] = hubHt
    # loads['Uinf'] = Uinf
    # loads['Omega'] = Omega
    # loads['pitch'] = np.radians(pitch)
    # loads['azimuth'] = np.radians(azimuth)
    # loads['airfoil_parameterization'] = CST
    # loads['bemoptions'] = bemoptions
    # loads['airfoil_analysis_options'] = airfoil_analysis_options
    #
    # loads.run()
    #
    # print 'Np', loads['Np']
    # print 'Tp', loads['Tp']
    # print 'Omega', loads['Omega']

    ##### Test CCBlade
    Uinf = np.array([10.0])  # Needs to be an array for CCBlade group
    tsr = 7.55
    pitch = np.array([0.0])
    Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
    tsr = np.linspace(2, 14, 20)
    Omega = 10.0 * np.ones_like(tsr)
    Uinf = Omega*pi/30.0 * Rtip/tsr


    tsr = np.linspace(2, 14, 20)
    Omega = 10.0 * np.ones_like(tsr)
    Uinf = Omega*pi/30.0 * Rtip/tsr
    pitch = np.zeros_like(tsr)
    n2 = len(Uinf)

    ccblade = Problem(impl=impl)
    ccblade.root = CCBlade(nSector, n, n2)

    ### SETUP OPTIMIZATION
    # ccblade.driver = pyOptSparseDriver()
    # ccblade.driver.options['optimizer'] = 'SNOPT'
    # ccblade.driver.add_desvar('Omega', lower=1.5, upper=25.0)
    # ccblade.driver.add_desvar('airfoil_parameterization', lower=-1.0, upper=1.0)
    # ccblade.driver.add_objective('obj')
    # recorder = SqliteRecorder('recorder')
    # recorder = DumpRecorder('optimization_cp.log')
    # recorder.options['record_params'] = True
    # recorder.options['record_metadata'] = True
    # ccblade.driver.add_recorder(recorder)

    ccblade.setup(check=False)

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
    ccblade['Omega'] = Omega
    ccblade['pitch'] = np.radians(pitch)
    ccblade['airfoil_parameterization'] = CST
    ccblade['bemoptions'] = bemoptions
    ccblade['airfoil_analysis_options'] = airfoil_analysis_options
    ccblade['af'] = af

    # import time
    # t0 = time.time()
    ccblade.run()
    partial = open('partial.txt', 'w')
    total = open('total.txt', 'w')
    # partial_dict = ccblade.check_partial_derivatives(out_stream=partial)
    # ccblade.check_total_derivatives(out_stream=total)
    # t = time.time()
    # print t - t0

    print 'CP', ccblade['CP']
    print 'CT', ccblade['CT']
    print 'CQ', ccblade['CQ']
    # print ccblade['airfoil_parameterization']

