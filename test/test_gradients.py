"""
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

import unittest
import numpy as np
from math import pi
from os import path
from openmdao.api import Problem

from ccblade import CCAirfoil, CCBlade, LoadsGroup

class TestGradientsClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
       pass


class TestGradients(TestGradientsClass):

    @classmethod
    def setUpClass(self):
        super(TestGradients, self).setUpClass()
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

        # atmosphere
        rho = 1.225
        mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
        basepath = path.join(path.dirname(path.realpath(__file__)), '5MW_AFFiles') + path.sep

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
        azimuth = 90.0

        bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)
        n = len(r)

        ## Load gradients
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
        loads['Omega'] = Omega
        loads['pitch'] = np.radians(pitch)
        loads['azimuth'] = np.radians(azimuth)
        loads['af'] = af
        loads['bemoptions'] = bemoptions

        loads.run()
        loads_test_total_gradients = open('loads_test_total_gradients.txt', 'w')
        loads_gradients = loads.check_total_derivatives(out_stream=loads_test_total_gradients, unknown_list=['Np', 'Tp'])
        # loads_partials = loads.check_partial_derivatives(out_stream=loads_test_total_gradients)

        ## Power Gradients
        ccblade = Problem()
        root = ccblade.root = CCBlade(nSector, n)
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
        ccblade['Omega'] = Omega
        ccblade['pitch'] = np.radians(pitch)

        ccblade.run()

        print "Generating gradients for Test 1. Please wait..."
        power_test_total_gradients = open('power_test_total_gradients.txt', 'w')
        power_gradients = ccblade.check_total_derivatives(out_stream=power_test_total_gradients, unknown_list=['CP', 'CT', 'CQ', 'P', 'T', 'Q', 'Omega'])
        # power_partial = ccblade.check_partial_derivatives(out_stream=power_test_total_gradients)
        print "Gradients generated for Test 1."

        self.loads_gradients = loads_gradients
        self.power_gradients = power_gradients
        self.n = len(r)
        self.npts = 1  # len(Uinf)

    def test_dr1(self):

        dNp_dr = self.loads_gradients['Np', 'r']['J_fwd']
        dTp_dr = self.loads_gradients['Tp', 'r']['J_fwd']
        dNp_dr_fd = self.loads_gradients['Np', 'r']['J_fd']
        dTp_dr_fd = self.loads_gradients['Tp', 'r']['J_fd']

        np.testing.assert_allclose(dNp_dr_fd, dNp_dr, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dr_fd, dTp_dr, rtol=1e-4, atol=1e-8)


    def test_dr2(self):

        dT_dr = self.power_gradients['T', 'r']['J_fwd']
        dQ_dr = self.power_gradients['Q', 'r']['J_fwd']
        dP_dr = self.power_gradients['P', 'r']['J_fwd']
        dT_dr_fd = self.power_gradients['T', 'r']['J_fd']
        dQ_dr_fd = self.power_gradients['Q', 'r']['J_fd']
        dP_dr_fd = self.power_gradients['P', 'r']['J_fd']


        np.testing.assert_allclose(dT_dr_fd, dT_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dr_fd, dQ_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dr_fd, dP_dr, rtol=3e-4, atol=1e-8)


    def test_dr3(self):

        dCT_dr = self.power_gradients['CT', 'r']['J_fwd']
        dCQ_dr = self.power_gradients['CQ', 'r']['J_fwd']
        dCP_dr = self.power_gradients['CP', 'r']['J_fwd']
        dCT_dr_fd = self.power_gradients['CT', 'r']['J_fd']
        dCQ_dr_fd = self.power_gradients['CQ', 'r']['J_fd']
        dCP_dr_fd = self.power_gradients['CP', 'r']['J_fd']

        np.testing.assert_allclose(dCT_dr_fd, dCT_dr, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dr_fd, dCQ_dr, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dr_fd, dCP_dr, rtol=3e-4, atol=1e-8)



    def test_dchord1(self):

        dNp_dchord = self.loads_gradients['Np', 'chord']['J_fwd']
        dTp_dchord = self.loads_gradients['Tp', 'chord']['J_fwd']
        dNp_dchord_fd = self.loads_gradients['Np', 'chord']['J_fd']
        dTp_dchord_fd = self.loads_gradients['Tp', 'chord']['J_fd']

        np.testing.assert_allclose(dNp_dchord_fd, dNp_dchord, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dchord_fd, dTp_dchord, rtol=5e-5, atol=1e-8)



    def test_dchord2(self):

        dT_dchord = self.power_gradients['T', 'chord']['J_fwd']
        dQ_dchord = self.power_gradients['Q', 'chord']['J_fwd']
        dP_dchord = self.power_gradients['P', 'chord']['J_fwd']
        dT_dchord_fd = self.power_gradients['T', 'chord']['J_fd']
        dQ_dchord_fd = self.power_gradients['Q', 'chord']['J_fd']
        dP_dchord_fd = self.power_gradients['P', 'chord']['J_fd']

        np.testing.assert_allclose(dT_dchord_fd, dT_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dQ_dchord_fd, dQ_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dchord_fd, dP_dchord, rtol=7e-5, atol=1e-8)

    def test_dchord3(self):

        dCT_dchord = self.power_gradients['CT', 'chord']['J_fwd']
        dCQ_dchord = self.power_gradients['CQ', 'chord']['J_fwd']
        dCP_dchord = self.power_gradients['CP', 'chord']['J_fwd']
        dCT_dchord_fd = self.power_gradients['CT', 'chord']['J_fd']
        dCQ_dchord_fd = self.power_gradients['CQ', 'chord']['J_fd']
        dCP_dchord_fd = self.power_gradients['CP', 'chord']['J_fd']

        np.testing.assert_allclose(dCT_dchord_fd, dCT_dchord, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCQ_dchord_fd, dCQ_dchord, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dchord_fd, dCP_dchord, rtol=7e-5, atol=1e-8)




    def test_dtheta1(self):

        dNp_dtheta = self.loads_gradients['Np', 'theta']['J_fwd']
        dTp_dtheta = self.loads_gradients['Tp', 'theta']['J_fwd']
        dNp_dtheta_fd = self.loads_gradients['Np', 'theta']['J_fwd']
        dTp_dtheta_fd = self.loads_gradients['Tp', 'theta']['J_fwd']

        np.testing.assert_allclose(dNp_dtheta_fd, dNp_dtheta, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dtheta_fd, dTp_dtheta, rtol=1e-4, atol=1e-8)


    def test_dtheta2(self):

        dT_dtheta = self.power_gradients['T', 'theta']['J_fwd']
        dQ_dtheta = self.power_gradients['Q', 'theta']['J_fwd']
        dP_dtheta = self.power_gradients['P', 'theta']['J_fwd']
        dT_dtheta_fd = self.power_gradients['T', 'theta']['J_fd']
        dQ_dtheta_fd = self.power_gradients['Q', 'theta']['J_fd']
        dP_dtheta_fd = self.power_gradients['P', 'theta']['J_fd']

        np.testing.assert_allclose(dT_dtheta_fd, dT_dtheta, rtol=7e-4, atol=1e-6) # TODO: rtol=7e-5, atol=1e-8
        np.testing.assert_allclose(dQ_dtheta_fd, dQ_dtheta, rtol=7e-4, atol=1e-6)
        np.testing.assert_allclose(dP_dtheta_fd, dP_dtheta, rtol=7e-4, atol=1e-6)



    def test_dtheta3(self):

        dCT_dtheta = self.power_gradients['CT', 'theta']['J_fwd']
        dCQ_dtheta = self.power_gradients['CQ', 'theta']['J_fwd']
        dCP_dtheta = self.power_gradients['CP', 'theta']['J_fwd']
        dCT_dtheta_fd = self.power_gradients['CT', 'theta']['J_fd']
        dCQ_dtheta_fd = self.power_gradients['CQ', 'theta']['J_fd']
        dCP_dtheta_fd = self.power_gradients['CP', 'theta']['J_fd']

        np.testing.assert_allclose(dCT_dtheta_fd, dCT_dtheta, rtol=5e-6, atol=1e-8)
        np.testing.assert_allclose(dCQ_dtheta_fd, dCQ_dtheta, rtol=7e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dtheta_fd, dCP_dtheta, rtol=7e-5, atol=1e-8)



    def test_dRhub1(self):

        dNp_dRhub = self.loads_gradients['Np', 'Rhub']['J_fwd']
        dTp_dRhub = self.loads_gradients['Tp', 'Rhub']['J_fwd']

        dNp_dRhub_fd = self.loads_gradients['Np', 'Rhub']['J_fd']
        dTp_dRhub_fd = self.loads_gradients['Tp', 'Rhub']['J_fd']

        np.testing.assert_allclose(dNp_dRhub_fd, dNp_dRhub, rtol=1e-5, atol=1.5e-6) # TODO
        np.testing.assert_allclose(dTp_dRhub_fd, dTp_dRhub, rtol=1e-4, atol=1.5e-6)


    def test_dRhub2(self):

        dT_dRhub = self.power_gradients['T', 'Rhub']['J_fwd']
        dQ_dRhub = self.power_gradients['Q', 'Rhub']['J_fwd']
        dP_dRhub = self.power_gradients['P', 'Rhub']['J_fwd']

        dT_dRhub_fd = self.power_gradients['T', 'Rhub']['J_fd']
        dQ_dRhub_fd = self.power_gradients['Q', 'Rhub']['J_fd']
        dP_dRhub_fd = self.power_gradients['P', 'Rhub']['J_fd']

        np.testing.assert_allclose(dT_dRhub_fd, dT_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dRhub_fd, dQ_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dRhub_fd, dP_dRhub, rtol=5e-5, atol=1e-8)


    def test_dRhub3(self):

        dCT_dRhub = self.power_gradients['CT', 'Rhub']['J_fwd']
        dCQ_dRhub = self.power_gradients['CQ', 'Rhub']['J_fwd']
        dCP_dRhub = self.power_gradients['CP', 'Rhub']['J_fwd']

        dCT_dRhub_fd = self.power_gradients['CT', 'Rhub']['J_fd']
        dCQ_dRhub_fd = self.power_gradients['CQ', 'Rhub']['J_fd']
        dCP_dRhub_fd = self.power_gradients['CP', 'Rhub']['J_fd']

        np.testing.assert_allclose(dCT_dRhub_fd, dCT_dRhub, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dRhub_fd, dCQ_dRhub, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dRhub_fd, dCP_dRhub, rtol=5e-5, atol=1e-8)


    def test_dRtip1(self):

        dNp_dRtip = self.loads_gradients['Np', 'Rtip']['J_fwd']
        dTp_dRtip = self.loads_gradients['Tp', 'Rtip']['J_fwd']

        dNp_dRtip_fd = self.loads_gradients['Np', 'Rtip']['J_fd']
        dTp_dRtip_fd = self.loads_gradients['Tp', 'Rtip']['J_fd']

        np.testing.assert_allclose(dNp_dRtip_fd, dNp_dRtip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dRtip_fd, dTp_dRtip, rtol=1e-4, atol=1e-8)


    def test_dRtip2(self):

        dT_dRtip = self.power_gradients['T', 'Rtip']['J_fwd']
        dQ_dRtip = self.power_gradients['Q', 'Rtip']['J_fwd']
        dP_dRtip = self.power_gradients['P', 'Rtip']['J_fwd']

        dT_dRtip_fd = self.power_gradients['T', 'Rtip']['J_fd']
        dQ_dRtip_fd = self.power_gradients['Q', 'Rtip']['J_fd']
        dP_dRtip_fd = self.power_gradients['P', 'Rtip']['J_fd']

        np.testing.assert_allclose(dT_dRtip_fd, dT_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dRtip_fd, dQ_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dRtip_fd, dP_dRtip, rtol=5e-5, atol=1e-8)


    def test_dRtip3(self):

        dCT_dRtip = self.power_gradients['CT', 'Rtip']['J_fwd']
        dCQ_dRtip = self.power_gradients['CQ', 'Rtip']['J_fwd']
        dCP_dRtip = self.power_gradients['CP', 'Rtip']['J_fwd']

        dCT_dRtip_fd = self.power_gradients['CT', 'Rtip']['J_fd']
        dCQ_dRtip_fd = self.power_gradients['CQ', 'Rtip']['J_fd']
        dCP_dRtip_fd = self.power_gradients['CP', 'Rtip']['J_fd']

        np.testing.assert_allclose(dCT_dRtip_fd, dCT_dRtip, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dRtip_fd, dCQ_dRtip, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dRtip_fd, dCP_dRtip, rtol=5e-5, atol=1e-8)


    def test_dprecone1(self):

        dNp_dprecone = self.loads_gradients['Np', 'precone']['J_fwd']
        dTp_dprecone = self.loads_gradients['Tp', 'precone']['J_fwd']

        dNp_dprecone_fd = self.loads_gradients['Np', 'precone']['J_fd']
        dTp_dprecone_fd = self.loads_gradients['Tp', 'precone']['J_fd']

        np.testing.assert_allclose(dNp_dprecone_fd, dNp_dprecone, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(dTp_dprecone_fd, dTp_dprecone, rtol=1e-5, atol=1e-7)



    def test_dprecone2(self):

        dT_dprecone = self.power_gradients['T', 'precone']['J_fwd']
        dQ_dprecone = self.power_gradients['Q', 'precone']['J_fwd']
        dP_dprecone = self.power_gradients['P', 'precone']['J_fwd']

        dT_dprecone_fd = self.power_gradients['T', 'precone']['J_fd']
        dQ_dprecone_fd = self.power_gradients['Q', 'precone']['J_fd']
        dP_dprecone_fd = self.power_gradients['P', 'precone']['J_fd']

        np.testing.assert_allclose(dT_dprecone_fd, dT_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecone_fd, dQ_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dprecone_fd, dP_dprecone, rtol=5e-5, atol=1e-8)


    def test_dprecone3(self):

        dCT_dprecone = self.power_gradients['CT', 'precone']['J_fwd']
        dCQ_dprecone = self.power_gradients['CQ', 'precone']['J_fwd']
        dCP_dprecone = self.power_gradients['CP', 'precone']['J_fwd']

        dCT_dprecone_fd = self.power_gradients['CT', 'precone']['J_fd']
        dCQ_dprecone_fd = self.power_gradients['CQ', 'precone']['J_fd']
        dCP_dprecone_fd = self.power_gradients['CP', 'precone']['J_fd']

        np.testing.assert_allclose(dCT_dprecone_fd, dCT_dprecone, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecone_fd, dCQ_dprecone, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecone_fd, dCP_dprecone, rtol=5e-5, atol=1e-8)


    def test_dtilt1(self):

        dNp_dtilt = self.loads_gradients['Np', 'tilt']['J_fwd']
        dTp_dtilt = self.loads_gradients['Tp', 'tilt']['J_fwd']

        dNp_dtilt_fd = self.loads_gradients['Np', 'tilt']['J_fd']
        dTp_dtilt_fd = self.loads_gradients['Tp', 'tilt']['J_fd']

        np.testing.assert_allclose(dNp_dtilt_fd, dNp_dtilt, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dtilt_fd, dTp_dtilt, rtol=1e-5, atol=1e-8)


    def test_dtilt2(self):

        dT_dtilt = self.power_gradients['T', 'tilt']['J_fwd']
        dQ_dtilt = self.power_gradients['Q', 'tilt']['J_fwd']
        dP_dtilt = self.power_gradients['P', 'tilt']['J_fwd']

        dT_dtilt_fd = self.power_gradients['T', 'tilt']['J_fd']
        dQ_dtilt_fd = self.power_gradients['Q', 'tilt']['J_fd']
        dP_dtilt_fd = self.power_gradients['P', 'tilt']['J_fd']

        np.testing.assert_allclose(dT_dtilt_fd, dT_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dtilt_fd, dQ_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dtilt_fd, dP_dtilt, rtol=5e-5, atol=1e-8)


    def test_dtilt3(self):

        dCT_dtilt = self.power_gradients['CT', 'tilt']['J_fwd']
        dCQ_dtilt = self.power_gradients['CQ', 'tilt']['J_fwd']
        dCP_dtilt = self.power_gradients['CP', 'tilt']['J_fwd']

        dCT_dtilt_fd = self.power_gradients['CT', 'tilt']['J_fd']
        dCQ_dtilt_fd = self.power_gradients['CQ', 'tilt']['J_fd']
        dCP_dtilt_fd = self.power_gradients['CP', 'tilt']['J_fd']

        np.testing.assert_allclose(dCT_dtilt_fd, dCT_dtilt, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dtilt_fd, dCQ_dtilt, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dtilt_fd, dCP_dtilt, rtol=5e-5, atol=1e-8)


    def test_dhubht1(self):

        dNp_dhubht = self.loads_gradients['Np', 'hubHt']['J_fwd']
        dTp_dhubht = self.loads_gradients['Tp', 'hubHt']['J_fwd']

        dNp_dhubht_fd = self.loads_gradients['Np', 'hubHt']['J_fd']
        dTp_dhubht_fd = self.loads_gradients['Tp', 'hubHt']['J_fd']

        np.testing.assert_allclose(dNp_dhubht_fd, dNp_dhubht, rtol=1e-4, atol=1e-6) # TODO rtol = 1e-5 atol=1e-8
        np.testing.assert_allclose(dTp_dhubht_fd, dTp_dhubht, rtol=1e-4, atol=1e-6)


    def test_dhubht2(self):

        dT_dhubht = self.power_gradients['T', 'hubHt']['J_fwd']
        dQ_dhubht = self.power_gradients['Q', 'hubHt']['J_fwd']
        dP_dhubht = self.power_gradients['P', 'hubHt']['J_fwd']

        dT_dhubht_fd = self.power_gradients['T', 'hubHt']['J_fd']
        dQ_dhubht_fd = self.power_gradients['Q', 'hubHt']['J_fd']
        dP_dhubht_fd = self.power_gradients['P', 'hubHt']['J_fd']

        np.testing.assert_allclose(dT_dhubht_fd, dT_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dhubht_fd, dQ_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dhubht_fd, dP_dhubht, rtol=5e-5, atol=1e-8)



    def test_dhubht3(self):

        dCT_dhubht = self.power_gradients['CT', 'hubHt']['J_fwd']
        dCQ_dhubht = self.power_gradients['CQ', 'hubHt']['J_fwd']
        dCP_dhubht = self.power_gradients['CP', 'hubHt']['J_fwd']

        dCT_dhubht_fd = self.power_gradients['CT', 'hubHt']['J_fd']
        dCQ_dhubht_fd = self.power_gradients['CQ', 'hubHt']['J_fd']
        dCP_dhubht_fd = self.power_gradients['CP', 'hubHt']['J_fd']

        np.testing.assert_allclose(dCT_dhubht_fd, dCT_dhubht, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dhubht_fd, dCQ_dhubht, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dhubht_fd, dCP_dhubht, rtol=5e-5, atol=1e-8)



    def test_dyaw1(self):

        dNp_dyaw = self.loads_gradients['Np', 'yaw']['J_fwd']
        dTp_dyaw = self.loads_gradients['Tp', 'yaw']['J_fwd']

        dNp_dyaw_fd = self.loads_gradients['Np', 'yaw']['J_fd']
        dTp_dyaw_fd = self.loads_gradients['Tp', 'yaw']['J_fd']

        np.testing.assert_allclose(dNp_dyaw_fd, dNp_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dyaw_fd, dTp_dyaw, rtol=1e-5, atol=1e-8)


    def test_dyaw2(self):

        dT_dyaw = self.power_gradients['T', 'yaw']['J_fwd']
        dQ_dyaw = self.power_gradients['Q', 'yaw']['J_fwd']
        dP_dyaw = self.power_gradients['P', 'yaw']['J_fwd']

        dT_dyaw_fd = self.power_gradients['T', 'yaw']['J_fd']
        dQ_dyaw_fd = self.power_gradients['Q', 'yaw']['J_fd']
        dP_dyaw_fd = self.power_gradients['P', 'yaw']['J_fd']

        np.testing.assert_allclose(dT_dyaw_fd, dT_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dyaw_fd, dQ_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dyaw_fd, dP_dyaw, rtol=5e-5, atol=1e-8)



    def test_dyaw3(self):

        dCT_dyaw = self.power_gradients['CT', 'yaw']['J_fwd']
        dCQ_dyaw = self.power_gradients['CQ', 'yaw']['J_fwd']
        dCP_dyaw = self.power_gradients['CP', 'yaw']['J_fwd']

        dCT_dyaw_fd = self.power_gradients['CT', 'yaw']['J_fd']
        dCQ_dyaw_fd = self.power_gradients['CQ', 'yaw']['J_fd']
        dCP_dyaw_fd = self.power_gradients['CP', 'yaw']['J_fd']

        np.testing.assert_allclose(dCT_dyaw_fd, dCT_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dyaw_fd, dCQ_dyaw, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dyaw_fd, dCP_dyaw, rtol=5e-5, atol=1e-8)



    def test_dazimuth1(self):

        dNp_dazimuth = self.loads_gradients['Np', 'azimuth']['J_fwd']
        dTp_dazimuth = self.loads_gradients['Tp', 'azimuth']['J_fwd']

        dNp_dazimuth_fd = self.loads_gradients['Np', 'azimuth']['J_fd']
        dTp_dazimuth_fd = self.loads_gradients['Tp', 'azimuth']['J_fd']

        np.testing.assert_allclose(dNp_dazimuth_fd, dNp_dazimuth, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dazimuth_fd, dTp_dazimuth, rtol=1e-5, atol=1e-6)


    def test_dUinf1(self):

        dNp_dUinf = self.loads_gradients['Np', 'Uinf']['J_fwd']
        dTp_dUinf = self.loads_gradients['Tp', 'Uinf']['J_fwd']

        dNp_dUinf_fd = self.loads_gradients['Np', 'Uinf']['J_fd']
        dTp_dUinf_fd = self.loads_gradients['Tp', 'Uinf']['J_fd']

        np.testing.assert_allclose(dNp_dUinf_fd, dNp_dUinf, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dUinf_fd, dTp_dUinf, rtol=1e-5, atol=1e-6)


    def test_dUinf2(self):

        dT_dUinf = self.power_gradients['T', 'Uinf']['J_fwd']
        dQ_dUinf = self.power_gradients['Q', 'Uinf']['J_fwd']
        dP_dUinf = self.power_gradients['P', 'Uinf']['J_fwd']

        dT_dUinf_fd = self.power_gradients['T', 'Uinf']['J_fd']
        dQ_dUinf_fd = self.power_gradients['Q', 'Uinf']['J_fd']
        dP_dUinf_fd = self.power_gradients['P', 'Uinf']['J_fd']

        np.testing.assert_allclose(dT_dUinf_fd, dT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dUinf_fd, dQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dUinf_fd, dP_dUinf, rtol=5e-5, atol=1e-8)



    def test_dUinf3(self):

        dCT_dUinf = self.power_gradients['CT', 'Uinf']['J_fwd']
        dCQ_dUinf = self.power_gradients['CQ', 'Uinf']['J_fwd']
        dCP_dUinf = self.power_gradients['CP', 'Uinf']['J_fwd']

        dCT_dUinf_fd = self.power_gradients['CT', 'Uinf']['J_fd']
        dCQ_dUinf_fd = self.power_gradients['CQ', 'Uinf']['J_fd']
        dCP_dUinf_fd = self.power_gradients['CP', 'Uinf']['J_fd']

        np.testing.assert_allclose(dCT_dUinf_fd, dCT_dUinf, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dUinf_fd, dCQ_dUinf, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dUinf_fd, dCP_dUinf, rtol=5e-5, atol=1e-8)


    def test_dOmega1(self):

        dNp_dOmega = self.loads_gradients['Np', 'Omega']['J_fwd']
        dTp_dOmega = self.loads_gradients['Tp', 'Omega']['J_fwd']

        dNp_dOmega_fd = self.loads_gradients['Np', 'Omega']['J_fd']
        dTp_dOmega_fd = self.loads_gradients['Tp', 'Omega']['J_fd']

        np.testing.assert_allclose(dNp_dOmega_fd, dNp_dOmega, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dOmega_fd, dTp_dOmega, rtol=1e-5, atol=1e-6)


    def test_dOmega2(self):

        dT_dOmega = self.power_gradients['T', 'Omega']['J_fwd']
        dQ_dOmega = self.power_gradients['Q', 'Omega']['J_fwd']
        dP_dOmega = self.power_gradients['P', 'Omega']['J_fwd']

        dT_dOmega_fd = self.power_gradients['T', 'Omega']['J_fd']
        dQ_dOmega_fd = self.power_gradients['Q', 'Omega']['J_fd']
        dP_dOmega_fd = self.power_gradients['P', 'Omega']['J_fd']

        np.testing.assert_allclose(dT_dOmega_fd, dT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dOmega_fd, dQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dOmega_fd, dP_dOmega, rtol=5e-5, atol=1e-8)



    def test_dOmega3(self):

        dCT_dOmega = self.power_gradients['CT', 'Omega']['J_fwd']
        dCQ_dOmega = self.power_gradients['CQ', 'Omega']['J_fwd']
        dCP_dOmega = self.power_gradients['CP', 'Omega']['J_fwd']

        dCT_dOmega_fd = self.power_gradients['CT', 'Omega']['J_fd']
        dCQ_dOmega_fd = self.power_gradients['CQ', 'Omega']['J_fd']
        dCP_dOmega_fd = self.power_gradients['CP', 'Omega']['J_fd']

        np.testing.assert_allclose(dCT_dOmega_fd, dCT_dOmega, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dOmega_fd, dCQ_dOmega, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dOmega_fd, dCP_dOmega, rtol=5e-5, atol=1e-8)



    def test_dpitch1(self):

        dNp_dpitch = self.loads_gradients['Np', 'pitch']['J_fwd']
        dTp_dpitch = self.loads_gradients['Tp', 'pitch']['J_fwd']

        dNp_dpitch_fd = self.loads_gradients['Np', 'pitch']['J_fd']
        dTp_dpitch_fd = self.loads_gradients['Tp', 'pitch']['J_fd']

        np.testing.assert_allclose(dNp_dpitch_fd, dNp_dpitch, rtol=5e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dpitch_fd, dTp_dpitch, rtol=5e-5, atol=1e-6)


    def test_dpitch2(self):

        dT_dpitch = self.power_gradients['T', 'pitch']['J_fwd']
        dQ_dpitch = self.power_gradients['Q', 'pitch']['J_fwd']
        dP_dpitch = self.power_gradients['P', 'pitch']['J_fwd']

        dT_dpitch_fd = self.power_gradients['T', 'pitch']['J_fd']
        dQ_dpitch_fd = self.power_gradients['Q', 'pitch']['J_fd']
        dP_dpitch_fd = self.power_gradients['P', 'pitch']['J_fd']

        np.testing.assert_allclose(dT_dpitch_fd, dT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dQ_dpitch_fd, dQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dP_dpitch_fd, dP_dpitch, rtol=5e-5, atol=1e-8)



    def test_dpitch3(self):

        dCT_dpitch = self.power_gradients['CT', 'pitch']['J_fwd']
        dCQ_dpitch = self.power_gradients['CQ', 'pitch']['J_fwd']
        dCP_dpitch = self.power_gradients['CP', 'pitch']['J_fwd']

        dCT_dpitch_fd = self.power_gradients['CT', 'pitch']['J_fd']
        dCQ_dpitch_fd = self.power_gradients['CQ', 'pitch']['J_fd']
        dCP_dpitch_fd = self.power_gradients['CP', 'pitch']['J_fd']

        np.testing.assert_allclose(dCT_dpitch_fd, dCT_dpitch, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpitch_fd, dCQ_dpitch, rtol=5e-5, atol=1e-8)
        np.testing.assert_allclose(dCP_dpitch_fd, dCP_dpitch, rtol=5e-5, atol=1e-8)



    def test_dprecurve1(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        dNp_dprecurve = self.loads_gradients['Np', 'precurve']['J_fwd']
        dTp_dprecurve = self.loads_gradients['Tp', 'precurve']['J_fwd']

        dNp_dprecurve_fd = self.loads_gradients['Np', 'precurve']['J_fwd']
        dTp_dprecurve_fd = self.loads_gradients['Tp', 'precurve']['J_fwd']

        np.testing.assert_allclose(dNp_dprecurve_fd, dNp_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurve_fd, dTp_dprecurve, rtol=3e-4, atol=1e-8)


    def test_dprecurve2(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # P, T, Q, dP, dT, dQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)

        dT_dprecurve = self.power_gradients['T', 'precurve']['J_fwd']
        dQ_dprecurve = self.power_gradients['Q', 'precurve']['J_fwd']
        dP_dprecurve = self.power_gradients['P', 'precurve']['J_fwd']

        dT_dprecurve_fd = self.power_gradients['T', 'precurve']['J_fd']
        dQ_dprecurve_fd = self.power_gradients['Q', 'precurve']['J_fd']
        dP_dprecurve_fd = self.power_gradients['P', 'precurve']['J_fd']

        np.testing.assert_allclose(dT_dprecurve_fd, dT_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecurve_fd, dQ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dprecurve_fd, dP_dprecurve, rtol=3e-4, atol=1e-8)


    def test_dprecurve3(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # CP, CT, CQ, dCP, dCT, dCQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)

        dCT_dprecurve = self.power_gradients['CT', 'precurve']['J_fwd']
        dCQ_dprecurve = self.power_gradients['CQ', 'precurve']['J_fwd']
        dCP_dprecurve = self.power_gradients['CP', 'precurve']['J_fwd']


        dCT_dprecurve_fd = self.power_gradients['CT', 'precurve']['J_fd']
        dCQ_dprecurve_fd = self.power_gradients['CQ', 'precurve']['J_fd']
        dCP_dprecurve_fd = self.power_gradients['CP', 'precurve']['J_fd']

        np.testing.assert_allclose(dCT_dprecurve_fd, dCT_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecurve_fd, dCQ_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecurve_fd, dCP_dprecurve, rtol=3e-4, atol=1e-8)


    def test_dpresweep1(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        dNp_dpresweep = self.loads_gradients['Np', 'presweep']['J_fwd']
        dTp_dpresweep = self.loads_gradients['Tp', 'presweep']['J_fwd']

        dNp_dpresweep_fd = self.loads_gradients['Np', 'presweep']['J_fwd']
        dTp_dpresweep_fd = self.loads_gradients['Tp', 'presweep']['J_fwd']

        np.testing.assert_allclose(dNp_dpresweep_fd, dNp_dpresweep, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweep_fd, dTp_dpresweep, rtol=1e-5, atol=1e-8)


    def test_dpresweep2(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # P, T, Q, dP, dT, dQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)

        dT_dpresweep = self.power_gradients['T', 'presweep']['J_fwd']
        dQ_dpresweep = self.power_gradients['Q', 'presweep']['J_fwd']
        dP_dpresweep = self.power_gradients['P', 'presweep']['J_fwd']


        dT_dpresweep_fd = self.power_gradients['T', 'presweep']['J_fd']
        dQ_dpresweep_fd = self.power_gradients['Q', 'presweep']['J_fd']
        dP_dpresweep_fd = self.power_gradients['P', 'presweep']['J_fd']

        np.testing.assert_allclose(dT_dpresweep_fd, dT_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dpresweep_fd, dQ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dpresweep_fd, dP_dpresweep, rtol=3e-4, atol=1e-8)




    def test_dpresweep3(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # CP, CT, CQ, dCP, dCT, dCQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)

        dCT_dpresweep = self.power_gradients['CT', 'presweep']['J_fwd']
        dCQ_dpresweep = self.power_gradients['CQ', 'presweep']['J_fwd']
        dCP_dpresweep = self.power_gradients['CP', 'presweep']['J_fwd']

        dCT_dpresweep_fd = self.power_gradients['CT', 'presweep']['J_fd']
        dCQ_dpresweep_fd = self.power_gradients['CQ', 'presweep']['J_fd']
        dCP_dpresweep_fd = self.power_gradients['CP', 'presweep']['J_fd']

        np.testing.assert_allclose(dCT_dpresweep_fd, dCT_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpresweep_fd, dCQ_dpresweep, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dpresweep_fd, dCP_dpresweep, rtol=3e-4, atol=1e-8)



    def test_dprecurveTip1(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        dNp_dprecurveTip_fd = self.loads_gradients['Np', 'precurveTip']['J_fd']
        dTp_dprecurveTip_fd = self.loads_gradients['Tp', 'precurveTip']['J_fd']


        np.testing.assert_allclose(dNp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)


    def test_dprecurveTip2(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # P, T, Q, dP, dT, dQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)

        dT_dprecurveTip = self.power_gradients['T', 'precurveTip']['J_fwd']
        dQ_dprecurveTip = self.power_gradients['Q', 'precurveTip']['J_fwd']
        dP_dprecurveTip = self.power_gradients['P', 'precurveTip']['J_fwd']

        dT_dprecurveTip_fd = self.power_gradients['T', 'precurveTip']['J_fd']
        dQ_dprecurveTip_fd = self.power_gradients['Q', 'precurveTip']['J_fd']
        dP_dprecurveTip_fd = self.power_gradients['P', 'precurveTip']['J_fd']

        np.testing.assert_allclose(dT_dprecurveTip_fd, dT_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dprecurveTip_fd, dQ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dprecurveTip_fd, dP_dprecurveTip, rtol=1e-4, atol=1e-8)



    def test_dprecurveTip3(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # CP, CT, CQ, dCP, dCT, dCQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)

        dCT_dprecurveTip = self.power_gradients['CT', 'precurveTip']['J_fwd']
        dCQ_dprecurveTip = self.power_gradients['CQ', 'precurveTip']['J_fwd']
        dCP_dprecurveTip = self.power_gradients['CP', 'precurveTip']['J_fwd']

        dCT_dprecurveTip_fd = self.power_gradients['CT', 'precurveTip']['J_fd']
        dCQ_dprecurveTip_fd = self.power_gradients['CQ', 'precurveTip']['J_fd']
        dCP_dprecurveTip_fd = self.power_gradients['CP', 'precurveTip']['J_fd']

        np.testing.assert_allclose(dCT_dprecurveTip_fd, dCT_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dprecurveTip_fd, dCQ_dprecurveTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dprecurveTip_fd, dCP_dprecurveTip, rtol=1e-4, atol=1e-8)


    def test_dpresweepTip1(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        dNp_dpresweepTip_fd = self.loads_gradients['Np', 'presweepTip']['J_fd']
        dTp_dpresweepTip_fd = self.loads_gradients['Tp', 'presweepTip']['J_fd']

        np.testing.assert_allclose(dNp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)


    def test_dpresweepTip2(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # P, T, Q, dP, dT, dQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)

        dT_dpresweepTip = self.power_gradients['T', 'presweepTip']['J_fwd']
        dQ_dpresweepTip = self.power_gradients['Q', 'presweepTip']['J_fwd']
        dP_dpresweepTip = self.power_gradients['P', 'presweepTip']['J_fwd']

        dT_dpresweepTip_fd = self.power_gradients['T', 'presweepTip']['J_fd']
        dQ_dpresweepTip_fd = self.power_gradients['Q', 'presweepTip']['J_fd']
        dP_dpresweepTip_fd = self.power_gradients['P', 'presweepTip']['J_fd']

        np.testing.assert_allclose(dT_dpresweepTip_fd, dT_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dQ_dpresweepTip_fd, dQ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dP_dpresweepTip_fd, dP_dpresweepTip, rtol=1e-4, atol=1e-8)



    def test_dpresweepTip3(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # CP, CT, CQ, dCP, dCT, dCQ \
        #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)

        dCT_dpresweepTip = self.power_gradients['CT', 'presweepTip']['J_fwd']
        dCQ_dpresweepTip = self.power_gradients['CQ', 'presweepTip']['J_fwd']
        dCP_dpresweepTip = self.power_gradients['CP', 'presweepTip']['J_fwd']

        dCT_dpresweepTip_fd = self.power_gradients['CT', 'presweepTip']['J_fd']
        dCQ_dpresweepTip_fd = self.power_gradients['CQ', 'presweepTip']['J_fd']
        dCP_dpresweepTip_fd = self.power_gradients['CP', 'presweepTip']['J_fd']

        np.testing.assert_allclose(dCT_dpresweepTip_fd, dCT_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCQ_dpresweepTip_fd, dCQ_dpresweepTip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dCP_dpresweepTip_fd, dCP_dpresweepTip, rtol=1e-4, atol=1e-8)



class TestGradientsNotRotating(TestGradientsClass):

    @classmethod
    def setUpClass(cls):
        super(TestGradientsNotRotating, cls).setUpClass()

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

        # atmosphere
        rho = 1.225
        mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
        basepath = path.join(path.dirname(path.realpath(__file__)), '5MW_AFFiles') + path.sep

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
        pitch = 0.0
        Omega = 0.0  # convert to RPM
        azimuth = 90.

        n = len(r)
        bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)

        ## Load gradients
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
        # loads['nSector'] = nSector
        loads['Uinf'] = Uinf
        loads['Omega'] = Omega
        loads['pitch'] = np.radians(pitch)
        loads['azimuth'] = np.radians(azimuth)
        loads['af'] = af
        loads['bemoptions'] = bemoptions

        loads.run()
        loads_test_total_gradients = open('loads_test_total_gradients.txt', 'w')
        print "Generating gradients for Test 2. Please wait."
        loads_gradients = loads.check_total_derivatives(out_stream=loads_test_total_gradients, unknown_list=['Np', 'Tp'])
        print "Gradients generated for Test 2."
        cls.loads_gradients = loads_gradients
        cls.n = len(r)
        cls.npts = 1  # len(Uinf)

    def test_dr1(self):

        dNp_dr = self.loads_gradients['Np', 'r']['J_fwd']
        dTp_dr = self.loads_gradients['Tp', 'r']['J_fwd']
        dNp_dr_fd = self.loads_gradients['Np', 'r']['J_fd']
        dTp_dr_fd = self.loads_gradients['Tp', 'r']['J_fd']

        np.testing.assert_allclose(dNp_dr_fd, dNp_dr, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dr_fd, dTp_dr, rtol=1e-4, atol=1e-8)


    def test_dchord1(self):

        dNp_dchord = self.loads_gradients['Np', 'chord']['J_fwd']
        dTp_dchord = self.loads_gradients['Tp', 'chord']['J_fwd']
        dNp_dchord_fd = self.loads_gradients['Np', 'chord']['J_fd']
        dTp_dchord_fd = self.loads_gradients['Tp', 'chord']['J_fd']

        np.testing.assert_allclose(dNp_dchord_fd, dNp_dchord, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dchord_fd, dTp_dchord, rtol=5e-5, atol=1e-8)


    def test_dtheta1(self):

        dNp_dtheta = self.loads_gradients['Np', 'theta']['J_fwd']
        dTp_dtheta = self.loads_gradients['Tp', 'theta']['J_fwd']
        dNp_dtheta_fd = self.loads_gradients['Np', 'theta']['J_fd']
        dTp_dtheta_fd = self.loads_gradients['Tp', 'theta']['J_fd']

        np.testing.assert_allclose(dNp_dtheta_fd, dNp_dtheta, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dTp_dtheta_fd, dTp_dtheta, rtol=1e-4, atol=1e-6)


    def test_dRhub1(self):

        dNp_dRhub = self.loads_gradients['Np', 'Rhub']['J_fwd']
        dTp_dRhub = self.loads_gradients['Tp', 'Rhub']['J_fwd']

        dNp_dRhub_fd = self.loads_gradients['Np', 'Rhub']['J_fd']
        dTp_dRhub_fd = self.loads_gradients['Tp', 'Rhub']['J_fd']

        np.testing.assert_allclose(dNp_dRhub_fd, dNp_dRhub, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(dTp_dRhub_fd, dTp_dRhub, rtol=1e-4, atol=1e-7)


    def test_dRtip1(self):

        dNp_dRtip = self.loads_gradients['Np', 'Rtip']['J_fwd']
        dTp_dRtip = self.loads_gradients['Tp', 'Rtip']['J_fwd']

        dNp_dRtip_fd = self.loads_gradients['Np', 'Rtip']['J_fd']
        dTp_dRtip_fd = self.loads_gradients['Tp', 'Rtip']['J_fd']

        np.testing.assert_allclose(dNp_dRtip_fd, dNp_dRtip, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dRtip_fd, dTp_dRtip, rtol=1e-4, atol=1e-8)


    def test_dprecone1(self):

        dNp_dprecone = self.loads_gradients['Np', 'precone']['J_fwd']
        dTp_dprecone = self.loads_gradients['Tp', 'precone']['J_fwd']

        dNp_dprecone_fd = self.loads_gradients['Np', 'precone']['J_fd']
        dTp_dprecone_fd = self.loads_gradients['Tp', 'precone']['J_fd']

        np.testing.assert_allclose(dNp_dprecone_fd, dNp_dprecone, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecone_fd, dTp_dprecone, rtol=1e-6, atol=1e-8)


    def test_dtilt1(self):

        dNp_dtilt = self.loads_gradients['Np', 'tilt']['J_fwd']
        dTp_dtilt = self.loads_gradients['Tp', 'tilt']['J_fwd']

        dNp_dtilt_fd = self.loads_gradients['Np', 'tilt']['J_fd']
        dTp_dtilt_fd = self.loads_gradients['Tp', 'tilt']['J_fd']

        np.testing.assert_allclose(dNp_dtilt_fd, dNp_dtilt, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dTp_dtilt_fd, dTp_dtilt, rtol=1e-5, atol=1e-6)


    def test_dhubht1(self):

        dNp_dhubht = self.loads_gradients['Np', 'hubHt']['J_fwd']
        dTp_dhubht = self.loads_gradients['Tp', 'hubHt']['J_fwd']

        dNp_dhubht_fd = self.loads_gradients['Np', 'hubHt']['J_fd']
        dTp_dhubht_fd = self.loads_gradients['Tp', 'hubHt']['J_fd']

        np.testing.assert_allclose(dNp_dhubht_fd, dNp_dhubht, rtol=1e-4, atol=1e-6) #TODO: rtol=1e-5, atol=1e-8
        np.testing.assert_allclose(dTp_dhubht_fd, dTp_dhubht, rtol=1e-4, atol=1e-6)


    def test_dyaw1(self):

        dNp_dyaw = self.loads_gradients['Np', 'yaw']['J_fwd']
        dTp_dyaw = self.loads_gradients['Tp', 'yaw']['J_fwd']

        dNp_dyaw_fd = self.loads_gradients['Np', 'yaw']['J_fd']
        dTp_dyaw_fd = self.loads_gradients['Tp', 'yaw']['J_fd']

        np.testing.assert_allclose(dNp_dyaw_fd, dNp_dyaw, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dyaw_fd, dTp_dyaw, rtol=1e-5, atol=1e-8)



    def test_dazimuth1(self):

        dNp_dazimuth = self.loads_gradients['Np', 'azimuth']['J_fwd']
        dTp_dazimuth = self.loads_gradients['Tp', 'azimuth']['J_fwd']

        dNp_dazimuth_fd = self.loads_gradients['Np', 'azimuth']['J_fd']
        dTp_dazimuth_fd = self.loads_gradients['Tp', 'azimuth']['J_fd']

        np.testing.assert_allclose(dNp_dazimuth_fd, dNp_dazimuth, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dazimuth_fd, dTp_dazimuth, rtol=1e-5, atol=1e-6)


    def test_dUinf1(self):

        dNp_dUinf = self.loads_gradients['Np', 'Uinf']['J_fwd']
        dTp_dUinf = self.loads_gradients['Tp', 'Uinf']['J_fwd']

        dNp_dUinf_fd = self.loads_gradients['Np', 'Uinf']['J_fd']
        dTp_dUinf_fd = self.loads_gradients['Tp', 'Uinf']['J_fd']

        np.testing.assert_allclose(dNp_dUinf_fd, dNp_dUinf, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dUinf_fd, dTp_dUinf, rtol=1e-5, atol=1e-6)


    #
    # Omega is fixed at 0 so no need to run derivatives test
    #


    def test_dpitch1(self):

        dNp_dpitch = self.loads_gradients['Np', 'pitch']['J_fwd']
        dTp_dpitch = self.loads_gradients['Tp', 'pitch']['J_fwd']

        dNp_dpitch_fd = self.loads_gradients['Np', 'pitch']['J_fd']
        dTp_dpitch_fd = self.loads_gradients['Tp', 'pitch']['J_fd']

        np.testing.assert_allclose(dNp_dpitch_fd, dNp_dpitch, rtol=5e-5, atol=1e-6)
        np.testing.assert_allclose(dTp_dpitch_fd, dTp_dpitch, rtol=5e-5, atol=1e-6)



    def test_dprecurve1(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)


        dNp_dprecurve = self.loads_gradients['Np', 'precurve']['J_fwd']
        dTp_dprecurve = self.loads_gradients['Tp', 'precurve']['J_fwd']

        dNp_dprecurve_fd = self.loads_gradients['Np', 'precurve']['J_fd']
        dTp_dprecurve_fd = self.loads_gradients['Tp', 'precurve']['J_fd']

        np.testing.assert_allclose(dNp_dprecurve_fd, dNp_dprecurve, rtol=3e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurve_fd, dTp_dprecurve, rtol=3e-4, atol=1e-8)

    def test_dpresweep1(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        dNp_dpresweep = self.loads_gradients['Np', 'presweep']['J_fwd']
        dTp_dpresweep = self.loads_gradients['Tp', 'presweep']['J_fd']

        dNp_dpresweep_fd = self.loads_gradients['Np', 'presweep']['J_fd']
        dTp_dpresweep_fd = self.loads_gradients['Tp', 'presweep']['J_fd']

        np.testing.assert_allclose(dNp_dpresweep_fd, dNp_dpresweep, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweep_fd, dTp_dpresweep, rtol=1e-5, atol=1e-8)


    def test_dprecurveTip1(self):

        # precurve = np.linspace(1, 10, self.n)
        # precurveTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        dNp_dprecurveTip_fd = self.loads_gradients['Np', 'precurveTip']['J_fd']
        dTp_dprecurveTip_fd = self.loads_gradients['Tp', 'precurveTip']['J_fd']

        np.testing.assert_allclose(dNp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)


    def test_dpresweepTip1(self):

        # presweep = np.linspace(1, 10, self.n)
        # presweepTip = 10.1
        # precone = 0.0
        # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
        #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
        #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
        #
        # Np, Tp, dNp, dTp \
        #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)

        dNp_dpresweepTip_fd = self.loads_gradients['Np', 'presweepTip']['J_fd']
        dTp_dpresweepTip_fd = self.loads_gradients['Tp', 'presweepTip']['J_fd']

        np.testing.assert_allclose(dNp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(dTp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)



class TestGradientsFreestreamArray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestGradientsFreestreamArray, cls).setUpClass()

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

        # atmosphere
        rho = 1.225
        mu = 1.81206e-5

        afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
        basepath = path.join(path.dirname(path.realpath(__file__)), '5MW_AFFiles') + path.sep

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
        Uinf = np.array([10.0, 11.0, 12.0])
        tsr = 7.55
        pitch = np.zeros(3)
        Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM

        bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)
        n = len(r)

        ## Power Gradients
        ccblade = Problem()
        root = ccblade.root = CCBlade(nSector, n)
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
        ccblade['af'] = af
        ccblade['bemoptions'] = bemoptions

        power_gradients = [0]*len(Uinf)

        for i in range(len(Uinf)):
            ccblade['Uinf'] = Uinf[i]
            ccblade['Omega'] = Omega[i]
            ccblade['pitch'] = np.radians(pitch[i])

            ccblade.run()

            power_test_total_gradients = open('power_test_total_gradients.txt', 'w')
            print "Generating gradients for Test " + str(i+3) + ". Please wait..."
            power_gradients_sub = ccblade.check_total_derivatives(out_stream=power_test_total_gradients, unknown_list=['CP', 'CT', 'CQ', 'P', 'T', 'Q'])
            print "Gradients " + str(i+3) + " calculated."
            power_gradients[i] = power_gradients_sub

        cls.power_gradients = power_gradients
        cls.n = len(r)
        cls.npts = len(Uinf)

    def test_dUinf2(self):

        for i in range(self.npts):
            dT_dUinf = self.power_gradients[i]['T', 'Uinf']['J_fwd']
            dQ_dUinf = self.power_gradients[i]['Q', 'Uinf']['J_fwd']
            dP_dUinf = self.power_gradients[i]['P', 'Uinf']['J_fwd']

            dT_dUinf_fd = self.power_gradients[i]['T', 'Uinf']['J_fd']
            dQ_dUinf_fd = self.power_gradients[i]['Q', 'Uinf']['J_fd']
            dP_dUinf_fd = self.power_gradients[i]['P', 'Uinf']['J_fd']

            np.testing.assert_allclose(dT_dUinf_fd, dT_dUinf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(dQ_dUinf_fd, dQ_dUinf, rtol=5e-5, atol=1e-8)
            np.testing.assert_allclose(dP_dUinf_fd, dP_dUinf, rtol=5e-5, atol=1e-8)



    def test_dUinf3(self):

       for i in range(self.npts):
            dCT_dUinf = self.power_gradients[i]['CT', 'Uinf']['J_fwd']
            dCQ_dUinf = self.power_gradients[i]['CQ', 'Uinf']['J_fwd']
            dCP_dUinf = self.power_gradients[i]['CP', 'Uinf']['J_fwd']

            dCT_dUinf_fd = self.power_gradients[i]['CT', 'Uinf']['J_fd']
            dCQ_dUinf_fd = self.power_gradients[i]['CQ', 'Uinf']['J_fd']
            dCP_dUinf_fd = self.power_gradients[i]['CP', 'Uinf']['J_fd']

            np.testing.assert_allclose(dCT_dUinf_fd, dCT_dUinf, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(dCQ_dUinf_fd, dCQ_dUinf, rtol=5e-5, atol=1e-8)
            np.testing.assert_allclose(dCP_dUinf_fd, dCP_dUinf, rtol=5e-5, atol=1e-8)


    def test_dOmega2(self):

        for i in range(self.npts):
            dT_dOmega = self.power_gradients[i]['T', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
            dQ_dOmega = self.power_gradients[i]['Q', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
            dP_dOmega = self.power_gradients[i]['P', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1

            dT_dOmega_fd = self.power_gradients[i]['T', 'Uinf']['J_fd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
            dQ_dOmega_fd = self.power_gradients[i]['Q', 'Uinf']['J_fd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
            dP_dOmega_fd = self.power_gradients[i]['P', 'Uinf']['J_fd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1

            np.testing.assert_allclose(dT_dOmega_fd, dT_dOmega, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(dQ_dOmega_fd, dQ_dOmega, rtol=5e-5, atol=1e-8)
            np.testing.assert_allclose(dP_dOmega_fd, dP_dOmega, rtol=5e-5, atol=1e-8)



    def test_dOmega3(self):

        for i in range(self.npts):
            dCT_dOmega = self.power_gradients[i]['CT', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
            dCQ_dOmega = self.power_gradients[i]['CQ', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
            dCP_dOmega = self.power_gradients[i]['CP', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1

            dCT_dOmega_fd = self.power_gradients[i]['CT', 'Uinf']['J_fd']* self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
            dCQ_dOmega_fd = self.power_gradients[i]['CQ', 'Uinf']['J_fd']* self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
            dCP_dOmega_fd = self.power_gradients[i]['CP', 'Uinf']['J_fd']* self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1

            np.testing.assert_allclose(dCT_dOmega_fd, dCT_dOmega, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(dCQ_dOmega_fd, dCQ_dOmega, rtol=5e-5, atol=1e-8)
            np.testing.assert_allclose(dCP_dOmega_fd, dCP_dOmega, rtol=5e-5, atol=1e-8)


    def test_dpitch2(self):

        for i in range(self.npts):
            dT_dpitch = self.power_gradients[i]['T', 'pitch']['J_fwd']
            dQ_dpitch = self.power_gradients[i]['Q', 'pitch']['J_fwd']
            dP_dpitch = self.power_gradients[i]['P', 'pitch']['J_fwd']

            dT_dpitch_fd = self.power_gradients[i]['T', 'pitch']['J_fd']
            dQ_dpitch_fd = self.power_gradients[i]['Q', 'pitch']['J_fd']
            dP_dpitch_fd = self.power_gradients[i]['P', 'pitch']['J_fd']

            np.testing.assert_allclose(dT_dpitch_fd, dT_dpitch, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(dQ_dpitch_fd, dQ_dpitch, rtol=5e-5, atol=1e-8)
            np.testing.assert_allclose(dP_dpitch_fd, dP_dpitch, rtol=5e-5, atol=1e-8)



    def test_dpitch3(self):

        for i in range(self.npts):
            dCT_dpitch = self.power_gradients[i]['CT', 'pitch']['J_fwd']
            dCQ_dpitch = self.power_gradients[i]['CQ', 'pitch']['J_fwd']
            dCP_dpitch = self.power_gradients[i]['CP', 'pitch']['J_fwd']

            dCT_dpitch_fd = self.power_gradients[i]['CT', 'pitch']['J_fd']
            dCQ_dpitch_fd = self.power_gradients[i]['CQ', 'pitch']['J_fd']
            dCP_dpitch_fd = self.power_gradients[i]['CP', 'pitch']['J_fd']

            np.testing.assert_allclose(dCT_dpitch_fd, dCT_dpitch, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(dCQ_dpitch_fd, dCQ_dpitch, rtol=5e-5, atol=1e-8)
            np.testing.assert_allclose(dCP_dpitch_fd, dCP_dpitch, rtol=5e-5, atol=1e-8)


if __name__ == '__main__':
    unittest.main()