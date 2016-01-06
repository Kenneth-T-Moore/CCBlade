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
from openmdao.api import IndepVarComp, Problem, Group
from utilities import check_gradient_unit_test #, check_for_missing_unit_tests

from ccblade2 import CCAirfoil, CCBlade, LoadsGroup, BrentGroup, AirfoilComp, FlowCondition, DistributedAeroLoads, WindComponents, CCInit, CCEvaluate

# class TestGradientsClass(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#        pass
#
#
# class TestGradientsPower_Loads(TestGradientsClass):
#
#     @classmethod
#     def setUpClass(self):
#         super(TestGradientsPower_Loads, self).setUpClass()
#         # geometry
#         Rhub = 1.5
#         Rtip = 63.0
#
#         r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
#                       28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
#                       56.1667, 58.9000, 61.6333])
#         chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
#                           3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
#         theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
#                           6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
#         B = 3  # number of blades
#
#         # atmosphere
#         rho = 1.225
#         mu = 1.81206e-5
#
#         afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
#         basepath = path.join(path.dirname(path.realpath(__file__)), '5MW_AFFiles') + path.sep
#
#         # load all airfoils
#         airfoil_types = [0]*8
#         airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
#         airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
#         airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
#         airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
#         airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
#         airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
#         airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
#         airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')
#
#         # place at appropriate radial stations
#         af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
#
#         af = [0]*len(r)
#         for i in range(len(r)):
#             af[i] = airfoil_types[af_idx[i]]
#
#
#         tilt = -5.0
#         precone = 2.5
#         yaw = 0.0
#         shearExp = 0.2
#         hubHt = 80.0
#         nSector = 8
#
#         # set conditions
#         Uinf = 10.0
#         tsr = 7.55
#         pitch = 0.0
#         Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
#         azimuth = 90.0
#
#         bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)
#         n = len(r)
#
#         ## Load gradients
#         loads = Problem()
#         root = loads.root = LoadsGroup(n)
#         loads.setup(check=False)
#
#         loads['Rhub'] = Rhub
#         loads['Rtip'] = Rtip
#         loads['r'] = r
#         loads['chord'] = chord
#         loads['theta'] = np.radians(theta)
#         loads['rho'] = rho
#         loads['mu'] = mu
#         loads['tilt'] = np.radians(tilt)
#         loads['precone'] = np.radians(precone)
#         loads['yaw'] = np.radians(yaw)
#         loads['shearExp'] = shearExp
#         loads['hubHt'] = hubHt
#         loads['Uinf'] = Uinf
#         loads['Omega'] = Omega
#         loads['pitch'] = np.radians(pitch)
#         loads['azimuth'] = np.radians(azimuth)
#         loads['af'] = af
#         loads['bemoptions'] = bemoptions
#
#         loads.run()
#         loads_test_total_gradients = open('loads_test_total_gradients.txt', 'w')
#         loads_gradients = loads.check_total_derivatives(out_stream=loads_test_total_gradients, unknown_list=['Np', 'Tp'])
#         # loads_partials = loads.check_partial_derivatives(out_stream=loads_test_total_gradients)
#
#         ## Power Gradients
#         ccblade = Problem()
#         root = ccblade.root = CCBlade(nSector, n)
#         ccblade.setup(check=False)
#         ccblade['Rhub'] = Rhub
#         ccblade['Rtip'] = Rtip
#         ccblade['r'] = r
#         ccblade['chord'] = chord
#         ccblade['theta'] = np.radians(theta)
#         ccblade['B'] = B
#         ccblade['rho'] = rho
#         ccblade['mu'] = mu
#         ccblade['tilt'] = np.radians(tilt)
#         ccblade['precone'] = np.radians(precone)
#         ccblade['yaw'] = np.radians(yaw)
#         ccblade['shearExp'] = shearExp
#         ccblade['hubHt'] = hubHt
#         ccblade['nSector'] = nSector
#         ccblade['Uinf'] = Uinf
#         ccblade['Omega'] = Omega
#         ccblade['pitch'] = np.radians(pitch)
#         ccblade['af'] = af
#         ccblade['bemoptions'] = bemoptions
#
#         ccblade.run()
#
#         print "Generating gradients for Test 1. Please wait..."
#         power_test_total_gradients = open('power_test_total_gradients.txt', 'w')
#         # power_gradients = ccblade.check_total_derivatives(out_stream=power_test_total_gradients, unknown_list=['CP', 'CT', 'CQ', 'P', 'T', 'Q'])
#         # power_partial = ccblade.check_partial_derivatives(out_stream=power_test_total_gradients)
#         print "Gradients generated for Test 1."
#
#         self.loads_gradients = loads_gradients
#         # self.power_gradients = power_gradients
#         self.n = len(r)
#         self.npts = 1  # len(Uinf)
#
#     def test_dr1(self):
#
#         dNp_dr = self.loads_gradients['Np', 'r']['J_fwd']
#         dTp_dr = self.loads_gradients['Tp', 'r']['J_fwd']
#         dNp_dr_fd = self.loads_gradients['Np', 'r']['J_fd']
#         dTp_dr_fd = self.loads_gradients['Tp', 'r']['J_fd']
#
#         np.testing.assert_allclose(dNp_dr_fd, dNp_dr, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dr_fd, dTp_dr, rtol=1e-4, atol=1e-8)
#
#
#     def test_dr2(self):
#
#         dT_dr = self.power_gradients['T', 'r']['J_fwd']
#         dQ_dr = self.power_gradients['Q', 'r']['J_fwd']
#         dP_dr = self.power_gradients['P', 'r']['J_fwd']
#         dT_dr_fd = self.power_gradients['T', 'r']['J_fd']
#         dQ_dr_fd = self.power_gradients['Q', 'r']['J_fd']
#         dP_dr_fd = self.power_gradients['P', 'r']['J_fd']
#
#
#         np.testing.assert_allclose(dT_dr_fd, dT_dr, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dr_fd, dQ_dr, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dP_dr_fd, dP_dr, rtol=3e-4, atol=1e-8)
#
#
#     def test_dr3(self):
#
#         dCT_dr = self.power_gradients['CT', 'r']['J_fwd']
#         dCQ_dr = self.power_gradients['CQ', 'r']['J_fwd']
#         dCP_dr = self.power_gradients['CP', 'r']['J_fwd']
#         dCT_dr_fd = self.power_gradients['CT', 'r']['J_fd']
#         dCQ_dr_fd = self.power_gradients['CQ', 'r']['J_fd']
#         dCP_dr_fd = self.power_gradients['CP', 'r']['J_fd']
#
#         np.testing.assert_allclose(dCT_dr_fd, dCT_dr, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dr_fd, dCQ_dr, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dCP_dr_fd, dCP_dr, rtol=3e-4, atol=1e-8)
#
#
#
#     def test_dchord1(self):
#
#         dNp_dchord = self.loads_gradients['Np', 'chord']['J_fwd']
#         dTp_dchord = self.loads_gradients['Tp', 'chord']['J_fwd']
#         dNp_dchord_fd = self.loads_gradients['Np', 'chord']['J_fd']
#         dTp_dchord_fd = self.loads_gradients['Tp', 'chord']['J_fd']
#
#         np.testing.assert_allclose(dNp_dchord_fd, dNp_dchord, rtol=1e-6, atol=1e-8)
#         np.testing.assert_allclose(dTp_dchord_fd, dTp_dchord, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dchord2(self):
#
#         dT_dchord = self.power_gradients['T', 'chord']['J_fwd']
#         dQ_dchord = self.power_gradients['Q', 'chord']['J_fwd']
#         dP_dchord = self.power_gradients['P', 'chord']['J_fwd']
#         dT_dchord_fd = self.power_gradients['T', 'chord']['J_fd']
#         dQ_dchord_fd = self.power_gradients['Q', 'chord']['J_fd']
#         dP_dchord_fd = self.power_gradients['P', 'chord']['J_fd']
#
#         np.testing.assert_allclose(dT_dchord_fd, dT_dchord, rtol=5e-6, atol=1e-8)
#         np.testing.assert_allclose(dQ_dchord_fd, dQ_dchord, rtol=7e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dchord_fd, dP_dchord, rtol=7e-5, atol=1e-8)
#
#     def test_dchord3(self):
#
#         dCT_dchord = self.power_gradients['CT', 'chord']['J_fwd']
#         dCQ_dchord = self.power_gradients['CQ', 'chord']['J_fwd']
#         dCP_dchord = self.power_gradients['CP', 'chord']['J_fwd']
#         dCT_dchord_fd = self.power_gradients['CT', 'chord']['J_fd']
#         dCQ_dchord_fd = self.power_gradients['CQ', 'chord']['J_fd']
#         dCP_dchord_fd = self.power_gradients['CP', 'chord']['J_fd']
#
#         np.testing.assert_allclose(dCT_dchord_fd, dCT_dchord, rtol=5e-6, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dchord_fd, dCQ_dchord, rtol=7e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dchord_fd, dCP_dchord, rtol=7e-5, atol=1e-8)
#
#
#
#
#     def test_dtheta1(self):
#
#         dNp_dtheta = self.loads_gradients['Np', 'theta']['J_fwd']
#         dTp_dtheta = self.loads_gradients['Tp', 'theta']['J_fwd']
#         dNp_dtheta_fd = self.loads_gradients['Np', 'theta']['J_fwd']
#         dTp_dtheta_fd = self.loads_gradients['Tp', 'theta']['J_fwd']
#
#         np.testing.assert_allclose(dNp_dtheta_fd, dNp_dtheta, rtol=1e-6, atol=1e-8)
#         np.testing.assert_allclose(dTp_dtheta_fd, dTp_dtheta, rtol=1e-4, atol=1e-8)
#
#
#     def test_dtheta2(self):
#
#         dT_dtheta = self.power_gradients['T', 'theta']['J_fwd']
#         dQ_dtheta = self.power_gradients['Q', 'theta']['J_fwd']
#         dP_dtheta = self.power_gradients['P', 'theta']['J_fwd']
#         dT_dtheta_fd = self.power_gradients['T', 'theta']['J_fd']
#         dQ_dtheta_fd = self.power_gradients['Q', 'theta']['J_fd']
#         dP_dtheta_fd = self.power_gradients['P', 'theta']['J_fd']
#
#         np.testing.assert_allclose(dT_dtheta_fd, dT_dtheta, rtol=7e-4, atol=1e-6) # TODO: rtol=7e-5, atol=1e-8
#         np.testing.assert_allclose(dQ_dtheta_fd, dQ_dtheta, rtol=7e-4, atol=1e-6)
#         np.testing.assert_allclose(dP_dtheta_fd, dP_dtheta, rtol=7e-4, atol=1e-6)
#
#
#
#     def test_dtheta3(self):
#
#         dCT_dtheta = self.power_gradients['CT', 'theta']['J_fwd']
#         dCQ_dtheta = self.power_gradients['CQ', 'theta']['J_fwd']
#         dCP_dtheta = self.power_gradients['CP', 'theta']['J_fwd']
#         dCT_dtheta_fd = self.power_gradients['CT', 'theta']['J_fd']
#         dCQ_dtheta_fd = self.power_gradients['CQ', 'theta']['J_fd']
#         dCP_dtheta_fd = self.power_gradients['CP', 'theta']['J_fd']
#
#         np.testing.assert_allclose(dCT_dtheta_fd, dCT_dtheta, rtol=5e-6, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dtheta_fd, dCQ_dtheta, rtol=7e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dtheta_fd, dCP_dtheta, rtol=7e-5, atol=1e-8)
#
#
#
#     def test_dRhub1(self):
#
#         dNp_dRhub = self.loads_gradients['Np', 'Rhub']['J_fwd']
#         dTp_dRhub = self.loads_gradients['Tp', 'Rhub']['J_fwd']
#
#         dNp_dRhub_fd = self.loads_gradients['Np', 'Rhub']['J_fd']
#         dTp_dRhub_fd = self.loads_gradients['Tp', 'Rhub']['J_fd']
#
#         np.testing.assert_allclose(dNp_dRhub_fd, dNp_dRhub, rtol=1e-5, atol=1.5e-6) # TODO
#         np.testing.assert_allclose(dTp_dRhub_fd, dTp_dRhub, rtol=1e-4, atol=1.5e-6)
#
#
#     def test_dRhub2(self):
#
#         dT_dRhub = self.power_gradients['T', 'Rhub']['J_fwd']
#         dQ_dRhub = self.power_gradients['Q', 'Rhub']['J_fwd']
#         dP_dRhub = self.power_gradients['P', 'Rhub']['J_fwd']
#
#         dT_dRhub_fd = self.power_gradients['T', 'Rhub']['J_fd']
#         dQ_dRhub_fd = self.power_gradients['Q', 'Rhub']['J_fd']
#         dP_dRhub_fd = self.power_gradients['P', 'Rhub']['J_fd']
#
#         np.testing.assert_allclose(dT_dRhub_fd, dT_dRhub, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dRhub_fd, dQ_dRhub, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dRhub_fd, dP_dRhub, rtol=5e-5, atol=1e-8)
#
#
#     def test_dRhub3(self):
#
#         dCT_dRhub = self.power_gradients['CT', 'Rhub']['J_fwd']
#         dCQ_dRhub = self.power_gradients['CQ', 'Rhub']['J_fwd']
#         dCP_dRhub = self.power_gradients['CP', 'Rhub']['J_fwd']
#
#         dCT_dRhub_fd = self.power_gradients['CT', 'Rhub']['J_fd']
#         dCQ_dRhub_fd = self.power_gradients['CQ', 'Rhub']['J_fd']
#         dCP_dRhub_fd = self.power_gradients['CP', 'Rhub']['J_fd']
#
#         np.testing.assert_allclose(dCT_dRhub_fd, dCT_dRhub, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dRhub_fd, dCQ_dRhub, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dRhub_fd, dCP_dRhub, rtol=5e-5, atol=1e-8)
#
#
#     def test_dRtip1(self):
#
#         dNp_dRtip = self.loads_gradients['Np', 'Rtip']['J_fwd']
#         dTp_dRtip = self.loads_gradients['Tp', 'Rtip']['J_fwd']
#
#         dNp_dRtip_fd = self.loads_gradients['Np', 'Rtip']['J_fd']
#         dTp_dRtip_fd = self.loads_gradients['Tp', 'Rtip']['J_fd']
#
#         np.testing.assert_allclose(dNp_dRtip_fd, dNp_dRtip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dRtip_fd, dTp_dRtip, rtol=1e-4, atol=1e-8)
#
#
#     def test_dRtip2(self):
#
#         dT_dRtip = self.power_gradients['T', 'Rtip']['J_fwd']
#         dQ_dRtip = self.power_gradients['Q', 'Rtip']['J_fwd']
#         dP_dRtip = self.power_gradients['P', 'Rtip']['J_fwd']
#
#         dT_dRtip_fd = self.power_gradients['T', 'Rtip']['J_fd']
#         dQ_dRtip_fd = self.power_gradients['Q', 'Rtip']['J_fd']
#         dP_dRtip_fd = self.power_gradients['P', 'Rtip']['J_fd']
#
#         np.testing.assert_allclose(dT_dRtip_fd, dT_dRtip, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dRtip_fd, dQ_dRtip, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dRtip_fd, dP_dRtip, rtol=5e-5, atol=1e-8)
#
#
#     def test_dRtip3(self):
#
#         dCT_dRtip = self.power_gradients['CT', 'Rtip']['J_fwd']
#         dCQ_dRtip = self.power_gradients['CQ', 'Rtip']['J_fwd']
#         dCP_dRtip = self.power_gradients['CP', 'Rtip']['J_fwd']
#
#         dCT_dRtip_fd = self.power_gradients['CT', 'Rtip']['J_fd']
#         dCQ_dRtip_fd = self.power_gradients['CQ', 'Rtip']['J_fd']
#         dCP_dRtip_fd = self.power_gradients['CP', 'Rtip']['J_fd']
#
#         np.testing.assert_allclose(dCT_dRtip_fd, dCT_dRtip, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dRtip_fd, dCQ_dRtip, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dRtip_fd, dCP_dRtip, rtol=5e-5, atol=1e-8)
#
#
#     def test_dprecone1(self):
#
#         dNp_dprecone = self.loads_gradients['Np', 'precone']['J_fwd']
#         dTp_dprecone = self.loads_gradients['Tp', 'precone']['J_fwd']
#
#         dNp_dprecone_fd = self.loads_gradients['Np', 'precone']['J_fd']
#         dTp_dprecone_fd = self.loads_gradients['Tp', 'precone']['J_fd']
#
#         np.testing.assert_allclose(dNp_dprecone_fd, dNp_dprecone, rtol=1e-5, atol=1e-7)
#         np.testing.assert_allclose(dTp_dprecone_fd, dTp_dprecone, rtol=1e-5, atol=1e-7)
#
#
#
#     def test_dprecone2(self):
#
#         dT_dprecone = self.power_gradients['T', 'precone']['J_fwd']
#         dQ_dprecone = self.power_gradients['Q', 'precone']['J_fwd']
#         dP_dprecone = self.power_gradients['P', 'precone']['J_fwd']
#
#         dT_dprecone_fd = self.power_gradients['T', 'precone']['J_fd']
#         dQ_dprecone_fd = self.power_gradients['Q', 'precone']['J_fd']
#         dP_dprecone_fd = self.power_gradients['P', 'precone']['J_fd']
#
#         np.testing.assert_allclose(dT_dprecone_fd, dT_dprecone, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dprecone_fd, dQ_dprecone, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dprecone_fd, dP_dprecone, rtol=5e-5, atol=1e-8)
#
#
#     def test_dprecone3(self):
#
#         dCT_dprecone = self.power_gradients['CT', 'precone']['J_fwd']
#         dCQ_dprecone = self.power_gradients['CQ', 'precone']['J_fwd']
#         dCP_dprecone = self.power_gradients['CP', 'precone']['J_fwd']
#
#         dCT_dprecone_fd = self.power_gradients['CT', 'precone']['J_fd']
#         dCQ_dprecone_fd = self.power_gradients['CQ', 'precone']['J_fd']
#         dCP_dprecone_fd = self.power_gradients['CP', 'precone']['J_fd']
#
#         np.testing.assert_allclose(dCT_dprecone_fd, dCT_dprecone, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dprecone_fd, dCQ_dprecone, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dprecone_fd, dCP_dprecone, rtol=5e-5, atol=1e-8)
#
#
#     def test_dtilt1(self):
#
#         dNp_dtilt = self.loads_gradients['Np', 'tilt']['J_fwd']
#         dTp_dtilt = self.loads_gradients['Tp', 'tilt']['J_fwd']
#
#         dNp_dtilt_fd = self.loads_gradients['Np', 'tilt']['J_fd']
#         dTp_dtilt_fd = self.loads_gradients['Tp', 'tilt']['J_fd']
#
#         np.testing.assert_allclose(dNp_dtilt_fd, dNp_dtilt, rtol=1e-6, atol=1e-8)
#         np.testing.assert_allclose(dTp_dtilt_fd, dTp_dtilt, rtol=1e-5, atol=1e-8)
#
#
#     def test_dtilt2(self):
#
#         dT_dtilt = self.power_gradients['T', 'tilt']['J_fwd']
#         dQ_dtilt = self.power_gradients['Q', 'tilt']['J_fwd']
#         dP_dtilt = self.power_gradients['P', 'tilt']['J_fwd']
#
#         dT_dtilt_fd = self.power_gradients['T', 'tilt']['J_fd']
#         dQ_dtilt_fd = self.power_gradients['Q', 'tilt']['J_fd']
#         dP_dtilt_fd = self.power_gradients['P', 'tilt']['J_fd']
#
#         np.testing.assert_allclose(dT_dtilt_fd, dT_dtilt, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dtilt_fd, dQ_dtilt, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dtilt_fd, dP_dtilt, rtol=5e-5, atol=1e-8)
#
#
#     def test_dtilt3(self):
#
#         dCT_dtilt = self.power_gradients['CT', 'tilt']['J_fwd']
#         dCQ_dtilt = self.power_gradients['CQ', 'tilt']['J_fwd']
#         dCP_dtilt = self.power_gradients['CP', 'tilt']['J_fwd']
#
#         dCT_dtilt_fd = self.power_gradients['CT', 'tilt']['J_fd']
#         dCQ_dtilt_fd = self.power_gradients['CQ', 'tilt']['J_fd']
#         dCP_dtilt_fd = self.power_gradients['CP', 'tilt']['J_fd']
#
#         np.testing.assert_allclose(dCT_dtilt_fd, dCT_dtilt, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dtilt_fd, dCQ_dtilt, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dtilt_fd, dCP_dtilt, rtol=5e-5, atol=1e-8)
#
#
#     def test_dhubht1(self):
#
#         dNp_dhubht = self.loads_gradients['Np', 'hubHt']['J_fwd']
#         dTp_dhubht = self.loads_gradients['Tp', 'hubHt']['J_fwd']
#
#         dNp_dhubht_fd = self.loads_gradients['Np', 'hubHt']['J_fd']
#         dTp_dhubht_fd = self.loads_gradients['Tp', 'hubHt']['J_fd']
#
#         np.testing.assert_allclose(dNp_dhubht_fd, dNp_dhubht, rtol=1e-4, atol=1e-6) # TODO rtol = 1e-5 atol=1e-8
#         np.testing.assert_allclose(dTp_dhubht_fd, dTp_dhubht, rtol=1e-4, atol=1e-6)
#
#
#     def test_dhubht2(self):
#
#         dT_dhubht = self.power_gradients['T', 'hubHt']['J_fwd']
#         dQ_dhubht = self.power_gradients['Q', 'hubHt']['J_fwd']
#         dP_dhubht = self.power_gradients['P', 'hubHt']['J_fwd']
#
#         dT_dhubht_fd = self.power_gradients['T', 'hubHt']['J_fd']
#         dQ_dhubht_fd = self.power_gradients['Q', 'hubHt']['J_fd']
#         dP_dhubht_fd = self.power_gradients['P', 'hubHt']['J_fd']
#
#         np.testing.assert_allclose(dT_dhubht_fd, dT_dhubht, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dhubht_fd, dQ_dhubht, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dhubht_fd, dP_dhubht, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dhubht3(self):
#
#         dCT_dhubht = self.power_gradients['CT', 'hubHt']['J_fwd']
#         dCQ_dhubht = self.power_gradients['CQ', 'hubHt']['J_fwd']
#         dCP_dhubht = self.power_gradients['CP', 'hubHt']['J_fwd']
#
#         dCT_dhubht_fd = self.power_gradients['CT', 'hubHt']['J_fd']
#         dCQ_dhubht_fd = self.power_gradients['CQ', 'hubHt']['J_fd']
#         dCP_dhubht_fd = self.power_gradients['CP', 'hubHt']['J_fd']
#
#         np.testing.assert_allclose(dCT_dhubht_fd, dCT_dhubht, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dhubht_fd, dCQ_dhubht, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dhubht_fd, dCP_dhubht, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dyaw1(self):
#
#         dNp_dyaw = self.loads_gradients['Np', 'yaw']['J_fwd']
#         dTp_dyaw = self.loads_gradients['Tp', 'yaw']['J_fwd']
#
#         dNp_dyaw_fd = self.loads_gradients['Np', 'yaw']['J_fd']
#         dTp_dyaw_fd = self.loads_gradients['Tp', 'yaw']['J_fd']
#
#         np.testing.assert_allclose(dNp_dyaw_fd, dNp_dyaw, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dTp_dyaw_fd, dTp_dyaw, rtol=1e-5, atol=1e-8)
#
#
#     def test_dyaw2(self):
#
#         dT_dyaw = self.power_gradients['T', 'yaw']['J_fwd']
#         dQ_dyaw = self.power_gradients['Q', 'yaw']['J_fwd']
#         dP_dyaw = self.power_gradients['P', 'yaw']['J_fwd']
#
#         dT_dyaw_fd = self.power_gradients['T', 'yaw']['J_fd']
#         dQ_dyaw_fd = self.power_gradients['Q', 'yaw']['J_fd']
#         dP_dyaw_fd = self.power_gradients['P', 'yaw']['J_fd']
#
#         np.testing.assert_allclose(dT_dyaw_fd, dT_dyaw, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dyaw_fd, dQ_dyaw, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dyaw_fd, dP_dyaw, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dyaw3(self):
#
#         dCT_dyaw = self.power_gradients['CT', 'yaw']['J_fwd']
#         dCQ_dyaw = self.power_gradients['CQ', 'yaw']['J_fwd']
#         dCP_dyaw = self.power_gradients['CP', 'yaw']['J_fwd']
#
#         dCT_dyaw_fd = self.power_gradients['CT', 'yaw']['J_fd']
#         dCQ_dyaw_fd = self.power_gradients['CQ', 'yaw']['J_fd']
#         dCP_dyaw_fd = self.power_gradients['CP', 'yaw']['J_fd']
#
#         np.testing.assert_allclose(dCT_dyaw_fd, dCT_dyaw, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dyaw_fd, dCQ_dyaw, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dyaw_fd, dCP_dyaw, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dazimuth1(self):
#
#         dNp_dazimuth = self.loads_gradients['Np', 'azimuth']['J_fwd']
#         dTp_dazimuth = self.loads_gradients['Tp', 'azimuth']['J_fwd']
#
#         dNp_dazimuth_fd = self.loads_gradients['Np', 'azimuth']['J_fd']
#         dTp_dazimuth_fd = self.loads_gradients['Tp', 'azimuth']['J_fd']
#
#         np.testing.assert_allclose(dNp_dazimuth_fd, dNp_dazimuth, rtol=1e-5, atol=1e-6)
#         np.testing.assert_allclose(dTp_dazimuth_fd, dTp_dazimuth, rtol=1e-5, atol=1e-6)
#
#
#     def test_dUinf1(self):
#
#         dNp_dUinf = self.loads_gradients['Np', 'Uinf']['J_fwd']
#         dTp_dUinf = self.loads_gradients['Tp', 'Uinf']['J_fwd']
#
#         dNp_dUinf_fd = self.loads_gradients['Np', 'Uinf']['J_fd']
#         dTp_dUinf_fd = self.loads_gradients['Tp', 'Uinf']['J_fd']
#
#         np.testing.assert_allclose(dNp_dUinf_fd, dNp_dUinf, rtol=1e-5, atol=1e-6)
#         np.testing.assert_allclose(dTp_dUinf_fd, dTp_dUinf, rtol=1e-5, atol=1e-6)
#
#
#     def test_dUinf2(self):
#
#         dT_dUinf = self.power_gradients['T', 'Uinf']['J_fwd']
#         dQ_dUinf = self.power_gradients['Q', 'Uinf']['J_fwd']
#         dP_dUinf = self.power_gradients['P', 'Uinf']['J_fwd']
#
#         dT_dUinf_fd = self.power_gradients['T', 'Uinf']['J_fd']
#         dQ_dUinf_fd = self.power_gradients['Q', 'Uinf']['J_fd']
#         dP_dUinf_fd = self.power_gradients['P', 'Uinf']['J_fd']
#
#         np.testing.assert_allclose(dT_dUinf_fd, dT_dUinf, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dUinf_fd, dQ_dUinf, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dUinf_fd, dP_dUinf, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dUinf3(self):
#
#         dCT_dUinf = self.power_gradients['CT', 'Uinf']['J_fwd']
#         dCQ_dUinf = self.power_gradients['CQ', 'Uinf']['J_fwd']
#         dCP_dUinf = self.power_gradients['CP', 'Uinf']['J_fwd']
#
#         dCT_dUinf_fd = self.power_gradients['CT', 'Uinf']['J_fd']
#         dCQ_dUinf_fd = self.power_gradients['CQ', 'Uinf']['J_fd']
#         dCP_dUinf_fd = self.power_gradients['CP', 'Uinf']['J_fd']
#
#         np.testing.assert_allclose(dCT_dUinf_fd, dCT_dUinf, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dUinf_fd, dCQ_dUinf, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dUinf_fd, dCP_dUinf, rtol=5e-5, atol=1e-8)
#
#
#     def test_dOmega1(self):
#
#         dNp_dOmega = self.loads_gradients['Np', 'Omega']['J_fwd']
#         dTp_dOmega = self.loads_gradients['Tp', 'Omega']['J_fwd']
#
#         dNp_dOmega_fd = self.loads_gradients['Np', 'Omega']['J_fd']
#         dTp_dOmega_fd = self.loads_gradients['Tp', 'Omega']['J_fd']
#
#         np.testing.assert_allclose(dNp_dOmega_fd, dNp_dOmega, rtol=1e-5, atol=1e-6)
#         np.testing.assert_allclose(dTp_dOmega_fd, dTp_dOmega, rtol=1e-5, atol=1e-6)
#
#
#     def test_dOmega2(self):
#
#         dT_dOmega = self.power_gradients['T', 'Omega']['J_fwd']
#         dQ_dOmega = self.power_gradients['Q', 'Omega']['J_fwd']
#         dP_dOmega = self.power_gradients['P', 'Omega']['J_fwd']
#
#         dT_dOmega_fd = self.power_gradients['T', 'Omega']['J_fd']
#         dQ_dOmega_fd = self.power_gradients['Q', 'Omega']['J_fd']
#         dP_dOmega_fd = self.power_gradients['P', 'Omega']['J_fd']
#
#         np.testing.assert_allclose(dT_dOmega_fd, dT_dOmega, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dOmega_fd, dQ_dOmega, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dOmega_fd, dP_dOmega, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dOmega3(self):
#
#         dCT_dOmega = self.power_gradients['CT', 'Omega']['J_fwd']
#         dCQ_dOmega = self.power_gradients['CQ', 'Omega']['J_fwd']
#         dCP_dOmega = self.power_gradients['CP', 'Omega']['J_fwd']
#
#         dCT_dOmega_fd = self.power_gradients['CT', 'Omega']['J_fd']
#         dCQ_dOmega_fd = self.power_gradients['CQ', 'Omega']['J_fd']
#         dCP_dOmega_fd = self.power_gradients['CP', 'Omega']['J_fd']
#
#         np.testing.assert_allclose(dCT_dOmega_fd, dCT_dOmega, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dOmega_fd, dCQ_dOmega, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dOmega_fd, dCP_dOmega, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dpitch1(self):
#
#         dNp_dpitch = self.loads_gradients['Np', 'pitch']['J_fwd']
#         dTp_dpitch = self.loads_gradients['Tp', 'pitch']['J_fwd']
#
#         dNp_dpitch_fd = self.loads_gradients['Np', 'pitch']['J_fd']
#         dTp_dpitch_fd = self.loads_gradients['Tp', 'pitch']['J_fd']
#
#         np.testing.assert_allclose(dNp_dpitch_fd, dNp_dpitch, rtol=5e-5, atol=1e-6)
#         np.testing.assert_allclose(dTp_dpitch_fd, dTp_dpitch, rtol=5e-5, atol=1e-6)
#
#
#     def test_dpitch2(self):
#
#         dT_dpitch = self.power_gradients['T', 'pitch']['J_fwd']
#         dQ_dpitch = self.power_gradients['Q', 'pitch']['J_fwd']
#         dP_dpitch = self.power_gradients['P', 'pitch']['J_fwd']
#
#         dT_dpitch_fd = self.power_gradients['T', 'pitch']['J_fd']
#         dQ_dpitch_fd = self.power_gradients['Q', 'pitch']['J_fd']
#         dP_dpitch_fd = self.power_gradients['P', 'pitch']['J_fd']
#
#         np.testing.assert_allclose(dT_dpitch_fd, dT_dpitch, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dQ_dpitch_fd, dQ_dpitch, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dP_dpitch_fd, dP_dpitch, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dpitch3(self):
#
#         dCT_dpitch = self.power_gradients['CT', 'pitch']['J_fwd']
#         dCQ_dpitch = self.power_gradients['CQ', 'pitch']['J_fwd']
#         dCP_dpitch = self.power_gradients['CP', 'pitch']['J_fwd']
#
#         dCT_dpitch_fd = self.power_gradients['CT', 'pitch']['J_fd']
#         dCQ_dpitch_fd = self.power_gradients['CQ', 'pitch']['J_fd']
#         dCP_dpitch_fd = self.power_gradients['CP', 'pitch']['J_fd']
#
#         np.testing.assert_allclose(dCT_dpitch_fd, dCT_dpitch, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dpitch_fd, dCQ_dpitch, rtol=5e-5, atol=1e-8)
#         np.testing.assert_allclose(dCP_dpitch_fd, dCP_dpitch, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dprecurve1(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#         dNp_dprecurve = self.loads_gradients['Np', 'precurve']['J_fwd']
#         dTp_dprecurve = self.loads_gradients['Tp', 'precurve']['J_fwd']
#
#         dNp_dprecurve_fd = self.loads_gradients['Np', 'precurve']['J_fwd']
#         dTp_dprecurve_fd = self.loads_gradients['Tp', 'precurve']['J_fwd']
#
#         np.testing.assert_allclose(dNp_dprecurve_fd, dNp_dprecurve, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dprecurve_fd, dTp_dprecurve, rtol=3e-4, atol=1e-8)
#
#
#     def test_dprecurve2(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # P, T, Q, dP, dT, dQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)
#
#         dT_dprecurve = self.power_gradients['T', 'precurve']['J_fwd']
#         dQ_dprecurve = self.power_gradients['Q', 'precurve']['J_fwd']
#         dP_dprecurve = self.power_gradients['P', 'precurve']['J_fwd']
#
#         dT_dprecurve_fd = self.power_gradients['T', 'precurve']['J_fd']
#         dQ_dprecurve_fd = self.power_gradients['Q', 'precurve']['J_fd']
#         dP_dprecurve_fd = self.power_gradients['P', 'precurve']['J_fd']
#
#         np.testing.assert_allclose(dT_dprecurve_fd, dT_dprecurve, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dQ_dprecurve_fd, dQ_dprecurve, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dP_dprecurve_fd, dP_dprecurve, rtol=3e-4, atol=1e-8)
#
#
#     def test_dprecurve3(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # CP, CT, CQ, dCP, dCT, dCQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)
#
#         dCT_dprecurve = self.power_gradients['CT', 'precurve']['J_fwd']
#         dCQ_dprecurve = self.power_gradients['CQ', 'precurve']['J_fwd']
#         dCP_dprecurve = self.power_gradients['CP', 'precurve']['J_fwd']
#
#
#         dCT_dprecurve_fd = self.power_gradients['CT', 'precurve']['J_fd']
#         dCQ_dprecurve_fd = self.power_gradients['CQ', 'precurve']['J_fd']
#         dCP_dprecurve_fd = self.power_gradients['CP', 'precurve']['J_fd']
#
#         np.testing.assert_allclose(dCT_dprecurve_fd, dCT_dprecurve, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dprecurve_fd, dCQ_dprecurve, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dCP_dprecurve_fd, dCP_dprecurve, rtol=3e-4, atol=1e-8)
#
#
#     def test_dpresweep1(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#         dNp_dpresweep = self.loads_gradients['Np', 'presweep']['J_fwd']
#         dTp_dpresweep = self.loads_gradients['Tp', 'presweep']['J_fwd']
#
#         dNp_dpresweep_fd = self.loads_gradients['Np', 'presweep']['J_fwd']
#         dTp_dpresweep_fd = self.loads_gradients['Tp', 'presweep']['J_fwd']
#
#         np.testing.assert_allclose(dNp_dpresweep_fd, dNp_dpresweep, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dTp_dpresweep_fd, dTp_dpresweep, rtol=1e-5, atol=1e-8)
#
#
#     def test_dpresweep2(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # P, T, Q, dP, dT, dQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)
#
#         dT_dpresweep = self.power_gradients['T', 'presweep']['J_fwd']
#         dQ_dpresweep = self.power_gradients['Q', 'presweep']['J_fwd']
#         dP_dpresweep = self.power_gradients['P', 'presweep']['J_fwd']
#
#
#         dT_dpresweep_fd = self.power_gradients['T', 'presweep']['J_fd']
#         dQ_dpresweep_fd = self.power_gradients['Q', 'presweep']['J_fd']
#         dP_dpresweep_fd = self.power_gradients['P', 'presweep']['J_fd']
#
#         np.testing.assert_allclose(dT_dpresweep_fd, dT_dpresweep, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dQ_dpresweep_fd, dQ_dpresweep, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dP_dpresweep_fd, dP_dpresweep, rtol=3e-4, atol=1e-8)
#
#
#
#
#     def test_dpresweep3(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # CP, CT, CQ, dCP, dCT, dCQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)
#
#         dCT_dpresweep = self.power_gradients['CT', 'presweep']['J_fwd']
#         dCQ_dpresweep = self.power_gradients['CQ', 'presweep']['J_fwd']
#         dCP_dpresweep = self.power_gradients['CP', 'presweep']['J_fwd']
#
#         dCT_dpresweep_fd = self.power_gradients['CT', 'presweep']['J_fd']
#         dCQ_dpresweep_fd = self.power_gradients['CQ', 'presweep']['J_fd']
#         dCP_dpresweep_fd = self.power_gradients['CP', 'presweep']['J_fd']
#
#         np.testing.assert_allclose(dCT_dpresweep_fd, dCT_dpresweep, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dpresweep_fd, dCQ_dpresweep, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dCP_dpresweep_fd, dCP_dpresweep, rtol=3e-4, atol=1e-8)
#
#
#
#     def test_dprecurveTip1(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#         dNp_dprecurveTip_fd = self.loads_gradients['Np', 'precurveTip']['J_fd']
#         dTp_dprecurveTip_fd = self.loads_gradients['Tp', 'precurveTip']['J_fd']
#
#
#         np.testing.assert_allclose(dNp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#
#
#     def test_dprecurveTip2(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # P, T, Q, dP, dT, dQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)
#
#         dT_dprecurveTip = self.power_gradients['T', 'precurveTip']['J_fwd']
#         dQ_dprecurveTip = self.power_gradients['Q', 'precurveTip']['J_fwd']
#         dP_dprecurveTip = self.power_gradients['P', 'precurveTip']['J_fwd']
#
#         dT_dprecurveTip_fd = self.power_gradients['T', 'precurveTip']['J_fd']
#         dQ_dprecurveTip_fd = self.power_gradients['Q', 'precurveTip']['J_fd']
#         dP_dprecurveTip_fd = self.power_gradients['P', 'precurveTip']['J_fd']
#
#         np.testing.assert_allclose(dT_dprecurveTip_fd, dT_dprecurveTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dQ_dprecurveTip_fd, dQ_dprecurveTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dP_dprecurveTip_fd, dP_dprecurveTip, rtol=1e-4, atol=1e-8)
#
#
#
#     def test_dprecurveTip3(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # CP, CT, CQ, dCP, dCT, dCQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)
#
#         dCT_dprecurveTip = self.power_gradients['CT', 'precurveTip']['J_fwd']
#         dCQ_dprecurveTip = self.power_gradients['CQ', 'precurveTip']['J_fwd']
#         dCP_dprecurveTip = self.power_gradients['CP', 'precurveTip']['J_fwd']
#
#         dCT_dprecurveTip_fd = self.power_gradients['CT', 'precurveTip']['J_fd']
#         dCQ_dprecurveTip_fd = self.power_gradients['CQ', 'precurveTip']['J_fd']
#         dCP_dprecurveTip_fd = self.power_gradients['CP', 'precurveTip']['J_fd']
#
#         np.testing.assert_allclose(dCT_dprecurveTip_fd, dCT_dprecurveTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dprecurveTip_fd, dCQ_dprecurveTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dCP_dprecurveTip_fd, dCP_dprecurveTip, rtol=1e-4, atol=1e-8)
#
#
#     def test_dpresweepTip1(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#         dNp_dpresweepTip_fd = self.loads_gradients['Np', 'presweepTip']['J_fd']
#         dTp_dpresweepTip_fd = self.loads_gradients['Tp', 'presweepTip']['J_fd']
#
#         np.testing.assert_allclose(dNp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#
#
#     def test_dpresweepTip2(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # P, T, Q, dP, dT, dQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=False)
#
#         dT_dpresweepTip = self.power_gradients['T', 'presweepTip']['J_fwd']
#         dQ_dpresweepTip = self.power_gradients['Q', 'presweepTip']['J_fwd']
#         dP_dpresweepTip = self.power_gradients['P', 'presweepTip']['J_fwd']
#
#         dT_dpresweepTip_fd = self.power_gradients['T', 'presweepTip']['J_fd']
#         dQ_dpresweepTip_fd = self.power_gradients['Q', 'presweepTip']['J_fd']
#         dP_dpresweepTip_fd = self.power_gradients['P', 'presweepTip']['J_fd']
#
#         np.testing.assert_allclose(dT_dpresweepTip_fd, dT_dpresweepTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dQ_dpresweepTip_fd, dQ_dpresweepTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dP_dpresweepTip_fd, dP_dpresweepTip, rtol=1e-4, atol=1e-8)
#
#
#
#     def test_dpresweepTip3(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # CP, CT, CQ, dCP, dCT, dCQ \
#         #     = rotor.evaluate([self.Uinf], [self.Omega], [self.pitch], coefficient=True)
#
#         dCT_dpresweepTip = self.power_gradients['CT', 'presweepTip']['J_fwd']
#         dCQ_dpresweepTip = self.power_gradients['CQ', 'presweepTip']['J_fwd']
#         dCP_dpresweepTip = self.power_gradients['CP', 'presweepTip']['J_fwd']
#
#         dCT_dpresweepTip_fd = self.power_gradients['CT', 'presweepTip']['J_fd']
#         dCQ_dpresweepTip_fd = self.power_gradients['CQ', 'presweepTip']['J_fd']
#         dCP_dpresweepTip_fd = self.power_gradients['CP', 'presweepTip']['J_fd']
#
#         np.testing.assert_allclose(dCT_dpresweepTip_fd, dCT_dpresweepTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dCQ_dpresweepTip_fd, dCQ_dpresweepTip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dCP_dpresweepTip_fd, dCP_dpresweepTip, rtol=1e-4, atol=1e-8)



# class TestGradientsNotRotating(TestGradientsClass):
#
#     @classmethod
#     def setUpClass(cls):
#         super(TestGradientsNotRotating, cls).setUpClass()
#
#         # geometry
#         Rhub = 1.5
#         Rtip = 63.0
#
#         r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
#                       28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
#                       56.1667, 58.9000, 61.6333])
#         chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
#                           3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
#         theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
#                           6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
#         B = 3  # number of blades
#
#         # atmosphere
#         rho = 1.225
#         mu = 1.81206e-5
#
#         afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
#         basepath = path.join(path.dirname(path.realpath(__file__)), '5MW_AFFiles') + path.sep
#
#         # load all airfoils
#         airfoil_types = [0]*8
#         airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
#         airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
#         airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
#         airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
#         airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
#         airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
#         airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
#         airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')
#
#         # place at appropriate radial stations
#         af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
#
#         af = [0]*len(r)
#         for i in range(len(r)):
#             af[i] = airfoil_types[af_idx[i]]
#
#
#         tilt = -5.0
#         precone = 2.5
#         yaw = 0.0
#         shearExp = 0.2
#         hubHt = 80.0
#         nSector = 8
#
#         # set conditions
#         Uinf = 10.0
#         pitch = 0.0
#         Omega = 0.0  # convert to RPM
#         azimuth = 90.
#
#         n = len(r)
#         bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)
#
#         ## Load gradients
#         loads = Problem()
#         root = loads.root = LoadsGroup(n)
#
#         loads.setup(check=False)
#
#         loads['Rhub'] = Rhub
#         loads['Rtip'] = Rtip
#         loads['r'] = r
#         loads['chord'] = chord
#         loads['theta'] = np.radians(theta)
#         loads['rho'] = rho
#         loads['mu'] = mu
#         loads['tilt'] = np.radians(tilt)
#         loads['precone'] = np.radians(precone)
#         loads['yaw'] = np.radians(yaw)
#         loads['shearExp'] = shearExp
#         loads['hubHt'] = hubHt
#         # loads['nSector'] = nSector
#         loads['Uinf'] = Uinf
#         loads['Omega'] = Omega
#         loads['pitch'] = np.radians(pitch)
#         loads['azimuth'] = np.radians(azimuth)
#         loads['af'] = af
#         loads['bemoptions'] = bemoptions
#
#         loads.run()
#         loads_test_total_gradients = open('loads_test_total_gradients.txt', 'w')
#         print "Generating gradients for Test 2. Please wait."
#         loads_gradients = loads.check_total_derivatives(out_stream=loads_test_total_gradients, unknown_list=['Np', 'Tp'])
#         print "Gradients generated for Test 2."
#         cls.loads_gradients = loads_gradients
#         cls.n = len(r)
#         cls.npts = 1  # len(Uinf)
#
#     def test_dr1(self):
#
#         dNp_dr = self.loads_gradients['Np', 'r']['J_fwd']
#         dTp_dr = self.loads_gradients['Tp', 'r']['J_fwd']
#         dNp_dr_fd = self.loads_gradients['Np', 'r']['J_fd']
#         dTp_dr_fd = self.loads_gradients['Tp', 'r']['J_fd']
#
#         np.testing.assert_allclose(dNp_dr_fd, dNp_dr, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dr_fd, dTp_dr, rtol=1e-4, atol=1e-8)
#
#
#     def test_dchord1(self):
#
#         dNp_dchord = self.loads_gradients['Np', 'chord']['J_fwd']
#         dTp_dchord = self.loads_gradients['Tp', 'chord']['J_fwd']
#         dNp_dchord_fd = self.loads_gradients['Np', 'chord']['J_fd']
#         dTp_dchord_fd = self.loads_gradients['Tp', 'chord']['J_fd']
#
#         np.testing.assert_allclose(dNp_dchord_fd, dNp_dchord, rtol=1e-6, atol=1e-8)
#         np.testing.assert_allclose(dTp_dchord_fd, dTp_dchord, rtol=5e-5, atol=1e-8)
#
#
#     def test_dtheta1(self):
#
#         dNp_dtheta = self.loads_gradients['Np', 'theta']['J_fwd']
#         dTp_dtheta = self.loads_gradients['Tp', 'theta']['J_fwd']
#         dNp_dtheta_fd = self.loads_gradients['Np', 'theta']['J_fd']
#         dTp_dtheta_fd = self.loads_gradients['Tp', 'theta']['J_fd']
#
#         np.testing.assert_allclose(dNp_dtheta_fd, dNp_dtheta, rtol=1e-6, atol=1e-6)
#         np.testing.assert_allclose(dTp_dtheta_fd, dTp_dtheta, rtol=1e-4, atol=1e-6)
#
#
#     def test_dRhub1(self):
#
#         dNp_dRhub = self.loads_gradients['Np', 'Rhub']['J_fwd']
#         dTp_dRhub = self.loads_gradients['Tp', 'Rhub']['J_fwd']
#
#         dNp_dRhub_fd = self.loads_gradients['Np', 'Rhub']['J_fd']
#         dTp_dRhub_fd = self.loads_gradients['Tp', 'Rhub']['J_fd']
#
#         np.testing.assert_allclose(dNp_dRhub_fd, dNp_dRhub, rtol=1e-5, atol=1e-7)
#         np.testing.assert_allclose(dTp_dRhub_fd, dTp_dRhub, rtol=1e-4, atol=1e-7)
#
#
#     def test_dRtip1(self):
#
#         dNp_dRtip = self.loads_gradients['Np', 'Rtip']['J_fwd']
#         dTp_dRtip = self.loads_gradients['Tp', 'Rtip']['J_fwd']
#
#         dNp_dRtip_fd = self.loads_gradients['Np', 'Rtip']['J_fd']
#         dTp_dRtip_fd = self.loads_gradients['Tp', 'Rtip']['J_fd']
#
#         np.testing.assert_allclose(dNp_dRtip_fd, dNp_dRtip, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dRtip_fd, dTp_dRtip, rtol=1e-4, atol=1e-8)
#
#
#     def test_dprecone1(self):
#
#         dNp_dprecone = self.loads_gradients['Np', 'precone']['J_fwd']
#         dTp_dprecone = self.loads_gradients['Tp', 'precone']['J_fwd']
#
#         dNp_dprecone_fd = self.loads_gradients['Np', 'precone']['J_fd']
#         dTp_dprecone_fd = self.loads_gradients['Tp', 'precone']['J_fd']
#
#         np.testing.assert_allclose(dNp_dprecone_fd, dNp_dprecone, rtol=1e-6, atol=1e-8)
#         np.testing.assert_allclose(dTp_dprecone_fd, dTp_dprecone, rtol=1e-6, atol=1e-8)
#
#
#     def test_dtilt1(self):
#
#         dNp_dtilt = self.loads_gradients['Np', 'tilt']['J_fwd']
#         dTp_dtilt = self.loads_gradients['Tp', 'tilt']['J_fwd']
#
#         dNp_dtilt_fd = self.loads_gradients['Np', 'tilt']['J_fd']
#         dTp_dtilt_fd = self.loads_gradients['Tp', 'tilt']['J_fd']
#
#         np.testing.assert_allclose(dNp_dtilt_fd, dNp_dtilt, rtol=1e-6, atol=1e-6)
#         np.testing.assert_allclose(dTp_dtilt_fd, dTp_dtilt, rtol=1e-5, atol=1e-6)
#
#
#     def test_dhubht1(self):
#
#         dNp_dhubht = self.loads_gradients['Np', 'hubHt']['J_fwd']
#         dTp_dhubht = self.loads_gradients['Tp', 'hubHt']['J_fwd']
#
#         dNp_dhubht_fd = self.loads_gradients['Np', 'hubHt']['J_fd']
#         dTp_dhubht_fd = self.loads_gradients['Tp', 'hubHt']['J_fd']
#
#         np.testing.assert_allclose(dNp_dhubht_fd, dNp_dhubht, rtol=1e-4, atol=1e-6) #TODO: rtol=1e-5, atol=1e-8
#         np.testing.assert_allclose(dTp_dhubht_fd, dTp_dhubht, rtol=1e-4, atol=1e-6)
#
#
#     def test_dyaw1(self):
#
#         dNp_dyaw = self.loads_gradients['Np', 'yaw']['J_fwd']
#         dTp_dyaw = self.loads_gradients['Tp', 'yaw']['J_fwd']
#
#         dNp_dyaw_fd = self.loads_gradients['Np', 'yaw']['J_fd']
#         dTp_dyaw_fd = self.loads_gradients['Tp', 'yaw']['J_fd']
#
#         np.testing.assert_allclose(dNp_dyaw_fd, dNp_dyaw, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dTp_dyaw_fd, dTp_dyaw, rtol=1e-5, atol=1e-8)
#
#
#
#     def test_dazimuth1(self):
#
#         dNp_dazimuth = self.loads_gradients['Np', 'azimuth']['J_fwd']
#         dTp_dazimuth = self.loads_gradients['Tp', 'azimuth']['J_fwd']
#
#         dNp_dazimuth_fd = self.loads_gradients['Np', 'azimuth']['J_fd']
#         dTp_dazimuth_fd = self.loads_gradients['Tp', 'azimuth']['J_fd']
#
#         np.testing.assert_allclose(dNp_dazimuth_fd, dNp_dazimuth, rtol=1e-5, atol=1e-6)
#         np.testing.assert_allclose(dTp_dazimuth_fd, dTp_dazimuth, rtol=1e-5, atol=1e-6)
#
#
#     def test_dUinf1(self):
#
#         dNp_dUinf = self.loads_gradients['Np', 'Uinf']['J_fwd']
#         dTp_dUinf = self.loads_gradients['Tp', 'Uinf']['J_fwd']
#
#         dNp_dUinf_fd = self.loads_gradients['Np', 'Uinf']['J_fd']
#         dTp_dUinf_fd = self.loads_gradients['Tp', 'Uinf']['J_fd']
#
#         np.testing.assert_allclose(dNp_dUinf_fd, dNp_dUinf, rtol=1e-5, atol=1e-6)
#         np.testing.assert_allclose(dTp_dUinf_fd, dTp_dUinf, rtol=1e-5, atol=1e-6)
#
#
#     #
#     # Omega is fixed at 0 so no need to run derivatives test
#     #
#
#
#     def test_dpitch1(self):
#
#         dNp_dpitch = self.loads_gradients['Np', 'pitch']['J_fwd']
#         dTp_dpitch = self.loads_gradients['Tp', 'pitch']['J_fwd']
#
#         dNp_dpitch_fd = self.loads_gradients['Np', 'pitch']['J_fd']
#         dTp_dpitch_fd = self.loads_gradients['Tp', 'pitch']['J_fd']
#
#         np.testing.assert_allclose(dNp_dpitch_fd, dNp_dpitch, rtol=5e-5, atol=1e-6)
#         np.testing.assert_allclose(dTp_dpitch_fd, dTp_dpitch, rtol=5e-5, atol=1e-6)
#
#
#
#     def test_dprecurve1(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#
#         dNp_dprecurve = self.loads_gradients['Np', 'precurve']['J_fwd']
#         dTp_dprecurve = self.loads_gradients['Tp', 'precurve']['J_fwd']
#
#         dNp_dprecurve_fd = self.loads_gradients['Np', 'precurve']['J_fd']
#         dTp_dprecurve_fd = self.loads_gradients['Tp', 'precurve']['J_fd']
#
#         np.testing.assert_allclose(dNp_dprecurve_fd, dNp_dprecurve, rtol=3e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dprecurve_fd, dTp_dprecurve, rtol=3e-4, atol=1e-8)
#
#     def test_dpresweep1(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#         dNp_dpresweep = self.loads_gradients['Np', 'presweep']['J_fwd']
#         dTp_dpresweep = self.loads_gradients['Tp', 'presweep']['J_fd']
#
#         dNp_dpresweep_fd = self.loads_gradients['Np', 'presweep']['J_fd']
#         dTp_dpresweep_fd = self.loads_gradients['Tp', 'presweep']['J_fd']
#
#         np.testing.assert_allclose(dNp_dpresweep_fd, dNp_dpresweep, rtol=1e-5, atol=1e-8)
#         np.testing.assert_allclose(dTp_dpresweep_fd, dTp_dpresweep, rtol=1e-5, atol=1e-8)
#
#
#     def test_dprecurveTip1(self):
#
#         # precurve = np.linspace(1, 10, self.n)
#         # precurveTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, precurve=precurve, precurveTip=precurveTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#         dNp_dprecurveTip_fd = self.loads_gradients['Np', 'precurveTip']['J_fd']
#         dTp_dprecurveTip_fd = self.loads_gradients['Tp', 'precurveTip']['J_fd']
#
#         np.testing.assert_allclose(dNp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dprecurveTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#
#
#     def test_dpresweepTip1(self):
#
#         # presweep = np.linspace(1, 10, self.n)
#         # presweepTip = 10.1
#         # precone = 0.0
#         # rotor = CCBlade(self.r, self.chord, self.theta, self.af, self.Rhub, self.Rtip,
#         #     self.B, self.rho, self.mu, precone, self.tilt, self.yaw, self.shearExp,
#         #     self.hubHt, self.nSector, derivatives=True, presweep=presweep, presweepTip=presweepTip)
#         #
#         # Np, Tp, dNp, dTp \
#         #     = rotor.distributedAeroLoads(self.Uinf, self.Omega, self.pitch, self.azimuth)
#
#         dNp_dpresweepTip_fd = self.loads_gradients['Np', 'presweepTip']['J_fd']
#         dTp_dpresweepTip_fd = self.loads_gradients['Tp', 'presweepTip']['J_fd']
#
#         np.testing.assert_allclose(dNp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#         np.testing.assert_allclose(dTp_dpresweepTip_fd, 0.0, rtol=1e-4, atol=1e-8)
#

# @unittest.skip("Test takes a long time")
# class TestGradientsFreestreamArray(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         super(TestGradientsFreestreamArray, cls).setUpClass()
#
#         # geometry
#         Rhub = 1.5
#         Rtip = 63.0
#
#         r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
#                       28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
#                       56.1667, 58.9000, 61.6333])
#         chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
#                           3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
#         theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
#                           6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
#         B = 3  # number of blades
#
#         # atmosphere
#         rho = 1.225
#         mu = 1.81206e-5
#
#         afinit = CCAirfoil.initFromAerodynFile  # just for shorthand
#         basepath = path.join(path.dirname(path.realpath(__file__)), '5MW_AFFiles') + path.sep
#
#         # load all airfoils
#         airfoil_types = [0]*8
#         airfoil_types[0] = afinit(basepath + 'Cylinder1.dat')
#         airfoil_types[1] = afinit(basepath + 'Cylinder2.dat')
#         airfoil_types[2] = afinit(basepath + 'DU40_A17.dat')
#         airfoil_types[3] = afinit(basepath + 'DU35_A17.dat')
#         airfoil_types[4] = afinit(basepath + 'DU30_A17.dat')
#         airfoil_types[5] = afinit(basepath + 'DU25_A17.dat')
#         airfoil_types[6] = afinit(basepath + 'DU21_A17.dat')
#         airfoil_types[7] = afinit(basepath + 'NACA64_A17.dat')
#
#         # place at appropriate radial stations
#         af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
#
#         af = [0]*len(r)
#         for i in range(len(r)):
#             af[i] = airfoil_types[af_idx[i]]
#
#
#         tilt = -5.0
#         precone = 2.5
#         yaw = 0.0
#         shearExp = 0.2
#         hubHt = 80.0
#         nSector = 8
#
#
#         # set conditions
#         Uinf = np.array([10.0, 11.0, 12.0])
#         tsr = 7.55
#         pitch = np.zeros(3)
#         Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM
#
#         bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)
#         n = len(r)
#
#         ## Power Gradients
#         ccblade = Problem()
#         root = ccblade.root = CCBlade(nSector, n)
#         ccblade.setup(check=False)
#         ccblade['Rhub'] = Rhub
#         ccblade['Rtip'] = Rtip
#         ccblade['r'] = r
#         ccblade['chord'] = chord
#         ccblade['theta'] = np.radians(theta)
#         ccblade['B'] = B
#         ccblade['rho'] = rho
#         ccblade['mu'] = mu
#         ccblade['tilt'] = np.radians(tilt)
#         ccblade['precone'] = np.radians(precone)
#         ccblade['yaw'] = np.radians(yaw)
#         ccblade['shearExp'] = shearExp
#         ccblade['hubHt'] = hubHt
#         ccblade['nSector'] = nSector
#         ccblade['af'] = af
#         ccblade['bemoptions'] = bemoptions
#
#         power_gradients = [0]*len(Uinf)
#
#         for i in range(len(Uinf)):
#             ccblade['Uinf'] = Uinf[i]
#             ccblade['Omega'] = Omega[i]
#             ccblade['pitch'] = np.radians(pitch[i])
#
#             ccblade.run()
#
#             power_test_total_gradients = open('power_test_total_gradients.txt', 'w')
#             print "Generating gradients for Test " + str(i+3) + ". Please wait..."
#             power_gradients_sub = ccblade.check_total_derivatives(out_stream=power_test_total_gradients, unknown_list=['CP', 'CT', 'CQ', 'P', 'T', 'Q'])
#             print "Gradients " + str(i+3) + " calculated."
#             power_gradients[i] = power_gradients_sub
#
#         cls.power_gradients = power_gradients
#         cls.n = len(r)
#         cls.npts = len(Uinf)
#
#     def test_dUinf2(self):
#
#         for i in range(self.npts):
#             dT_dUinf = self.power_gradients[i]['T', 'Uinf']['J_fwd']
#             dQ_dUinf = self.power_gradients[i]['Q', 'Uinf']['J_fwd']
#             dP_dUinf = self.power_gradients[i]['P', 'Uinf']['J_fwd']
#
#             dT_dUinf_fd = self.power_gradients[i]['T', 'Uinf']['J_fd']
#             dQ_dUinf_fd = self.power_gradients[i]['Q', 'Uinf']['J_fd']
#             dP_dUinf_fd = self.power_gradients[i]['P', 'Uinf']['J_fd']
#
#             np.testing.assert_allclose(dT_dUinf_fd, dT_dUinf, rtol=1e-5, atol=1e-8)
#             np.testing.assert_allclose(dQ_dUinf_fd, dQ_dUinf, rtol=5e-5, atol=1e-8)
#             np.testing.assert_allclose(dP_dUinf_fd, dP_dUinf, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dUinf3(self):
#
#        for i in range(self.npts):
#             dCT_dUinf = self.power_gradients[i]['CT', 'Uinf']['J_fwd']
#             dCQ_dUinf = self.power_gradients[i]['CQ', 'Uinf']['J_fwd']
#             dCP_dUinf = self.power_gradients[i]['CP', 'Uinf']['J_fwd']
#
#             dCT_dUinf_fd = self.power_gradients[i]['CT', 'Uinf']['J_fd']
#             dCQ_dUinf_fd = self.power_gradients[i]['CQ', 'Uinf']['J_fd']
#             dCP_dUinf_fd = self.power_gradients[i]['CP', 'Uinf']['J_fd']
#
#             np.testing.assert_allclose(dCT_dUinf_fd, dCT_dUinf, rtol=1e-5, atol=1e-8)
#             np.testing.assert_allclose(dCQ_dUinf_fd, dCQ_dUinf, rtol=5e-5, atol=1e-8)
#             np.testing.assert_allclose(dCP_dUinf_fd, dCP_dUinf, rtol=5e-5, atol=1e-8)
#
#
#     def test_dOmega2(self):
#
#         for i in range(self.npts):
#             dT_dOmega = self.power_gradients[i]['T', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
#             dQ_dOmega = self.power_gradients[i]['Q', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
#             dP_dOmega = self.power_gradients[i]['P', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
#
#             dT_dOmega_fd = self.power_gradients[i]['T', 'Uinf']['J_fd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
#             dQ_dOmega_fd = self.power_gradients[i]['Q', 'Uinf']['J_fd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
#             dP_dOmega_fd = self.power_gradients[i]['P', 'Uinf']['J_fd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
#
#             np.testing.assert_allclose(dT_dOmega_fd, dT_dOmega, rtol=1e-5, atol=1e-8)
#             np.testing.assert_allclose(dQ_dOmega_fd, dQ_dOmega, rtol=5e-5, atol=1e-8)
#             np.testing.assert_allclose(dP_dOmega_fd, dP_dOmega, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dOmega3(self):
#
#         for i in range(self.npts):
#             dCT_dOmega = self.power_gradients[i]['CT', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
#             dCQ_dOmega = self.power_gradients[i]['CQ', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
#             dCP_dOmega = self.power_gradients[i]['CP', 'Uinf']['J_fwd'] * self.power_gradients[i]['Omega', 'Uinf']['J_fwd']**-1
#
#             dCT_dOmega_fd = self.power_gradients[i]['CT', 'Uinf']['J_fd']* self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
#             dCQ_dOmega_fd = self.power_gradients[i]['CQ', 'Uinf']['J_fd']* self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
#             dCP_dOmega_fd = self.power_gradients[i]['CP', 'Uinf']['J_fd']* self.power_gradients[i]['Omega', 'Uinf']['J_fd']**-1
#
#             np.testing.assert_allclose(dCT_dOmega_fd, dCT_dOmega, rtol=1e-5, atol=1e-8)
#             np.testing.assert_allclose(dCQ_dOmega_fd, dCQ_dOmega, rtol=5e-5, atol=1e-8)
#             np.testing.assert_allclose(dCP_dOmega_fd, dCP_dOmega, rtol=5e-5, atol=1e-8)
#
#
#     def test_dpitch2(self):
#
#         for i in range(self.npts):
#             dT_dpitch = self.power_gradients[i]['T', 'pitch']['J_fwd']
#             dQ_dpitch = self.power_gradients[i]['Q', 'pitch']['J_fwd']
#             dP_dpitch = self.power_gradients[i]['P', 'pitch']['J_fwd']
#
#             dT_dpitch_fd = self.power_gradients[i]['T', 'pitch']['J_fd']
#             dQ_dpitch_fd = self.power_gradients[i]['Q', 'pitch']['J_fd']
#             dP_dpitch_fd = self.power_gradients[i]['P', 'pitch']['J_fd']
#
#             np.testing.assert_allclose(dT_dpitch_fd, dT_dpitch, rtol=1e-5, atol=1e-8)
#             np.testing.assert_allclose(dQ_dpitch_fd, dQ_dpitch, rtol=5e-5, atol=1e-8)
#             np.testing.assert_allclose(dP_dpitch_fd, dP_dpitch, rtol=5e-5, atol=1e-8)
#
#
#
#     def test_dpitch3(self):
#
#         for i in range(self.npts):
#             dCT_dpitch = self.power_gradients[i]['CT', 'pitch']['J_fwd']
#             dCQ_dpitch = self.power_gradients[i]['CQ', 'pitch']['J_fwd']
#             dCP_dpitch = self.power_gradients[i]['CP', 'pitch']['J_fwd']
#
#             dCT_dpitch_fd = self.power_gradients[i]['CT', 'pitch']['J_fd']
#             dCQ_dpitch_fd = self.power_gradients[i]['CQ', 'pitch']['J_fd']
#             dCP_dpitch_fd = self.power_gradients[i]['CP', 'pitch']['J_fd']
#
#             np.testing.assert_allclose(dCT_dpitch_fd, dCT_dpitch, rtol=1e-5, atol=1e-8)
#             np.testing.assert_allclose(dCQ_dpitch_fd, dCQ_dpitch, rtol=5e-5, atol=1e-8)
#             np.testing.assert_allclose(dCP_dpitch_fd, dCP_dpitch, rtol=5e-5, atol=1e-8)


class TestBrentGroup(unittest.TestCase):

    def test1(self):
        Rhub = 1.5
        Rtip = 63.0
        rho = 1.225
        mu = 1.81206e-5
        tsr = 7.55
        Uinf = 10.0
        Omega = Uinf*tsr/Rtip * 30.0/pi
        B = 3
        pitch = 0.0
        bemoptions = dict(usecd=True, tiploss=True, hubloss=True, wakerotation=True)
        r = 2.8667
        chord = 3.542
        theta = 13.308
        Vx = 9.96219424
        Vy = 2.56068618

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
        n = len(af_idx)
        af = [0]*n
        for i in range(n):
            af[i] = airfoil_types[af_idx[i]]

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', BrentGroup(n, 0), promotes=['*'])
        prob.root.add('Rhub', IndepVarComp('Rhub', 0.0), promotes=['*'])
        prob.root.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        prob.root.add('rho', IndepVarComp('rho', 0.0), promotes=['*'])
        prob.root.add('mu', IndepVarComp('mu', 0.0), promotes=['*'])
        prob.root.add('Omega', IndepVarComp('Omega', 0.0), promotes=['*'])
        prob.root.add('B', IndepVarComp('B', 0, pass_by_obj=True), promotes=['*'])
        prob.root.add('pitch', IndepVarComp('pitch', 0.0), promotes=['*'])
        prob.root.add('bemoptions', IndepVarComp('bemoptions', {}, pass_by_obj=True), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', 0.0), promotes=['*'])
        prob.root.add('chord', IndepVarComp('chord', 0.0), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', 0.0), promotes=['*'])
        prob.root.add('Vx', IndepVarComp('Vx', 0.0), promotes=['*'])
        prob.root.add('Vy', IndepVarComp('Vy', 0.0), promotes=['*'])
        prob.root.add('af', IndepVarComp('af', np.zeros(len(af)), pass_by_obj=True), promotes=['*'])

        prob.setup(check=False)

        prob['Rhub'] = Rhub
        prob['Rtip'] = Rtip
        prob['rho'] = rho
        prob['mu'] = mu
        prob['Omega'] = Omega
        prob['B'] = B
        prob['pitch'] = pitch
        prob['bemoptions'] = bemoptions
        prob['r'] = r
        prob['chord'] = chord
        prob['theta'] = theta
        prob['Vx'] = Vx
        prob['Vy'] = Vy
        prob['af'] = af

        check_gradient_unit_test(self, prob)

class TestAirfoilComp(unittest.TestCase):

    def test1(self):

        alpha_sub = 1.08669181
        Re_sub = 2256848.32455109

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
        n = len(af_idx)
        af = [0]*n
        for i in range(n):
            af[i] = airfoil_types[af_idx[i]]

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', AirfoilComp(n, 0), promotes=['*'])
        prob.root.add('alpha_sub', IndepVarComp('alpha_sub', 0.0), promotes=['*'])
        prob.root.add('Re_sub', IndepVarComp('Re_sub', 0.0), promotes=['*'])
        prob.root.add('af', IndepVarComp('af', np.zeros(len(af)), pass_by_obj=True), promotes=['*'])
        prob.setup(check=False)

        prob['alpha_sub'] = alpha_sub
        prob['Re_sub'] = Re_sub
        prob['af'] = af

        check_gradient_unit_test(self, prob)

class TestFlowCondition(unittest.TestCase):

    def test1(self):

        pitch = 0.0
        Vx = 9.96219424
        Vy = 2.56068618
        chord = 3.542
        theta = 0.23226842
        rho = 1.225
        mu = 1.81206e-05
        a_sub = 0.08282631
        ap_sub = -0.08282631
        phi_sub = 1.31896023
        da_dx = np.array([-0.01430950725054113, 0.02144723621485688, 0.0, 0.0, 0.0, -0.041390638830529222, 0.028458955774978541, -2.0281001151246333e-16, 0.0])
        dap_dx = np.array([0.014309507250541129, -0.021447236214856873, 0.0, 0.0, 0.0, 0.041390638830529222, -0.028458955774978541, 2.0281001151246333e-16, 0.0])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', FlowCondition(), promotes=['*'])
        prob.root.add('pitch', IndepVarComp('pitch', 0.0), promotes=['*'])
        prob.root.add('Vx', IndepVarComp('Vx', 0.0), promotes=['*'])
        prob.root.add('Vy', IndepVarComp('Vy', 0.0), promotes=['*'])
        prob.root.add('chord', IndepVarComp('chord', 0.0), promotes=['*'])
        prob.root.add('theta', IndepVarComp('theta', 0.0), promotes=['*'])
        prob.root.add('rho', IndepVarComp('rho', 0.0), promotes=['*'])
        prob.root.add('mu', IndepVarComp('mu', 0.0), promotes=['*'])
        prob.root.add('a_sub', IndepVarComp('a_sub', 0.0), promotes=['*'])
        prob.root.add('ap_sub', IndepVarComp('ap_sub', 0.0), promotes=['*'])
        prob.root.add('phi_sub', IndepVarComp('phi_sub', 0.0), promotes=['*'])
        prob.root.add('da_dx', IndepVarComp('da_dx',  np.zeros(len(da_dx))), promotes=['*'])
        prob.root.add('dap_dx', IndepVarComp('dap_dx', np.zeros(len(dap_dx))), promotes=['*'])

        prob.setup(check=False)

        prob['pitch'] = pitch
        prob['Vx'] = Vx
        prob['Vy'] = Vy
        prob['chord'] = chord
        prob['theta'] = theta
        prob['rho'] = rho
        prob['mu'] = mu
        prob['a_sub'] = a_sub
        prob['ap_sub'] = ap_sub
        prob['mu'] = mu
        prob['phi_sub'] = phi_sub
        prob['da_dx'] = da_dx
        prob['dap_dx'] = dap_dx

        check_gradient_unit_test(self, prob, tol=0.0035)

class TestWindComponents(unittest.TestCase):

    def test1(self):

        Omega = 11.44399829
        Uinf = 10.0
        azimuth = 1.57079633
        hubHt = 80.0
        precone = 0.04363323
        precurve = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        presweep = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        r = np.array([  2.8667,   5.6,      8.3333,  11.75,    15.85,    19.95,    24.05,    28.15,  32.25,    36.35,   40.45,    44.55,    48.65,    52.75,    56.1667,  58.9,  61.6333])
        shearExp = 0.2
        tilt = -0.08726646
        yaw = 0.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', WindComponents(len(r)), promotes=['*'])
        prob.root.add('Omega', IndepVarComp('Omega', 0.0), promotes=['*'])
        prob.root.add('Uinf', IndepVarComp('Uinf', 0.0), promotes=['*'])
        prob.root.add('azimuth', IndepVarComp('azimuth', 0.0), promotes=['*'])
        prob.root.add('hubHt', IndepVarComp('hubHt', 0.0), promotes=['*'])
        prob.root.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        prob.root.add('precurve', IndepVarComp('precurve', np.zeros(len(precurve))), promotes=['*'])
        prob.root.add('presweep', IndepVarComp('presweep', np.zeros(len(presweep))), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', np.zeros(len(r))), promotes=['*'])
        prob.root.add('shearExp', IndepVarComp('shearExp', 0.0), promotes=['*'])
        prob.root.add('tilt', IndepVarComp('tilt', 0.0), promotes=['*'])
        prob.root.add('yaw', IndepVarComp('yaw',  0.0), promotes=['*'])

        prob.setup(check=False)

        prob['Omega'] = Omega
        prob['Uinf'] = Uinf
        prob['azimuth'] = azimuth
        prob['hubHt'] = hubHt
        prob['precone'] = precone
        prob['precurve'] = precurve
        prob['presweep'] = presweep
        prob['r'] = r
        prob['shearExp'] = shearExp
        prob['tilt'] = tilt
        prob['yaw'] = yaw

        check_gradient_unit_test(self, prob, tol=0.0035)

class TestDistributedAeroLoads(unittest.TestCase):

    def test1(self):

        W = np.array([  9.42519406,  11.00846879 , 13.1174378  , 16.05055251 , 20.43226424,  24.90720529,  29.51014394,  34.17267701,
                        38.90361415,  43.66643053,  48.45620991,  53.28256712,  58.11330631,  62.95218981,  66.98199402,  70.19798011,  73.43300001])
        cd = np.array([ 0.5,         0.5,         0.35 ,       0.17769315,  0.01383506,  0.01233991,  0.00999475,  0.00843288,  0.0083512 ,
                        0.0065993 ,  0.00662224 , 0.00745803,  0.00754494,  0.00768151,  0.00772987,  0.00761652,  0.00745758])
        cl = np.array([ 0.,          0.,          0.,          1.6180614,   1.42274313,  1.18581358,  1.0331037,   1.00659709,  0.96345427,
                        0.96634016,  0.96888061,  0.93728115,  0.94502499,  0.95730035,  0.9616861,   0.95144054,  0.93724093])
        chord = np.array([ 3.542,  3.854, 4.167,  4.557,  4.652,  4.458,  4.249 , 4.007 , 3.748,  3.502,  3.256,  3.01,   2.764,  2.518,  2.313,  2.086,  1.419])
        rho = 1.225
        phi = np.array([ 1.31896023 , 1.04063146,  0.82975796,  0.49131092,  0.36342359,  0.30239784,  0.25681637,  0.21390471,  0.18541446,
                         0.15916546,  0.13910259 , 0.12795197,  0.1152679,   0.10363831,  0.09286224,  0.0824041,   0.07525332])

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', DistributedAeroLoads(len(W)), promotes=['*'])
        prob.root.add('W', IndepVarComp('W', W), promotes=['*'])
        prob.root.add('cd', IndepVarComp('cd', cd), promotes=['*'])
        prob.root.add('cl', IndepVarComp('cl', cl), promotes=['*'])
        prob.root.add('chord', IndepVarComp('chord', chord), promotes=['*'])
        prob.root.add('rho', IndepVarComp('rho', rho), promotes=['*'])
        prob.root.add('phi', IndepVarComp('phi', phi), promotes=['*'])

        prob.setup(check=False)

        prob['W'] = W
        prob['cd'] = cd
        prob['cl'] = cl
        prob['chord'] = chord
        prob['rho'] = rho
        prob['phi'] = phi

        check_gradient_unit_test(self, prob, tol=0.0035)

class TestCCInit(unittest.TestCase):

    def test1(self):

        Rtip = 63.0
        precone = 0.04363323
        preconeTip = 0.0

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CCInit(), promotes=['*'])
        prob.root.add('Rtip', IndepVarComp('Rtip', 0.0), promotes=['*'])
        prob.root.add('precone', IndepVarComp('precone', 0.0), promotes=['*'])
        prob.root.add('preconeTip', IndepVarComp('preconeTip', 0.0), promotes=['*'])

        prob.setup(check=False)

        prob['Rtip'] = Rtip
        prob['precone'] = precone
        prob['preconeTip'] = preconeTip

        check_gradient_unit_test(self, prob)

class TestCCEvaluate(unittest.TestCase):

    def test1(self):

        B = 3
        Np1 = np.array([   95.89982364,   130.04495014,   121.09286797,  1162.40338348,  1700.11811006,  2083.30752979,  2496.13657417,
                           3126.94918549,  3679.25043209,  4376.89127782,  5060.75248883,  5465.79836719, 6030.03251659, 6527.41583821,  6818.18803354,  6685.60780673,  4859.22345479])
        Tp1 = np.array([ -32.96795744,  -86.77161447, -119.48767376,  471.61625639,  617.38129701,  628.76962403,  642.33378415,  677.37235801,
                         694.34818292,  721.93407083,  738.35271834,  736.60488868,  740.02442915,  727.76778445,  687.72007991,  602.95061897,  393.63429427])
        Np2 = np.array([   94.260791,     126.16695152,   116.79843598,  1124.4204556,   1658.62799184,  2033.00543723,  2429.25097166,
                           3031.0472091,   3559.6216928,   4229.07469978,  4885.77553823,  5278.75794514,  5825.0204098,  6309.37183261,  6594.00339583,  6466.8145806,   4704.33856868])
        Tp2 = np.array([ -26.58114852,  -76.5804659,  -108.51316272,  451.81144254,  619.60747492,  624.27992238,  631.82348756,  660.4691244,
                         672.22594013,  694.6792831,  706.4553837,   701.66786922,  701.93363714,  687.7203066,   647.62372399,  565.61004088, 369.00291134])
        Np3 = np.array([   93.32229007,   123.39949292,   113.40063354,  1086.14127088,  1587.71986792,  1923.76957663,  2270.38374612,
                           2824.33636367,  3295.44320143,  3906.60013816,  4497.38136228,  4870.710399,    5372.15187815,  5824.48437713,  6090.89256666,  5973.95750939,  4382.8021122 ])
        Tp3 = np.array([ -24.01169958,  -72.32919305, -103.76106124,  436.25204388,  586.2287836,  578.25319279,  572.81777421,  588.79213152,
                         588.60067544,  599.76401098,  598.35194999,  587.28255315,  578.56859627,  558.60757188,  517.90011638,  445.27845089,  295.39346567])
        Np4 = np.array([   93.42295416,   123.10088948,   112.61509052,  1066.81554928,  1560.24579343,  2726.55508675,  3053.51252904,
                           3185.20694418,  3752.25744507,  4237.55439743,  4798.71447431,  4714.40384466, 5082.40118219,  5252.68609074,  5280.12896269,  5245.29486703,  3920.14772729])
        Tp4 = np.array([ -26.5272039,  -75.94887158,-107.37377364, 432.74809308,540.76680005,452.95794556,462.75150201,486.47445449,445.6527727,
                       419.9856239,385.35033645,399.28223548,374.2340118, 356.21161331,329.58990294,283.5455035, 195.60122853])
        Np5 = np.array([   94.69437465, 125.59580744, 114.96721247,1074.56438563,2087.41205865,2861.7814876, 3259.09571829,3584.78237616,
                          4020.73596929,4392.62990694,4869.78962454,4835.97525832,5105.57848969,5143.95681068,4738.54695262,4733.61030357,3585.54170043])
        Tp5 = np.array([ -32.7702211, -85.51981633, -117.36600558,436.63701788,435.73407486,376.61946846,374.19190386,369.54223641,340.70628083,
                         317.69544651,280.84274797,276.60075459,247.45630519,228.33800463,224.47197142,187.97454726,133.16228975])
        Np6 = np.array([   96.58651707, 129.66885798, 119.35709625,1108.54686962,2222.4251395,3000.36683063,3392.02021684,3691.77338263,
                           4116.09369881,4505.23431032,5019.84536062,4954.42719909,5275.95699989,5394.65720028,5355.57655756,5330.13035627,3991.25198091])
        Tp6 = np.array([ -39.35507533,-96.01672398, -128.54401773,447.63082212,415.24295864,380.27986362,394.87038169,408.01980241,393.85607753,
                         383.06653345,357.07695903,370.84825007,352.91323353,341.7568785, 321.9606429,278.32367162,193.62899014])
        Np7 = np.array([   97.81192362,   132.798206, 123.14492161,1151.34816635,2284.7433429,3083.012189,3399.93665969,2870.53581265,
                        3352.02766137,3987.75731489,4593.50105953,4986.19328904,5501.55731121,5968.02007009,6241.416444,6131.52647211,4510.84384143])
        Tp7 = np.array([ -42.29805028, -101.09695295, -134.24513754,465.03657685,463.18014166,445.42430765,494.65599602,560.60844035,
                         563.56916094,577.73375245,576.61236771,568.98896182,561.32510692,542.6081426, 502.7306091,433.994482,290.99898975])
        Np8 = np.array([   97.45836722, 132.9015302,123.83520595,1174.71018333,2183.38598954,3006.55043097,2443.11086396,3073.01839531,
                          3611.00844498,4300.79218582,4971.24374755,5385.2191982, 5945.13123515,6442.64263986,6733.84320188,6601.30538028,4806.60224977])
        Tp8 = np.array([ -39.58532375,-97.17203936, -130.42874167,477.38402094,565.14384144,544.09673951,604.91183802,637.77649815,651.80496793,
                         677.26144644,689.63353433,687.70910151,689.3199001, 676.58056932,637.39925133,556.0421487, 364.33682965])
        Omega = 11.44399829
        Uinf = 10.0
        Rhub = 1.5
        Rtip = 63.0
        nSector = 8
        precone = 0.04363323
        precurve = np.zeros(len(Np1))
        precurveTip = 0.0
        presweep = np.zeros(len(Np1))
        presweepTip = 0.0
        r = np.array([  2.8667,   5.6 ,     8.3333 , 11.75,    15.85,    19.95,    24.05,    28.15,  32.25,    36.35,    40.45 ,   44.55 ,   48.65,    52.75,    56.1667,  58.9,  61.6333])
        rho = 1.225
        rotorR = 62.94003796

        prob = Problem()
        prob.root = Group()
        prob.root.add('comp', CCEvaluate(len(r), nSector), promotes=['*'])
        prob.root.add('B', IndepVarComp('B', B, pass_by_obj=True), promotes=['*'])
        prob.root.add('Np1', IndepVarComp('Np1', Np1), promotes=['*'])
        prob.root.add('Np2', IndepVarComp('Np2', Np2), promotes=['*'])
        prob.root.add('Np3', IndepVarComp('Np3', Np3), promotes=['*'])
        prob.root.add('Np4', IndepVarComp('Np4', Np4), promotes=['*'])
        prob.root.add('Np5', IndepVarComp('Np5', Np5), promotes=['*'])
        prob.root.add('Np6', IndepVarComp('Np6', Np6), promotes=['*'])
        prob.root.add('Np7', IndepVarComp('Np7', Np7), promotes=['*'])
        prob.root.add('Np8', IndepVarComp('Np8', Np8), promotes=['*'])
        prob.root.add('Tp1', IndepVarComp('Tp1', Tp1), promotes=['*'])
        prob.root.add('Tp2', IndepVarComp('Tp2', Tp2), promotes=['*'])
        prob.root.add('Tp3', IndepVarComp('Tp3', Tp3), promotes=['*'])
        prob.root.add('Tp4', IndepVarComp('Tp4', Tp4), promotes=['*'])
        prob.root.add('Tp5', IndepVarComp('Tp5', Tp5), promotes=['*'])
        prob.root.add('Tp6', IndepVarComp('Tp6', Tp6), promotes=['*'])
        prob.root.add('Tp7', IndepVarComp('Tp7', Tp7), promotes=['*'])
        prob.root.add('Tp8', IndepVarComp('Tp8', Tp8), promotes=['*'])
        prob.root.add('Omega', IndepVarComp('Omega', Omega), promotes=['*'])
        prob.root.add('Uinf', IndepVarComp('Uinf', Uinf), promotes=['*'])
        prob.root.add('Rhub', IndepVarComp('Rhub', Rhub), promotes=['*'])
        prob.root.add('Rtip', IndepVarComp('Rtip', Rtip), promotes=['*'])
        prob.root.add('nSector', IndepVarComp('nSector', nSector, pass_by_obj=True), promotes=['*'])
        prob.root.add('precone', IndepVarComp('precone', precone), promotes=['*'])
        prob.root.add('precurve', IndepVarComp('precurve', precurve), promotes=['*'])
        prob.root.add('precurveTip', IndepVarComp('precurveTip', precurveTip), promotes=['*'])
        prob.root.add('presweep', IndepVarComp('presweep', presweep), promotes=['*'])
        prob.root.add('presweepTip', IndepVarComp('presweepTip', presweepTip), promotes=['*'])
        prob.root.add('r', IndepVarComp('r', r), promotes=['*'])
        prob.root.add('rho', IndepVarComp('rho', rho), promotes=['*'])
        prob.root.add('rotorR', IndepVarComp('rotorR', rotorR), promotes=['*'])

        prob.setup(check=False)

        prob['B'] = B
        prob['Np1'] = Np1
        prob['Np2'] = Np2
        prob['Np3'] = Np3
        prob['Np4'] = Np4
        prob['Np5'] = Np5
        prob['Np6'] = Np6
        prob['Np7'] = Np7
        prob['Np8'] = Np8
        prob['Tp1'] = Tp1
        prob['Tp2'] = Tp2
        prob['Tp3'] = Tp3
        prob['Tp4'] = Tp4
        prob['Tp5'] = Tp5
        prob['Tp6'] = Tp6
        prob['Tp7'] = Tp7
        prob['Tp8'] = Tp8
        prob['Omega'] = Omega
        prob['Uinf'] = Uinf
        prob['Rhub'] = Rhub
        prob['Rtip'] = Rtip
        prob['nSector'] = nSector
        prob['precone'] = precone
        prob['precurve'] = precurve
        prob['precurveTip'] = precurveTip
        prob['presweep'] = presweep
        prob['presweepTip'] = presweepTip
        prob['r'] = r
        prob['rho'] = rho
        prob['rotorR'] = rotorR

        check_gradient_unit_test(self, prob, tol=1e-4)

if __name__ == '__main__':
    unittest.main()