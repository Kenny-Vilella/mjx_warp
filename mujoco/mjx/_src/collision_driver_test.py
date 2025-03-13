# Copyright 2025 The Physics-Next Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests the collision driver."""

import mujoco
from mujoco import mjx
import numpy as np
import warp as wp
from absl.testing import parameterized


class ConvexTest(parameterized.TestCase):
  """Tests the convex contact functions."""

  _BOX_PLANE = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.7" euler="45 0 0">
              <freejoint/>
              <geom size="0.5 0.5 0.5" type="box"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _FLAT_BOX_PLANE = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.45">
              <freejoint/>
              <geom size="0.5 0.5 0.5" type="box"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _BOX_BOX_EDGE = """
        <mujoco>
          <worldbody>
            <body pos="-1.0 -1.0 0.2">
              <joint axis="1 0 0" type="free"/>
              <geom size="0.2 0.2 0.2" type="box"/>
            </body>
            <body pos="-1.0 -1.2 0.55" euler="0 45 30">
              <joint axis="1 0 0" type="free"/>
              <geom size="0.1 0.1 0.1" type="box"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _CONVEX_CONVEX = """
        <mujoco>
          <asset>
            <mesh name="poly"
            vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
            face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
          </asset>
          <worldbody>
            <body pos="0.0 2.0 0.35" euler="0 0 90">
              <freejoint/>
              <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
            </body>
            <body pos="0.0 2.0 2.281" euler="180 0 0">
              <freejoint/>
              <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _CONVEX_CONVEX_MULTI = """
        <mujoco>
          <asset>
            <mesh name="poly"
            vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
            face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
          </asset>
          <worldbody>
            <body pos="0.0 2.0 0.35" euler="0 0 90">
              <freejoint/>
              <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
            </body>
            <body pos="0.0 2.0 2.281" euler="180 0 0">
              <freejoint/>
              <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
            </body>
            <body pos="0.0 2.0 2.281" euler="180 0 0">
              <freejoint/>
              <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _CAPSULE_CAPSULE = """
        <mujoco model="two_capsules">
          <worldbody>
            <body>
              <joint type="free"/>
              <geom fromto="0.62235904  0.58846647 0.651046 1.5330081 0.33564585 0.977849"
               size="0.05" type="capsule"/>
            </body>
            <body>
              <joint type="free"/>
              <geom fromto="0.5505271 0.60345304 0.476661 1.3900293 0.30709633 0.932082"
               size="0.05" type="capsule"/>
            </body>
          </worldbody>
        </mujoco>
        """

  _SPHERE_SPHERE = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom pos="0 0 0" size="0.2" type="sphere"/>
            </body>
            <body >
              <joint type="free"/>
              <geom pos="0 0.3 0" size="0.11" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """

  @parameterized.parameters(
    (_BOX_PLANE),
    (_FLAT_BOX_PLANE),
    (_BOX_BOX_EDGE),
    (_CONVEX_CONVEX),
    (_CONVEX_CONVEX_MULTI),
    (_SPHERE_SPHERE),
    (_CAPSULE_CAPSULE),
  )
  def test_convex_collision(self, xml_string):
    """Tests convex collision with different geometries."""
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    mjx.collision(mx, dx)
    mujoco.mj_step(m, d)
    actual_dist = dx.contact.dist.numpy()[0]
    actual_pos = dx.contact.pos.numpy()[0, :]
    actual_frame = dx.contact.frame.numpy()[0].flatten()
    np.testing.assert_array_almost_equal(actual_dist, d.contact.dist[0], 2)
    np.testing.assert_array_almost_equal(actual_pos, d.contact.pos[0], 2)
    np.testing.assert_array_almost_equal(actual_frame, d.contact.frame[0], 2)
