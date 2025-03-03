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

import warp as wp

from .types import Model
from .types import Data
from .types import Contact
from .types import GeomType
from .math import make_frame
from .math import matmul_unroll_33


@wp.func
def _gradient_step(
  geom1_pos: wp.vec3,
  geom2_pos: wp.vec3,
  geom1_mat: wp.mat33,
  geom2_mat: wp.mat33,
  geom1_size: wp.vec3,
  geom2_size: wp.vec3,
  x: wp.vec3,
):
  """Performs a step of gradient descent."""
  amin = 1e-4  # minimum value for line search factor scaling the gradient
  amax = 2.0  # maximum value for line search factor scaling the gradient
  nlinesearch = 10  # line search points
  dh = 1.0e-5
  x_plus = wp.vec3(x[0] + dh, x[1] + dh, x[2] + dh)
  cylinder1_pos = _cylinder_frame(x, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_size)
  cylinder2_pos = _cylinder(x, geom2_size)
  cylinder1_pos_plus = _cylinder_frame(x_plus, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_size)
  cylinder2_pos_plus = _cylinder(x_plus, geom2_size)
  clearance = cylinder1_pos + cylinder2_pos + wp.abs(wp.max(cylinder1_pos, cylinder2_pos))
  clearance_plus = cylinder1_pos_plus + cylinder2_pos_plus + wp.abs(wp.max(cylinder1_pos_plus, cylinder2_pos_plus))
  grad_clearance = (clearance_plus - clearance) / dh
  ratio = (amax / amin) ** (1.0 / float(nlinesearch - 1))
  value_prev = 1.0e10
  candidate_prev = wp.vec3(0.0)
  for i in range(nlinesearch):
    alpha = amin * (ratio ** float(i))
    candidate = wp.vec3(x[0] - alpha * grad_clearance, x[1] - alpha * grad_clearance, x[2] - alpha * grad_clearance)
    cylinder1_pos = _cylinder_frame(candidate, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_size)
    cylinder2_pos = _cylinder(candidate, geom2_size)
    value = cylinder1_pos + cylinder2_pos + wp.abs(wp.max(cylinder1_pos, cylinder2_pos))
    if value < value_prev:
      value_prev = value
      candidate_prev = candidate
    else:
      return candidate_prev

  return candidate


@wp.func
def _gradient_descent(
  geom1_pos: wp.vec3,
  geom2_pos: wp.vec3,
  geom1_mat: wp.mat33,
  geom2_mat: wp.mat33,
  geom1_size: wp.vec3,
  geom2_size: wp.vec3,
  x: wp.vec3,
  niter: int,
):
  """Performs gradient descent with backtracking line search."""

  for _ in range(niter):
    x = _gradient_step(geom1_pos, geom2_pos, geom1_mat, geom2_mat, geom1_size, geom2_size, x)

  return x


@wp.func
def _cylinder(pos: wp.vec3, size: wp.vec3) -> wp.float32:
  a0 = wp.sqrt(pos[0] * pos[0] + pos[1] * pos[1]) - size[0]
  a1 = wp.abs(pos[2]) - size[1]
  b0 = wp.max(a0, 0.0)
  b1 = wp.max(a1, 0.0)
  return wp.min(wp.max(a0, a1), 0.0) + wp.sqrt(b0 * b0 + b1 * b1)


@wp.func
def _cylinder_frame(
    pos: wp.vec3,
    from_pos: wp.vec3,
    from_mat: wp.mat33,
    to_pos: wp.vec3,
    to_mat: wp.mat33,
    size: wp.vec3
) -> wp.float32:
  relmat = matmul_unroll_33(wp.transpose(to_mat), from_mat)
  relpos = wp.transpose(to_mat) @ (from_pos - to_pos)
  new_pos = relmat @ pos + relpos

  return _cylinder(new_pos, size)


@wp.func
def _optim(
    geom1_pos: wp.vec3,
    geom2_pos: wp.vec3,
    geom1_mat: wp.mat33,
    geom2_mat: wp.mat33,
    geom1_size: wp.vec3,
    geom2_size: wp.vec3,
    x0: wp.vec3,
):
  """Optimizes the clearance function."""
  x0 = wp.transpose(geom2_mat) @ (x0 - geom2_pos)
  pos = _gradient_descent(geom1_pos, geom2_pos, geom1_mat, geom2_mat, geom1_size, geom2_size, x0, 10)
  cylinder1_pos = _cylinder_frame(pos, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_size)
  cylinder2_pos = _cylinder(pos, geom2_size)
  dist = cylinder1_pos + cylinder2_pos

  dh = 1.0e-5
  pos_plus_x = wp.vec3(pos[0] + dh, pos[1], pos[2])
  pos_plus_y = wp.vec3(pos[0], pos[1] + dh, pos[2])
  pos_plus_z = wp.vec3(pos[0], pos[1], pos[2] + dh)
  grad_cylinder1_pos_x = (_cylinder_frame(pos_plus_x, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_size) - cylinder1_pos) / dh
  grad_cylinder2_pos_x = (_cylinder(pos_plus_x, geom2_size) - cylinder2_pos) / dh
  grad_cylinder1_pos_y = (_cylinder_frame(pos_plus_y, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_size) - cylinder1_pos) / dh
  grad_cylinder2_pos_y = (_cylinder(pos_plus_y, geom2_size) - cylinder2_pos) / dh
  grad_cylinder1_pos_z = (_cylinder_frame(pos_plus_z, geom2_pos, geom2_mat, geom1_pos, geom1_mat, geom1_size) - cylinder1_pos) / dh
  grad_cylinder2_pos_z = (_cylinder(pos_plus_z, geom2_size) - cylinder2_pos) / dh
  pos = geom2_mat @ pos + geom2_pos  # d2 to global frame
  n = wp.vec3(grad_cylinder1_pos_x - grad_cylinder2_pos_x, grad_cylinder1_pos_y - grad_cylinder2_pos_y, grad_cylinder1_pos_z - grad_cylinder2_pos_z)
  n = geom2_mat @ n
  return dist, pos, make_frame(n)


@wp.kernel
def plane_convex_kernel(m: Model, d: Data, group_key: int):
  """Calculates contacts between a plane and a convex object."""
  tid = wp.tid()
  num_candidate_contacts = d.narrowphase_candidate_group_count[group_key]
  if tid >= num_candidate_contacts:
    return

  geoms = d.narrowphase_candidate_geom[group_key, tid]
  worldid = d.narrowphase_candidate_worldid[group_key, tid]

  # plane is always first, convex could be box/mesh.
  plane_geom = geoms[0]
  convex_geom = geoms[1]

  convex_type = m.geom_type[convex_geom]
  # if convex_type == wp.static(GeomType.BOX.value):
  #  pass # box-specific stuff - many things can be hardcoded here
  # else:
  #  pass # mesh-specific stuff

  # if contact
  index = wp.atomic_add(d.contact_counter, 0, 1)
  # d.contact.dist[index] = dist
  # d.contact.pos[index] = pos
  # d.contact.frame[index] = frame
  # d.contact.worldid[index] = worldid


@wp.kernel
def cylinder_cylinder_kernel(m: Model, d: Data, group_key: int):
  """Calculates contacts between a cylinder and a cylinder object."""
  tid, condim = wp.tid()
  num_candidate_contacts = d.narrowphase_candidate_group_count[group_key]
  if tid >= num_candidate_contacts:
    return

  geoms = d.narrowphase_candidate_geom[group_key, tid]
  worldid = d.narrowphase_candidate_worldid[group_key, tid]

  cylinder_geom1 = geoms[0]
  cylinder_geom2 = geoms[1]

  geom1_pos = d.geom_xpos[worldid, cylinder_geom1]
  geom2_pos = d.geom_xpos[worldid, cylinder_geom2]
  geom1_mat = d.geom_xmat[worldid, cylinder_geom1]
  geom2_mat = d.geom_xmat[worldid, cylinder_geom2]
  geom1_size = m.geom_size[cylinder_geom1]
  geom2_size = m.geom_size[cylinder_geom2]

  basis = make_frame(geom2_pos - geom1_pos)
  mid = 0.5 * (geom1_pos + geom2_pos)
  r = wp.max(geom1_size[0], geom2_size[0])
  if condim == 0:
    x_condim = r * basis[1]
  elif condim == 1:
    x_condim = r * basis[2]
  elif condim == 2:
    x_condim = -r * basis[1]
  elif condim == 3:
    x_condim = -r * basis[2]
  x0 = mid + x_condim
  #x0 = wp.vec3(mid[0] + x_condim[condim], mid[1] + x_condim[condim], mid[2] + x_condim[condim])
  dist, pos, frame = _optim(geom1_pos, geom2_pos, geom1_mat, geom2_mat, geom1_size, geom2_size, x0)
  index = wp.atomic_add(d.contact_counter, 0, 1)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = frame
  d.contact.worldid[index] = worldid


def plane_sphere(m: Model, d: Data, group_key: int):
  pass


def plane_capsule(m: Model, d: Data, group_key: int):
  pass


def plane_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def plane_cylinder(m: Model, d: Data, group_key: int):
  pass


def plane_convex(m: Model, d: Data, group_key: int):
  wp.launch(
    kernel=plane_convex_kernel,
    dim=(d.nconmax),
    inputs=[m, d, group_key],
  )


def hfield_sphere(m: Model, d: Data, group_key: int):
  pass


def hfield_capsule(m: Model, d: Data, group_key: int):
  pass


def hfield_convex(m: Model, d: Data, group_key: int):
  pass


def hfield_convex(m: Model, d: Data, group_key: int):
  pass


def sphere_sphere(m: Model, d: Data, group_key: int):
  pass


def sphere_capsule(m: Model, d: Data, group_key: int):
  pass


def sphere_cylinder(m: Model, d: Data, group_key: int):
  pass


def sphere_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def sphere_convex(m: Model, d: Data, group_key: int):
  pass


def sphere_convex(m: Model, d: Data, group_key: int):
  pass


def capsule_capsule(m: Model, d: Data, group_key: int):
  pass


def capsule_convex(m: Model, d: Data, group_key: int):
  pass


def capsule_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def capsule_cylinder(m: Model, d: Data, group_key: int):
  pass


def capsule_convex(m: Model, d: Data, group_key: int):
  pass


def ellipsoid_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def ellipsoid_cylinder(m: Model, d: Data, group_key: int):
  pass


def cylinder_cylinder(m: Model, d: Data, group_key: int):
  wp.launch(
    kernel=cylinder_cylinder_kernel,
    dim=(d.nconmax, 4),
    inputs=[m, d, group_key],
  )


def box_box(m: Model, d: Data, group_key: int):
  pass


def convex_convex(m: Model, d: Data, group_key: int):
  pass


def convex_convex(m: Model, d: Data, group_key: int):
  pass
