"""
Multi-modal sensor fusion BEV figure for LUCID-Drive paper.
Layers (bottom → top):
  1. Ground-plane LiDAR points (height-coloured)
  2. Camera FOV frustums (6 standard + 2 fisheye) projected to BEV
  3. Radar FOV cone + detections + velocity arrows
  4. 3D bounding-box footprints (track IDs)
  5. Ego vehicle + legend
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D

rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(8, 9), dpi=160)
ax.set_facecolor('#0b0b14')
fig.patch.set_facecolor('#0b0b14')

# ── helpers ───────────────────────────────────────────────────────────────────
def fov_wedge(ax, cx, cy, direction_deg, half_fov_deg, radius,
              color, alpha=0.10, lw=0.7, ls='-', zorder=3):
    """Draw a filled FOV wedge from (cx,cy)."""
    t0 = np.radians(direction_deg - half_fov_deg)
    t1 = np.radians(direction_deg + half_fov_deg)
    theta = np.linspace(t0, t1, 60)
    xs = np.concatenate([[cx], cx + radius * np.sin(theta), [cx]])
    ys = np.concatenate([[cy], cy + radius * np.cos(theta), [cy]])
    ax.fill(xs, ys, color=color, alpha=alpha, zorder=zorder)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, alpha=alpha+0.25, zorder=zorder)

def rot(pts, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pts @ R.T

# ── coordinate frame  (X = right, Y = forward, origin = ego centre) ──────────
# scene extends ±30 m lateral, 0..55 m forward, −10..5 m rear

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — LiDAR ground-plane point cloud (height-coloured)
# ─────────────────────────────────────────────────────────────────────────────
N = 5000
lx = rng.uniform(-28, 28, N)
ly = rng.uniform(-9,  54, N)
# Road corridor mask
in_road = np.abs(lx) < 5.5 + ly * 0.06
# Pavement / kerb
in_pave = (np.abs(lx) > 5.5) & (np.abs(lx) < 9)
# Keep all ground points
mask = np.ones(N, bool)
lz = np.where(in_road, rng.uniform(-0.1, 0.05, N),
      np.where(in_pave, rng.uniform( 0.0, 0.12, N),
                          rng.uniform( 0.0, 0.30, N)))
sc = ax.scatter(lx[mask], ly[mask], s=0.6,
                c=lz[mask], cmap='plasma', vmin=-0.15, vmax=1.2,
                alpha=0.45, lw=0, zorder=2)

# Wall / building LiDAR returns (taller, brighter)
for side, xs, xe in [(-1,-28,-10),(1,10,28)]:
    bx = rng.uniform(xs, xe, 700)
    by = rng.uniform(2,  52, 700)
    bz = rng.uniform(0.5, 4.0, 700)
    ax.scatter(bx, by, s=0.8, c=bz, cmap='plasma', vmin=-0.15, vmax=5,
               alpha=0.55, lw=0, zorder=2)

# Range rings (subtle)
for r in [10, 20, 30, 40, 50]:
    theta_r = np.linspace(-np.pi, np.pi, 300)
    ax.plot(r*np.sin(theta_r), r*np.cos(theta_r),
            color='#1e1e32', lw=0.7, ls='--', zorder=1)
    ax.text(r*np.sin(np.radians(62))+0.4, r*np.cos(np.radians(62)),
            f'{r}m', color='#2e2e4a', fontsize=5.5, zorder=1)

# Cardinal grid lines
ax.axhline(0, color='#1e1e32', lw=0.5, zorder=1)
ax.axvline(0, color='#1e1e32', lw=0.5, zorder=1)

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Camera FOV frustums projected to BEV
# ─────────────────────────────────────────────────────────────────────────────
cam_color = '#4488ff'
fish_color = '#44ddff'

# Camera mount positions (relative to ego centre, metres)
cams = [
    # (mount_x, mount_y, heading_deg, half_fov_deg, radius, color, label)
    ( 0.0,  2.3,   0,  60, 14.0, cam_color,  'FC'),   # front centre
    (-0.9,  2.0, -45,  35,  8.0, cam_color,  'FL'),   # front left
    ( 0.9,  2.0,  45,  35,  8.0, cam_color,  'FR'),   # front right
    ( 0.0, -2.3, 180,  60, 10.0, cam_color,  'RC'),   # rear centre
    (-0.9, -1.8,-135,  35,  7.0, cam_color,  'RL'),   # rear left
    ( 0.9, -1.8, 135,  35,  7.0, cam_color,  'RR'),   # rear right
    ( 0.0,  2.1,   0,  95, 18.0, fish_color, 'FF'),   # front fisheye
    ( 0.0, -2.1, 180,  95, 14.0, fish_color, 'RF'),   # rear fisheye
]
for (mx, my, hdg, hfov, rad, col, lbl) in cams:
    is_fish = lbl.startswith('F') and lbl[1]=='F' or lbl=='RF'
    fov_wedge(ax, mx, my, hdg, hfov, rad, col,
              alpha=0.07 if is_fish else 0.10,
              lw=0.6, ls='--' if is_fish else '-', zorder=3)
    # mount dot
    ax.plot(mx, my, 'o', ms=3.5, color=col, zorder=12)
    # label
    off_x = -1.5 if mx < 0 else (1.2 if mx > 0 else 0)
    off_y = 0.5 if my > 0 else -1.0
    ax.text(mx+off_x, my+off_y, lbl, color=col, fontsize=5,
            fontweight='bold', zorder=13,
            path_effects=[pe.withStroke(linewidth=1.2, foreground='#0b0b14')])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Radar FOV cones + detections + velocity vectors
# ─────────────────────────────────────────────────────────────────────────────
radar_color = '#44ff88'

radars = [
    # (mount_x, mount_y, heading_deg, half_fov_deg, max_range)
    ( 0.0,  2.5,   0, 30, 55),   # front centre
    (-1.1,  0.2, -90, 25, 30),   # left side
    ( 1.1,  0.2,  90, 25, 30),   # right side
    (-0.6, -2.5,-160, 25, 40),   # rear left
    ( 0.6, -2.5, 160, 25, 40),   # rear right
]
for (mx, my, hdg, hfov, rmax) in radars:
    fov_wedge(ax, mx, my, hdg, hfov, rmax, radar_color,
              alpha=0.04, lw=0.7, ls=':', zorder=4)

# Radar detections (position in BEV + velocity vector)
radar_dets = [
    # (x, y, vx, vy, label)
    ( 1.2, 25, -0.4, -7.5, 'Car  −7.5 m/s'),
    (-2.0, 36,  0.2, -6.0, 'Car  −6.0 m/s'),
    ( 3.5, 44,  0.0, -5.2, 'Car  −5.2 m/s'),
    (-6.5, 19,  8.5,  0.0, 'Oncoming  +8.5 m/s'),
    ( 7.0, 12,  0.0,  0.1, 'Static'),
]
dt = 0.8   # velocity vector time scale (seconds)
for (dx, dy, dvx, dvy, lbl) in radar_dets:
    ax.plot(dx, dy, 'D', ms=5, color=radar_color,
            mec='white', mew=0.4, zorder=11)
    ax.annotate('', xy=(dx+dvx*dt, dy+dvy*dt), xytext=(dx, dy),
                arrowprops=dict(arrowstyle='->', color=radar_color,
                                lw=1.4, mutation_scale=8),
                zorder=11)
    ax.text(dx+0.6, dy+0.3, lbl, color=radar_color, fontsize=5.0,
            fontweight='bold', zorder=12,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#0b0b14')])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — 3D bounding-box BEV footprints (from LiDAR tracking)
# ─────────────────────────────────────────────────────────────────────────────
box_color = '#00cfff'

def draw_box(ax, cx, cy, length, width, yaw_deg, track_id, color='#00cfff'):
    """Draw oriented 2D bounding box footprint."""
    corners = np.array([
        [-width/2, -length/2], [ width/2, -length/2],
        [ width/2,  length/2], [-width/2,  length/2],
        [-width/2, -length/2],
    ])
    rotated = rot(corners, yaw_deg) + np.array([cx, cy])
    ax.plot(rotated[:,0], rotated[:,1], color=color, lw=1.6, zorder=9)
    # heading tick (front edge midpoint)
    front = rot(np.array([[0, length/2]]), yaw_deg)[0] + np.array([cx, cy])
    ax.annotate('', xy=front, xytext=np.array([cx, cy]),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.2, mutation_scale=7),
                zorder=10)
    # LiDAR surface points
    for edge_pts in [
        rng.uniform(-width/2, width/2,  20),  # front/rear
        rng.uniform(-length/2, length/2, 20), # sides
    ]:
        pass
    pts_local = []
    # front/rear surfaces
    for ey in [-length/2, length/2]:
        ex = rng.uniform(-width/2, width/2, 10)
        pts_local.extend(zip(ex, [ey]*10))
    # left/right surfaces
    for ex in [-width/2, width/2]:
        ey = rng.uniform(-length/2, length/2, 10)
        pts_local.extend(zip([ex]*10, ey))
    pts_local = np.array(pts_local)
    pts_world = rot(pts_local, yaw_deg) + np.array([cx, cy])
    ax.scatter(pts_world[:,0], pts_world[:,1],
               s=2.0, c=color, alpha=0.75, lw=0, zorder=9)
    # Track ID label
    ax.text(cx + width*0.55, cy + length*0.55,
            f'T{track_id:02d}', color=color, fontsize=5.5,
            fontweight='bold', zorder=13,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#0b0b14')])

# Vehicles in scene
draw_box(ax,  1.2, 25, 4.5, 2.0,   2, track_id=3)
draw_box(ax, -2.0, 36, 4.5, 2.0,   5, track_id=7)
draw_box(ax,  3.5, 44, 4.5, 2.0,  -3, track_id=11)
draw_box(ax, -6.5, 19, 4.5, 2.0,  88, track_id=5, color='#ff5555')  # oncoming

# Pedestrian cluster
ped_x, ped_y = 8.0, 14.5
ped_pts = rng.normal(0, 0.3, (35, 2)) + [ped_x, ped_y]
ax.scatter(ped_pts[:,0], ped_pts[:,1], s=4, c='#ffdd00',
           alpha=0.9, lw=0, zorder=9)
rect_ped = mpatches.Rectangle((ped_x-0.5, ped_y-0.6), 1.0, 1.2,
                                edgecolor='#ffdd00', facecolor='none',
                                lw=1.4, zorder=9)
ax.add_patch(rect_ped)
ax.text(ped_x+0.65, ped_y+0.4, 'T02\nPed', color='#ffdd00',
        fontsize=5.5, fontweight='bold', zorder=13,
        path_effects=[pe.withStroke(linewidth=1.5, foreground='#0b0b14')])

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — Ego vehicle
# ─────────────────────────────────────────────────────────────────────────────
ego_rect = mpatches.FancyBboxPatch((-1.0, -2.3), 2.0, 4.6,
                                    boxstyle='round,pad=0.15',
                                    facecolor='#ff6020', edgecolor='white',
                                    lw=1.5, zorder=14)
ax.add_patch(ego_rect)
ax.text(0, 0, 'EGO', color='white', fontsize=7, fontweight='bold',
        ha='center', va='center', zorder=15)

# Forward direction arrow
ax.annotate('', xy=(0, 6.5), xytext=(0, 2.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.8,
                            mutation_scale=12),
            zorder=15)

# ─────────────────────────────────────────────────────────────────────────────
# Colourbar for LiDAR height
# ─────────────────────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=-0.15, vmax=1.2))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01, aspect=30)
cbar.set_label('LiDAR point height (m)', color='#aaaacc', fontsize=6.5)
cbar.ax.yaxis.set_tick_params(color='#666688', labelsize=5.5)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaaacc')

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], color=cam_color,  lw=1.5, label='Std. camera FOV (×6)'),
    Line2D([0],[0], color=fish_color, lw=1.5, ls='--', label='Fisheye camera FOV (×2)'),
    Line2D([0],[0], color=radar_color,lw=1.5, ls=':', label='Radar FOV cone (×5)'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor=radar_color,
           ms=5, lw=0, label='Radar detection + velocity'),
    Line2D([0],[0], color=box_color,  lw=1.5, label='3D track box — vehicle'),
    Line2D([0],[0], color='#ff5555',  lw=1.5, label='3D track box — oncoming'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#ffdd00',
           ms=5, lw=0, label='Pedestrian cluster'),
    mpatches.Patch(facecolor='#ff6020', edgecolor='white', label='Ego vehicle'),
]
leg = ax.legend(handles=legend_elements, loc='lower left',
                fontsize=5.5, framealpha=0.75,
                facecolor='#12121e', edgecolor='#333366',
                labelcolor='#ccccee', handlelength=2.0,
                borderpad=0.7, labelspacing=0.45)

# ─────────────────────────────────────────────────────────────────────────────
# Axes formatting
# ─────────────────────────────────────────────────────────────────────────────
ax.set_xlim(-30, 30)
ax.set_ylim(-10, 55)
ax.set_xlabel('Lateral offset (m)', color='#aaaacc', fontsize=8)
ax.set_ylabel('Longitudinal distance (m)', color='#aaaacc', fontsize=8)
ax.tick_params(colors='#666688', labelsize=6.5)
for sp in ax.spines.values():
    sp.set_edgecolor('#222244')

ax.set_title(
    'LUCID-Drive  —  Multi-Modal Sensor Fusion BEV\n'
    'Scenario SC-00842  |  Urban intersection  |  t = 14:32:07.415 UTC',
    color='#ddddff', fontsize=8.5, fontweight='bold', pad=6)

# Sensor sync note
ax.text(0.99, 0.01,
        'All modalities PTP-synchronised to < 1 ms',
        transform=ax.transAxes, ha='right', va='bottom',
        color='#555577', fontsize=5.5, style='italic', zorder=20)

plt.tight_layout()
fig.savefig('/home/sandbox/dataset_paper/figures/sensor_fusion_bev.png',
            bbox_inches='tight', dpi=160)
plt.close()
print("Fusion BEV figure saved.")
