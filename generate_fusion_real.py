"""
Generate Figure 4: Real nuScenes Multi-Modal Sensor Fusion BEV
Uses real LiDAR point cloud + 3D bbox annotations from mmdetection3d demo data.
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = "/home/sandbox/dataset_paper/figures/nuscenes"
LIDAR_BIN = os.path.join(BASE, "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin")
PKL_FILE  = os.path.join(BASE, "n015-2018-07-24-11-22-45+0800.pkl")
OUT_FILE  = "/home/sandbox/dataset_paper/figures/sensor_fusion_bev.png"

# ── Class colours ─────────────────────────────────────────────────────────────
CLASS_COLORS = {
    'car':                  '#FF6B6B',
    'truck':                '#FF9F43',
    'bus':                  '#FECA57',
    'trailer':              '#FF6B9D',
    'construction_vehicle': '#C44569',
    'bicycle':              '#26C6DA',
    'motorcycle':           '#42A5F5',
    'pedestrian':           '#66BB6A',
    'traffic_cone':         '#FFA726',
    'barrier':              '#AB47BC',
    'unknown':              '#90A4AE',
}

# Camera FOV colours
CAM_COLORS = {
    'CAM_FRONT':       '#00E5FF',
    'CAM_FRONT_LEFT':  '#69F0AE',
    'CAM_FRONT_RIGHT': '#B2FF59',
    'CAM_BACK':        '#FF6E40',
    'CAM_BACK_LEFT':   '#EA80FC',
    'CAM_BACK_RIGHT':  '#FFD740',
}

# ── Load LiDAR ─────────────────────────────────────────────────────────────────
pts = np.fromfile(LIDAR_BIN, dtype=np.float32).reshape(-1, 5)
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

# BEV range
RANGE = 50.0  # metres from ego

mask = (np.abs(x) < RANGE) & (np.abs(y) < RANGE)
x, y, z = x[mask], y[mask], z[mask]
print(f"LiDAR points in ±{RANGE}m: {len(x):,}")

# ── Load annotations ──────────────────────────────────────────────────────────
with open(PKL_FILE, 'rb') as f:
    data = pickle.load(f)

instances  = data['data_list'][0]['instances']
label_map  = data['metainfo']['categories']   # dict name->id
id_to_name = {v: k for k, v in label_map.items()}

# Camera extrinsics (cam2ego translation x,y only)
cam_info = data['data_list'][0]['images']
cam_positions = {}
cam_headings  = {}   # heading angle relative to ego (rad)

for cam_name, info in cam_info.items():
    if 'cam2ego' in info:
        mat = np.array(info['cam2ego'])           # 4×4
        cam_positions[cam_name] = mat[:2, 3]      # (x, y)
        # Camera optical axis in ego frame = R * [0,0,1,0]
        fwd = mat[:3, 2]                          # third column = Z-axis (forward)
        cam_headings[cam_name] = np.arctan2(fwd[1], fwd[0])
    else:
        # fallback from earlier parsing
        fallback = {
            'CAM_FRONT':       ([1.701,  0.016], 0.0),
            'CAM_FRONT_LEFT':  ([1.524,  0.495], np.pi/3),
            'CAM_FRONT_RIGHT': ([1.551, -0.493], -np.pi/3),
            'CAM_BACK':        ([0.028,  0.003], np.pi),
            'CAM_BACK_LEFT':   ([1.036,  0.485], 2*np.pi/3),
            'CAM_BACK_RIGHT':  ([1.015, -0.481], -2*np.pi/3),
        }
        if cam_name in fallback:
            cam_positions[cam_name] = np.array(fallback[cam_name][0])
            cam_headings[cam_name]  = fallback[cam_name][1]

print("Cameras found:", list(cam_positions.keys()))

# ── BEV footprint helper ───────────────────────────────────────────────────────
def bbox_corners_bev(cx, cy, length, width, yaw):
    """Return 4 corners of the BEV bbox footprint."""
    corners = np.array([
        [ length/2,  width/2],
        [ length/2, -width/2],
        [-length/2, -width/2],
        [-length/2,  width/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rotated = (R @ corners.T).T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0D1117')
ax.set_facecolor('#0D1117')

# Grid
for v in np.arange(-RANGE, RANGE+1, 10):
    ax.axhline(v, color='#1E2832', linewidth=0.4, zorder=1)
    ax.axvline(v, color='#1E2832', linewidth=0.4, zorder=1)
for v in np.arange(-RANGE, RANGE+1, 10):
    ax.plot([-RANGE, RANGE], [v, v], color='#1E2832', lw=0.3, zorder=1)

# Range rings
for r in [10, 20, 30, 40, 50]:
    circle = plt.Circle((0, 0), r, fill=False, color='#1E2832', lw=0.6, zorder=2)
    ax.add_patch(circle)
    ax.text(r * np.cos(np.radians(45)), r * np.sin(np.radians(45)),
            f'{r}m', color='#3A4A5C', fontsize=7, va='center', ha='center', zorder=3)

# ── LiDAR points (height-coloured) ────────────────────────────────────────────
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
ax.scatter(x, y, c=z_norm, cmap='plasma', s=0.3, alpha=0.6, zorder=4, linewidths=0)

# ── Camera FOV wedges ─────────────────────────────────────────────────────────
FOV_DEG   = 70.0   # typical nuScenes camera FOV
FOV_RAD   = np.radians(FOV_DEG / 2)
CAM_RANGE = 45.0

for cname, cpos in cam_positions.items():
    if cname not in CAM_COLORS:
        continue
    heading = cam_headings.get(cname, 0.0)
    color   = CAM_COLORS[cname]

    theta_l = heading - FOV_RAD
    theta_r = heading + FOV_RAD
    angles  = np.linspace(theta_l, theta_r, 40)
    arc_x   = [cpos[0]] + [cpos[0] + CAM_RANGE * np.cos(a) for a in angles] + [cpos[0]]
    arc_y   = [cpos[1]] + [cpos[1] + CAM_RANGE * np.sin(a) for a in angles] + [cpos[1]]
    wedge   = plt.Polygon(list(zip(arc_x, arc_y)),
                          closed=True, fill=True, alpha=0.07,
                          facecolor=color, edgecolor=color,
                          linewidth=1.0, zorder=5)
    ax.add_patch(wedge)
    # Label
    lx = cpos[0] + (CAM_RANGE * 0.6) * np.cos(heading)
    ly = cpos[1] + (CAM_RANGE * 0.6) * np.sin(heading)
    short = cname.replace('CAM_', '')
    ax.text(lx, ly, short, color=color, fontsize=6.5, fontweight='bold',
            ha='center', va='center', zorder=10,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#0D1117')])

# ── 3D bounding box BEV footprints ────────────────────────────────────────────
radar_instances = []  # collect those with radar pts for later

for inst in instances:
    bbox   = inst['bbox_3d']          # [x, y, z, l, w, h, yaw]
    cx, cy, cz, l, w, h, yaw = bbox
    label_id = inst.get('bbox_label_3d', inst.get('bbox_label', -1))
    cls_name = id_to_name.get(int(label_id), 'unknown')
    color    = CLASS_COLORS.get(cls_name, '#90A4AE')
    nradar   = inst.get('num_radar_pts', 0)
    vel      = inst.get('velocity', [0, 0])

    if abs(cx) > RANGE or abs(cy) > RANGE:
        continue

    corners = bbox_corners_bev(cx, cy, l, w, yaw)
    poly    = plt.Polygon(corners, closed=True, fill=True,
                          facecolor=color, alpha=0.25,
                          edgecolor=color, linewidth=1.2, zorder=7)
    ax.add_patch(poly)

    # Class label on box
    ax.text(cx, cy, cls_name[:3].upper(), color='white', fontsize=5,
            ha='center', va='center', fontweight='bold', zorder=9,
            path_effects=[pe.withStroke(linewidth=1.2, foreground='#0D1117')])

    if nradar > 0:
        radar_instances.append({'cx': cx, 'cy': cy, 'vel': vel,
                                 'color': color, 'cls': cls_name})

# ── Radar detections ──────────────────────────────────────────────────────────
# Radar returns from instances that have num_radar_pts > 0
for ri in radar_instances:
    cx, cy = ri['cx'], ri['cy']
    vx, vy = ri['vel'][0], ri['vel'][1]
    speed  = np.sqrt(vx**2 + vy**2)

    ax.scatter(cx, cy, s=60, marker='D',
               facecolor='none', edgecolor='#00FFFF',
               linewidth=1.0, zorder=8)
    ax.scatter(cx, cy, s=12, color='#00FFFF', zorder=8, alpha=0.8)

    # Velocity arrow
    if speed > 0.3:
        scale = min(speed * 1.5, 6.0)
        ax.annotate('', xy=(cx + vx/speed*scale, cy + vy/speed*scale),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='#00FFFF',
                                    lw=1.0), zorder=9)

print(f"Radar detections plotted: {len(radar_instances)}")

# ── Ego vehicle ───────────────────────────────────────────────────────────────
ego = plt.Polygon([[-1.0, -2.3], [1.0, -2.3], [1.0, 2.3], [-1.0, 2.3]],
                   closed=True, facecolor='#FFD700', edgecolor='white',
                   linewidth=1.5, zorder=12)
ax.add_patch(ego)
ax.text(0, 0, 'EGO', color='#0D1117', fontsize=7, fontweight='bold',
        ha='center', va='center', zorder=13)

# Forward arrow
ax.annotate('', xy=(0, 5), xytext=(0, 2.5),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5), zorder=12)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = []
# Class boxes
shown_classes = set()
for inst in instances:
    label_id = inst.get('bbox_label_3d', inst.get('bbox_label', -1))
    cls_name = id_to_name.get(int(label_id), 'unknown')
    if cls_name not in shown_classes:
        shown_classes.add(cls_name)
        color = CLASS_COLORS.get(cls_name, '#90A4AE')
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.7,
                           label=cls_name.replace('_', ' ').title()))

# Sensor symbols
legend_elements.append(mpatches.Patch(facecolor='#FF6347', alpha=0.2,
                                       edgecolor='#FF6347', label='Camera FOV'))
legend_elements.append(
    plt.Line2D([0], [0], marker='D', color='#00FFFF', linewidth=0,
               markersize=6, markerfacecolor='none', label='Radar Detection'))
legend_elements.append(
    plt.Line2D([0], [0], color='#FFD700', linewidth=0,
               marker='s', markersize=8, label='Ego Vehicle'))

legend = ax.legend(handles=legend_elements, loc='lower left',
                   framealpha=0.4, facecolor='#0D1117',
                   edgecolor='#2E3F50', fontsize=6.5,
                   ncol=2, columnspacing=0.8,
                   handlelength=1.2, handletextpad=0.4)
plt.setp(legend.get_texts(), color='white')

# ── Colorbar for LiDAR height ─────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=z.min(), vmax=z.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, aspect=30)
cbar.set_label('LiDAR Height (m)', color='white', fontsize=8)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
cbar.outline.set_edgecolor('#2E3F50')

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-RANGE, RANGE)
ax.set_ylim(-RANGE, RANGE)
ax.set_aspect('equal')
ax.set_xlabel('X  (Forward →)  [m]', color='#8899AA', fontsize=9)
ax.set_ylabel('Y  (Left →)  [m]',    color='#8899AA', fontsize=9)
ax.tick_params(colors='#8899AA', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#2E3F50')

ax.set_title('Multi-Modal Sensor Fusion BEV — Real nuScenes Data\n'
             'LiDAR (height) · 3D Bounding Boxes · Camera FOVs · Radar Detections',
             color='white', fontsize=11, fontweight='bold', pad=12)

plt.tight_layout()
fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight',
            facecolor='#0D1117', edgecolor='none')
plt.close(fig)
print(f"Saved: {OUT_FILE}  ({os.path.getsize(OUT_FILE):,} bytes)")
