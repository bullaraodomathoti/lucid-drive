"""
Figure 4 (revised): Real nuScenes BEV + Real camera crops of detected objects.

Layout:
  - Left (60%): BEV with LiDAR / 3D bboxes / camera FOVs / radar
  - Right (40%): Grid of real camera crops for key detected objects,
                 connected by dotted lines to their BEV positions.
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import ConnectionPatch
from PIL import Image
import os, glob

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = "/home/sandbox/dataset_paper/figures/nuscenes"
LIDAR    = os.path.join(BASE, "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin")
PKL      = os.path.join(BASE, "n015-2018-07-24-11-22-45+0800.pkl")
OUT_FILE = "/home/sandbox/dataset_paper/figures/sensor_fusion_bev.png"

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
CAM_COLORS = {
    'CAM_FRONT':       '#00E5FF',
    'CAM_FRONT_LEFT':  '#69F0AE',
    'CAM_FRONT_RIGHT': '#B2FF59',
    'CAM_BACK':        '#FF6E40',
    'CAM_BACK_LEFT':   '#EA80FC',
    'CAM_BACK_RIGHT':  '#FFD740',
}

# ── Load data ─────────────────────────────────────────────────────────────────
pts = np.fromfile(LIDAR, dtype=np.float32).reshape(-1, 5)
with open(PKL, 'rb') as f:
    data = pickle.load(f)

instances  = data['data_list'][0]['instances']
img_info   = data['data_list'][0]['images']
label_map  = data['metainfo']['categories']
id_to_name = {v: k for k, v in label_map.items()}

RANGE = 50.0
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
mask = (np.abs(x) < RANGE) & (np.abs(y) < RANGE)
x, y, z = x[mask], y[mask], z[mask]

# ── Camera info ───────────────────────────────────────────────────────────────
cameras = {}
for cam_name, info in img_info.items():
    img_path = os.path.join(BASE, os.path.basename(info['img_path']))
    if not os.path.exists(img_path):
        # try any jpg with cam_name in filename
        candidates = glob.glob(os.path.join(BASE, f"*{cam_name}*"))
        img_path = candidates[0] if candidates else None
    cameras[cam_name] = {
        'K':        np.array(info['cam2img']),      # 3×3
        'lidar2cam': np.array(info['lidar2cam']),   # 4×4
        'cam2ego':  np.array(info['cam2ego']),      # 4×4
        'img_path': img_path,
    }

# ── Project 3D point (LiDAR frame) → camera image ────────────────────────────
def project_to_cam(cx, cy, cz, cam):
    p3d  = np.array([cx, cy, cz, 1.0])
    pcam = cam['lidar2cam'] @ p3d          # camera frame
    if pcam[2] <= 0.5:                     # behind camera
        return None, None, None
    K    = cam['K']
    uv   = K @ pcam[:3]
    u, v = uv[0] / uv[2], uv[1] / uv[2]
    return u, v, pcam[2]                   # u, v, depth

def best_camera(cx, cy, cz, min_px=30):
    """Return (cam_name, u, v, depth) for the camera where this point
    is most visible (in-bounds, closest to image centre, minimum depth)."""
    best = None
    best_score = 1e9
    for cname, cam in cameras.items():
        if cam['img_path'] is None:
            continue
        u, v, d = project_to_cam(cx, cy, cz, cam)
        if u is None:
            continue
        W, H = 1600, 900
        if min_px < u < W - min_px and min_px < v < H - min_px and d < 80:
            dist_to_center = np.sqrt((u - W/2)**2 + (v - H/2)**2)
            score = dist_to_center / (W/2)
            if score < best_score:
                best_score = score
                best = (cname, u, v, d)
    return best

# ── Crop object from camera image ─────────────────────────────────────────────
def crop_object(cam_name, u, v, length, width, depth):
    """Return PIL crop centred on (u,v) sized by projected bbox dims."""
    cam      = cameras[cam_name]
    img_path = cam['img_path']
    if img_path is None or not os.path.exists(img_path):
        return None
    img  = Image.open(img_path)
    W, H = img.size

    # Approximate projected size: focal * real_size / depth
    f        = cam['K'][0, 0]
    proj_l   = max(f * max(length, width) / max(depth, 1.0), 60)
    half     = int(proj_l * 0.65)
    half     = max(half, 50)
    half     = min(half, 300)

    x1 = max(0, int(u) - half)
    y1 = max(0, int(v) - half)
    x2 = min(W, int(u) + half)
    y2 = min(H, int(v) + half)
    crop = img.crop((x1, y1, x2, y2))
    return crop

# ── Select diverse objects to showcase ───────────────────────────────────────
priority_classes = ['car', 'truck', 'bus', 'bicycle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
selected = []   # list of dicts

seen_classes = set()
for inst in sorted(instances, key=lambda i: i['bbox_3d'][0]**2 + i['bbox_3d'][1]**2):
    lbl_id   = int(inst.get('bbox_label_3d', inst.get('bbox_label', -1)))
    cls_name = id_to_name.get(lbl_id, 'unknown')
    if cls_name not in priority_classes:
        continue
    if cls_name in seen_classes:
        continue
    cx, cy, cz, l, w, h, yaw = inst['bbox_3d']
    if abs(cx) > RANGE or abs(cy) > RANGE:
        continue
    result = best_camera(cx, cy, cz)
    if result is None:
        continue
    cname, u, v, depth = result
    crop = crop_object(cname, u, v, l, w, depth)
    if crop is None:
        continue
    selected.append({
        'cls': cls_name, 'bev_x': cx, 'bev_y': cy,
        'cam': cname, 'u': u, 'v': v, 'depth': depth,
        'crop': crop, 'color': CLASS_COLORS.get(cls_name, '#90A4AE'),
        'l': l, 'w': w,
    })
    seen_classes.add(cls_name)
    if len(selected) >= 8:
        break

# Also try to get more instances for common classes (up to 2 per class for variety)
seen2 = {s['cls']: 1 for s in selected}
for inst in sorted(instances, key=lambda i: i['bbox_3d'][0]**2 + i['bbox_3d'][1]**2):
    if len(selected) >= 9:
        break
    lbl_id   = int(inst.get('bbox_label_3d', inst.get('bbox_label', -1)))
    cls_name = id_to_name.get(lbl_id, 'unknown')
    if cls_name not in priority_classes[:4]:
        continue
    count = seen2.get(cls_name, 0)
    if count >= 2:
        continue
    cx, cy, cz, l, w, h, yaw = inst['bbox_3d']
    if abs(cx) > RANGE or abs(cy) > RANGE:
        continue
    # Skip already selected (by checking bev_x/y)
    already = any(abs(s['bev_x']-cx)<0.5 and abs(s['bev_y']-cy)<0.5 for s in selected)
    if already:
        continue
    result = best_camera(cx, cy, cz)
    if result is None:
        continue
    cname, u, v, depth = result
    crop = crop_object(cname, u, v, l, w, depth)
    if crop is None:
        continue
    selected.append({
        'cls': cls_name, 'bev_x': cx, 'bev_y': cy,
        'cam': cname, 'u': u, 'v': v, 'depth': depth,
        'crop': crop, 'color': CLASS_COLORS.get(cls_name, '#90A4AE'),
        'l': l, 'w': w,
    })
    seen2[cls_name] = count + 1

print(f"Selected {len(selected)} objects for crops: {[s['cls'] for s in selected]}")

# ── BEV helper ────────────────────────────────────────────────────────────────
def bbox_corners_bev(cx, cy, length, width, yaw):
    corners = np.array([[ length/2,  width/2],
                        [ length/2, -width/2],
                        [-length/2, -width/2],
                        [-length/2,  width/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s],[s, c]])
    rot = (R @ corners.T).T
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot

# ── Build figure ──────────────────────────────────────────────────────────────
N_CROPS   = len(selected)
CROP_COLS = 2
CROP_ROWS = (N_CROPS + 1) // 2

fig = plt.figure(figsize=(18, 11), facecolor='#0D1117')
# GridSpec: BEV on left (60%), crop grid on right (40%)
from matplotlib.gridspec import GridSpec
gs = GridSpec(CROP_ROWS, 4, figure=fig,
              left=0.03, right=0.97, top=0.94, bottom=0.04,
              wspace=0.15, hspace=0.22)

ax_bev = fig.add_subplot(gs[:, :2])       # BEV: columns 0-1
ax_bev.set_facecolor('#0D1117')

# ─── Draw BEV ────────────────────────────────────────────────────────────────
# Grid & rings
for v in np.arange(-RANGE, RANGE+1, 10):
    ax_bev.plot([-RANGE, RANGE], [v, v], color='#1E2832', lw=0.3, zorder=1)
    ax_bev.plot([v, v], [-RANGE, RANGE], color='#1E2832', lw=0.3, zorder=1)
for r in [10, 20, 30, 40, 50]:
    ax_bev.add_patch(plt.Circle((0,0), r, fill=False,
                                 color='#1E2832', lw=0.5, zorder=2))
    ax_bev.text(r*0.707, r*0.707, f'{r}m',
                color='#3A4A5C', fontsize=6.5, ha='center', va='center')

# LiDAR
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
sc = ax_bev.scatter(x, y, c=z_norm, cmap='plasma', s=0.3,
                    alpha=0.65, zorder=4, linewidths=0)

# Camera FOVs
FOV_HALF = np.radians(35)
CAM_LEN  = 44.0
for cname, cam in cameras.items():
    if cname not in CAM_COLORS:
        continue
    mat     = cam['cam2ego']
    fwd     = mat[:3, 2]
    heading = np.arctan2(fwd[1], fwd[0])
    cpos    = mat[:2, 3]
    color   = CAM_COLORS[cname]
    angles  = np.linspace(heading - FOV_HALF, heading + FOV_HALF, 40)
    ax      = [cpos[0]] + [cpos[0]+CAM_LEN*np.cos(a) for a in angles] + [cpos[0]]
    ay      = [cpos[1]] + [cpos[1]+CAM_LEN*np.sin(a) for a in angles] + [cpos[1]]
    ax_bev.add_patch(plt.Polygon(list(zip(ax, ay)), closed=True,
                                  fill=True, alpha=0.07,
                                  facecolor=color, edgecolor=color, lw=1.0, zorder=5))
    lx = cpos[0] + CAM_LEN*0.58*np.cos(heading)
    ly = cpos[1] + CAM_LEN*0.58*np.sin(heading)
    short = cname.replace('CAM_','')
    ax_bev.text(lx, ly, short, color=color, fontsize=6.2, fontweight='bold',
                ha='center', va='center', zorder=10,
                path_effects=[pe.withStroke(linewidth=1.4, foreground='#0D1117')])

# 3D bboxes + radar
radar_list = []
for inst in instances:
    bbox    = inst['bbox_3d']
    cx2,cy2,cz2,l,w,h,yaw = bbox
    lbl_id  = int(inst.get('bbox_label_3d', inst.get('bbox_label',-1)))
    cls_name= id_to_name.get(lbl_id,'unknown')
    color   = CLASS_COLORS.get(cls_name,'#90A4AE')
    nradar  = inst.get('num_radar_pts',0)
    vel     = inst.get('velocity',[0,0])
    if abs(cx2)>RANGE or abs(cy2)>RANGE:
        continue
    corners = bbox_corners_bev(cx2,cy2,l,w,yaw)
    ax_bev.add_patch(plt.Polygon(corners, closed=True, fill=True,
                                  facecolor=color, alpha=0.25,
                                  edgecolor=color, lw=1.3, zorder=7))
    ax_bev.text(cx2, cy2, cls_name[:3].upper(),
                color='white', fontsize=4.8, ha='center', va='center',
                fontweight='bold', zorder=9,
                path_effects=[pe.withStroke(linewidth=1.1, foreground='#0D1117')])
    if nradar > 0:
        radar_list.append((cx2,cy2,vel))

for (rx,ry,vel) in radar_list:
    ax_bev.scatter(rx,ry, s=55, marker='D', facecolor='none',
                   edgecolor='#00FFFF', lw=0.9, zorder=8)
    vx,vy = vel[0],vel[1]
    spd = np.sqrt(vx**2+vy**2)
    if spd > 0.3:
        sc2 = min(spd*1.5, 5.0)
        ax_bev.annotate('', xy=(rx+vx/spd*sc2,ry+vy/spd*sc2),
                        xytext=(rx,ry),
                        arrowprops=dict(arrowstyle='->',color='#00FFFF',lw=0.9),
                        zorder=9)

# Highlight selected objects with a marker ring
for sel in selected:
    ax_bev.add_patch(plt.Circle((sel['bev_x'], sel['bev_y']), 1.8,
                                 fill=False, edgecolor=sel['color'],
                                 lw=2.0, linestyle='--', zorder=11, alpha=0.9))

# Ego
ax_bev.add_patch(plt.Polygon([[-1.0,-2.3],[1.0,-2.3],[1.0,2.3],[-1.0,2.3]],
                               closed=True, facecolor='#FFD700',
                               edgecolor='white', lw=1.5, zorder=12))
ax_bev.text(0,0,'EGO',color='#0D1117',fontsize=7,fontweight='bold',
            ha='center',va='center',zorder=13)
ax_bev.annotate('',xy=(0,5),xytext=(0,2.6),
                arrowprops=dict(arrowstyle='->',color='white',lw=1.5),zorder=12)

# Legend
legend_elems = []
shown = set()
for inst in instances:
    lbl_id = int(inst.get('bbox_label_3d', inst.get('bbox_label',-1)))
    cn = id_to_name.get(lbl_id,'unknown')
    if cn not in shown:
        shown.add(cn)
        c = CLASS_COLORS.get(cn,'#90A4AE')
        legend_elems.append(mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.7,
                                            label=cn.replace('_',' ').title()))
legend_elems.append(plt.Line2D([0],[0],marker='D',color='#00FFFF',lw=0,
                                 markersize=5,markerfacecolor='none',label='Radar'))
legend_elems.append(mpatches.Patch(facecolor='#FFD700',edgecolor='white',label='Ego'))
leg = ax_bev.legend(handles=legend_elems, loc='lower left',
                    framealpha=0.35, facecolor='#0D1117',
                    edgecolor='#2E3F50', fontsize=6.0,
                    ncol=2, columnspacing=0.7,
                    handlelength=1.2, handletextpad=0.4)
plt.setp(leg.get_texts(), color='white')

# Colorbar
sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=z.min(), vmax=z.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_bev, fraction=0.022, pad=0.01, aspect=32)
cbar.set_label('Height (m)', color='white', fontsize=7.5)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=6.5)
cbar.outline.set_edgecolor('#2E3F50')

ax_bev.set_xlim(-RANGE, RANGE)
ax_bev.set_ylim(-RANGE, RANGE)
ax_bev.set_aspect('equal')
ax_bev.set_xlabel('X  (Forward)  [m]', color='#8899AA', fontsize=8.5)
ax_bev.set_ylabel('Y  (Left)  [m]',    color='#8899AA', fontsize=8.5)
ax_bev.tick_params(colors='#8899AA', labelsize=7.5)
for sp in ax_bev.spines.values():
    sp.set_edgecolor('#2E3F50')

# ─── Draw crop panels (right side) ───────────────────────────────────────────
crop_axes = []
for i, sel in enumerate(selected):
    row = i // CROP_COLS
    col = 2 + (i % CROP_COLS)
    ax_c = fig.add_subplot(gs[row, col])
    crop_axes.append(ax_c)

    crop_np = np.array(sel['crop'])
    ax_c.imshow(crop_np)
    ax_c.set_xticks([]); ax_c.set_yticks([])

    # Coloured border matching class colour
    for sp in ax_c.spines.values():
        sp.set_edgecolor(sel['color'])
        sp.set_linewidth(2.5)

    # Title label
    cam_short = sel['cam'].replace('CAM_','')
    ax_c.set_title(f"{sel['cls'].replace('_',' ').title()}\n"
                   f"({cam_short}, {sel['depth']:.0f}m)",
                   color=sel['color'], fontsize=7.2, fontweight='bold',
                   pad=2)

    # Draw connecting line from BEV to crop axes
    # BEV data coords → figure fraction via ax transform
    bev_x_fig, bev_y_fig = ax_bev.transData.transform(
        (sel['bev_x'], sel['bev_y']))
    bev_x_fig /= fig.get_size_inches()[0] * fig.dpi
    bev_y_fig /= fig.get_size_inches()[1] * fig.dpi

    # Crop axes left-centre in figure fraction
    bbox_c = ax_c.get_position()
    # We'll draw this after tight_layout; use ConnectionPatch instead
    con = ConnectionPatch(
        xyA=(sel['bev_x'], sel['bev_y']), coordsA=ax_bev.transData,
        xyB=(0.0, 0.5),                   coordsB=ax_c.transAxes,
        arrowstyle='->', color=sel['color'],
        linewidth=0.8, linestyle='--', alpha=0.55,
        zorder=20,
        axesA=ax_bev, axesB=ax_c
    )
    fig.add_artist(con)

# ─── Super-title ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Multi-Modal Sensor Fusion BEV  ·  Real nuScenes Data  ·  Camera Crops of Detected Objects',
    color='white', fontsize=11.5, fontweight='bold', y=0.98
)

fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight',
            facecolor='#0D1117', edgecolor='none')
plt.close(fig)
print(f"Saved: {OUT_FILE}  ({os.path.getsize(OUT_FILE):,} bytes)")
