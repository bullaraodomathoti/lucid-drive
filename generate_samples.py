"""
Generate sensor data sample figures for LUCID-Drive paper:
  1. Camera image sample (front-centre driving scene)
  2. LiDAR bird's-eye-view point cloud
  3. Radar range-velocity (Doppler) plot
  4. Combined multi-sensor overlay (BEV)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Camera front-centre scene
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
ax.set_xlim(0, 700); ax.set_ylim(0, 380); ax.axis('off')

# Sky gradient
sky = np.linspace([0.53,0.73,0.87], [0.78,0.88,0.96], 180)
for i, c in enumerate(sky):
    ax.axhspan(380-i*1, 380-(i+1)*1, color=c, lw=0)

# Clouds
for cx, cy, cr in [(120,340,28),(160,350,18),(300,330,35),(340,345,22),(560,338,30),(600,352,20)]:
    ax.add_patch(patches.Ellipse((cx,cy), cr*2.5, cr*0.7, color='white', alpha=0.85, zorder=2))

# Road surface
road = plt.Polygon([[180,200],[520,200],[700,0],[0,0]], color='#4a4a4a', zorder=3)
ax.add_patch(road)

# Road horizon blend
for i in range(30):
    alpha = 0.04*(30-i)/30
    ax.axhspan(200+i*0.5, 201+i*0.5, color='#4a4a4a', alpha=alpha, zorder=3)

# Lane markings (perspective dashed white lines)
for lane_x_bot, lane_x_top in [(350,342),(310,335),(390,348)]:
    for seg in range(8):
        t0 = seg/8; t1 = (seg+0.4)/8
        x0 = 350 + (lane_x_bot-350)*t0; x1 = 350 + (lane_x_bot-350)*t1
        y0 = 200*(1-t0); y1 = 200*(1-t1)
        w = max(1, 3*(1-t0))
        ax.plot([x0,x1],[y0,y1], color='white', lw=w, alpha=0.9, zorder=5)

# Left lane edge
ax.plot([180,350],[200,0], color='#f0c040', lw=2.5, alpha=0.9, zorder=5)
# Right lane edge
ax.plot([520,350],[200,0], color='#f0c040', lw=2.5, alpha=0.9, zorder=5)

# Pavement / kerb
ax.add_patch(plt.Polygon([[0,200],[180,200],[0,80]], color='#6e6e5e', zorder=3))
ax.add_patch(plt.Polygon([[700,200],[520,200],[700,80]], color='#6e6e5e', zorder=3))

# Buildings (left)
for bx, bw, bh, bc in [(0,70,160,'#8a9a8a'),(65,55,130,'#7a8a7a'),(115,50,110,'#9aaa9a')]:
    ax.add_patch(patches.Rectangle((bx,200), bw, bh, color=bc, zorder=4))
    for wx in range(bx+8, bx+bw-8, 14):
        for wy in range(210, 200+bh-10, 20):
            ax.add_patch(patches.Rectangle((wx,wy),8,10, color='#d4e0f0', alpha=0.7, zorder=5))

# Buildings (right)
for bx, bw, bh, bc in [(580,120,150,'#8a8a9a'),(650,50,120,'#7a7a8a')]:
    ax.add_patch(patches.Rectangle((bx,200), bw, bh, color=bc, zorder=4))
    for wx in range(bx+8, bx+bw-8, 14):
        for wy in range(210, 200+bh-10, 20):
            ax.add_patch(patches.Rectangle((wx,wy),8,10, color='#d4e0f0', alpha=0.7, zorder=5))

# Trees
for tx, th in [(155,70),(165,55),(540,65),(550,50),(565,75)]:
    ax.add_patch(patches.Rectangle((tx-3,200), 6, th*0.3, color='#5a4a30', zorder=4))
    ax.add_patch(patches.Ellipse((tx,200+th*0.3+th*0.35), th*0.5, th*0.7, color='#2d6a2d', zorder=4))

# Lead car (silver sedan, ~80 m ahead)
cx, cy, cw, ch = 295, 115, 110, 45
ax.add_patch(patches.FancyBboxPatch((cx,cy), cw, ch, boxstyle='round,pad=3',
                                     color='#c0c8d0', zorder=7))
ax.add_patch(patches.FancyBboxPatch((cx+15,cy+ch-18), cw-30, 20,
                                     boxstyle='round,pad=2', color='#7090b0', alpha=0.75, zorder=8))
ax.add_patch(patches.Ellipse((cx+12, cy+2), 22, 10, color='#1a1a1a', zorder=8))
ax.add_patch(patches.Ellipse((cx+cw-12, cy+2), 22, 10, color='#1a1a1a', zorder=8))
# Tail lights
ax.add_patch(patches.Rectangle((cx+2, cy+ch-8), 18, 7, color='#ff2020', alpha=0.9, zorder=9))
ax.add_patch(patches.Rectangle((cx+cw-20, cy+ch-8), 18, 7, color='#ff2020', alpha=0.9, zorder=9))

# Oncoming car (red, left lane)
ox, oy, ow, oh = 215, 155, 90, 38
ax.add_patch(patches.FancyBboxPatch((ox,oy), ow, oh, boxstyle='round,pad=3',
                                     color='#c03030', zorder=7))
ax.add_patch(patches.FancyBboxPatch((ox+12,oy+oh-16), ow-24, 18,
                                     boxstyle='round,pad=2', color='#7090b0', alpha=0.7, zorder=8))
ax.add_patch(patches.Ellipse((ox+10,oy+2),18,8, color='#1a1a1a', zorder=8))
ax.add_patch(patches.Ellipse((ox+ow-10,oy+2),18,8, color='#1a1a1a', zorder=8))

# Pedestrian (right side)
px, py = 495, 182
ax.add_patch(patches.Ellipse((px,py+28),10,12, color='#f0c080', zorder=7))
ax.add_patch(patches.Rectangle((px-5,py+4),10,24, color='#3060a0', zorder=7))
ax.plot([px-5,px-12],[py+18,py+4], color='#3060a0', lw=3, zorder=7)
ax.plot([px+5,px+11],[py+18,py+6], color='#3060a0', lw=3, zorder=7)
ax.plot([px-4,px-7],[py,py-14], color='#303030', lw=3, zorder=7)
ax.plot([px+4,px+7],[py,py-14], color='#303030', lw=3, zorder=7)

# Annotation overlays
ax.add_patch(patches.Rectangle((cx-2,cy-2), cw+4, ch+4,
                                 edgecolor='#00ff40', facecolor='none', lw=1.8, zorder=10))
ax.text(cx, cy+ch+5, 'Vehicle  0.94', color='#00ff40', fontsize=6.5,
        fontweight='bold', zorder=11,
        path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

ax.add_patch(patches.Rectangle((ox-2,oy-2), ow+4, oh+4,
                                 edgecolor='#00ff40', facecolor='none', lw=1.8, zorder=10))
ax.text(ox, oy+oh+4, 'Vehicle  0.91', color='#00ff40', fontsize=6.5,
        fontweight='bold', zorder=11,
        path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

ax.add_patch(patches.Rectangle((px-9,py-16), 22, 58,
                                 edgecolor='#ffcc00', facecolor='none', lw=1.8, zorder=10))
ax.text(px-9, py-24, 'Pedestrian  0.87', color='#ffcc00', fontsize=6.5,
        fontweight='bold', zorder=11,
        path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

# Camera label
ax.text(8, 370, 'FC — Front Centre Camera  |  1920×1080  |  t = 14:32:07.412 UTC',
        color='white', fontsize=6.5, fontweight='bold', zorder=12,
        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

ax.set_facecolor('#87ceeb')
plt.tight_layout(pad=0)
fig.savefig('/home/sandbox/dataset_paper/figures/sample_camera.png',
            bbox_inches='tight', pad_inches=0.02, dpi=150)
plt.close()
print("camera done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LiDAR bird's-eye-view point cloud
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
ax.set_facecolor('#0d0d0d')
fig.patch.set_facecolor('#0d0d0d')

# Range rings
for r in [10, 20, 30, 40, 50]:
    circle = plt.Circle((0,0), r, color='#2a2a3a', fill=False, lw=0.6)
    ax.add_patch(circle)
    ax.text(r+0.5, 1, f'{r}m', color='#444466', fontsize=5.5)

# Axes
ax.axhline(0, color='#2a2a3a', lw=0.5)
ax.axvline(0, color='#2a2a3a', lw=0.5)

# ── Road surface points (grey, flat)
road_x = rng.uniform(-18, 18, 1800)
road_y = rng.uniform(-2, 50, 1800)
# keep only in road corridor (narrowing perspective)
mask = np.abs(road_x) < (4 + road_y*0.15)
road_x, road_y = road_x[mask], road_y[mask]
ax.scatter(road_x, road_y, s=0.4, c='#555566', alpha=0.55, lw=0)

# ── Building / wall clusters (left and right)
for side, xs, xe, ys, ye in [(-1,-22,-17,5,45),(1,17,22,5,45)]:
    bx = rng.uniform(xs, xe, 600)
    by = rng.uniform(ys, ye, 600)
    bz = rng.uniform(0, 4, 600)
    sc = ax.scatter(bx, by, s=0.6, c=bz, cmap='Blues', vmin=0, vmax=5, alpha=0.7, lw=0)

# ── Lead vehicle box (~25 m ahead)
def draw_vehicle_bev(ax, cx, cy, length=4.5, width=2.0, angle=0, color='#00e5ff', label=None):
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    corners = np.array([[-length/2,-width/2],[length/2,-width/2],
                         [length/2,width/2],[-length/2,width/2],[-length/2,-width/2]])
    rot = np.array([[cos_a,-sin_a],[sin_a,cos_a]])
    corners = corners @ rot.T + np.array([cx, cy])
    ax.plot(corners[:,0], corners[:,1], color=color, lw=1.5, zorder=8)
    # heading arrow
    head = np.array([0, length/2]) @ rot.T + np.array([cx, cy])
    ax.annotate('', xy=head, xytext=[cx,cy],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    # lidar returns on vehicle surface
    vx = rng.uniform(-length/2, length/2, 80)
    vy = rng.uniform(-width/2, width/2, 80)
    # only surface
    surf = (np.abs(vx)>length/2-0.4)|(np.abs(vy)>width/2-0.35)
    vx, vy = vx[surf], vy[surf]
    pts = np.column_stack([vx,vy]) @ rot.T + np.array([cx,cy])
    ax.scatter(pts[:,0], pts[:,1], s=2.5, c=color, alpha=0.85, lw=0, zorder=9)
    if label:
        ax.text(cx+width, cy, label, color=color, fontsize=6, fontweight='bold', zorder=10)

draw_vehicle_bev(ax,  1.5, 22, color='#00e5ff', label='Car')
draw_vehicle_bev(ax, -3.5, 32, angle=5,  color='#00e5ff', label='Car')
draw_vehicle_bev(ax,  4.0, 40, angle=-3, color='#00e5ff', label='Car')

# Pedestrian cluster (~18 m, right)
px_c, py_c = 6.5, 18
ped = rng.normal(0, 0.25, (60,2)) + [px_c, py_c]
ax.scatter(ped[:,0], ped[:,1], s=3, c='#ffdd00', alpha=0.9, lw=0, zorder=9)
ax.text(px_c+0.6, py_c, 'Ped', color='#ffdd00', fontsize=5.5, fontweight='bold', zorder=10)

# Ego vehicle
ego = plt.Polygon([[-1,-2.2],[1,-2.2],[1,2.2],[-1,2.2]], color='#ff6633', alpha=0.9, zorder=10)
ax.add_patch(ego)
ax.text(1.3, 0, 'Ego', color='#ff6633', fontsize=6, fontweight='bold', zorder=11)

# Ground plane random noise
gx = rng.uniform(-50, 50, 400); gy = rng.uniform(0, 50, 400)
mask_g = np.sqrt(gx**2+gy**2)<50
ax.scatter(gx[mask_g], gy[mask_g], s=0.3, c='#333344', alpha=0.4, lw=0)

ax.set_xlim(-25, 25); ax.set_ylim(-5, 52)
ax.set_xlabel('Lateral (m)', color='#aaaacc', fontsize=7)
ax.set_ylabel('Longitudinal (m)', color='#aaaacc', fontsize=7)
ax.tick_params(colors='#666688', labelsize=6)
for spine in ax.spines.values(): spine.set_edgecolor('#333344')
ax.set_title('LiDAR BEV — 64-beam Rooftop  |  t = 14:32:07.415 UTC',
             color='#ccccee', fontsize=7, pad=4)

plt.tight_layout()
fig.savefig('/home/sandbox/dataset_paper/figures/sample_lidar.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("lidar done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Radar range-velocity (Doppler) plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), dpi=150,
                          gridspec_kw={'width_ratios':[1.3,1]})
fig.patch.set_facecolor('#0d0d0d')

# ── Left: Range-Doppler map ──
ax = axes[0]
ax.set_facecolor('#0a0a14')

# Simulate range-Doppler heatmap (noise + target blobs)
R = np.linspace(0, 80, 256)
V = np.linspace(-20, 20, 256)
RR, VV = np.meshgrid(R, V)
noise = rng.exponential(0.15, RR.shape)

def blob(r0, v0, ar=3, av=1.5, amp=1.0):
    return amp * np.exp(-((RR-r0)**2/(2*ar**2) + (VV-v0)**2/(2*av**2)))

hmap = (noise
        + blob(22, -8.5, 2.5, 1.2, 4.5)    # lead car, closing
        + blob(35, -6.0, 2.0, 1.0, 3.8)    # car further ahead
        + blob(18,  9.0, 1.5, 0.8, 2.5)    # oncoming
        + blob(12,  0.2, 1.0, 0.5, 1.8)    # near static
        + blob(50, -5.5, 2.0, 1.0, 2.2))

ax.pcolormesh(R, V, np.log1p(hmap), cmap='inferno', shading='gouraud', zorder=2)

# Target annotations
targets = [(22,-8.5,'Car\n-8.5 m/s'),(35,-6.0,'Car\n-6.0 m/s'),
           (18, 9.0,'Oncoming\n+9.0 m/s')]
for r0,v0,lbl in targets:
    ax.plot(r0, v0, 'o', ms=5, mfc='none', mec='#00ff88', mew=1.4, zorder=6)
    ax.text(r0+1.5, v0+0.8, lbl, color='#00ff88', fontsize=5.5,
            fontweight='bold', zorder=7)

ax.axhline(0, color='#4444aa', lw=0.8, ls='--', zorder=4)
ax.text(1, 0.5, 'Static', color='#4444aa', fontsize=5.5, zorder=5)
ax.set_xlabel('Range (m)', color='#aaaacc', fontsize=7)
ax.set_ylabel('Radial Velocity (m/s)', color='#aaaacc', fontsize=7)
ax.set_title('Front-Centre Radar\nRange–Doppler Map', color='#ccccee', fontsize=7, pad=3)
ax.tick_params(colors='#666688', labelsize=5.5)
for sp in ax.spines.values(): sp.set_edgecolor('#333366')

# ── Right: Radar BEV detections ──
ax2 = axes[1]
ax2.set_facecolor('#0a0a14')

# Range rings
for r in [20, 40, 60]:
    c = plt.Circle((0,0), r, color='#222233', fill=False, lw=0.5)
    ax2.add_patch(c)
    ax2.text(r+0.5, 0.5, f'{r}m', color='#333355', fontsize=5)

# Detections with velocity vectors
det = [(0.5,22,-8.5,'#00e5ff','Car'),(-2.0,35,-6.0,'#00e5ff','Car'),
       (-5.0,18, 9.0,'#ff4444','Oncoming'),( 0.1,12, 0.2,'#888899','Static')]
for dx, dr, dv, dc, lbl in det:
    ax2.scatter([dx],[dr], s=40, c=dc, marker='D', zorder=8, lw=0)
    # velocity vector (scale 0.5 s ahead)
    vx_disp = 0; vy_disp = dv * 0.5
    ax2.annotate('', xy=(dx+vx_disp, dr-vy_disp), xytext=(dx,dr),
                 arrowprops=dict(arrowstyle='->', color=dc, lw=1.2), zorder=9)
    ax2.text(dx+0.8, dr, lbl, color=dc, fontsize=5.5, fontweight='bold', zorder=9)

# Ego
ax2.scatter([0],[0], s=60, c='#ff6633', marker='^', zorder=10)
ax2.text(0.8, 0.8, 'Ego', color='#ff6633', fontsize=6, fontweight='bold')

# FOV cone
theta = np.linspace(-np.radians(30), np.radians(30), 60)
cone_x = 70*np.sin(theta); cone_y = 70*np.cos(theta)
ax2.fill(np.concatenate([[0],cone_x,[0]]),
          np.concatenate([[0],cone_y,[0]]),
          color='#001133', alpha=0.4, zorder=1)
ax2.plot(np.concatenate([[0],cone_x,[0]]),
          np.concatenate([[0],cone_y,[0]]),
          color='#003366', lw=0.7, zorder=2)

ax2.set_xlim(-15,15); ax2.set_ylim(-5,70)
ax2.set_xlabel('Lateral (m)', color='#aaaacc', fontsize=7)
ax2.set_ylabel('Longitudinal (m)', color='#aaaacc', fontsize=7)
ax2.set_title('Radar BEV Detections\n+ Velocity Vectors', color='#ccccee', fontsize=7, pad=3)
ax2.tick_params(colors='#666688', labelsize=5.5)
for sp in ax2.spines.values(): sp.set_edgecolor('#333366')

plt.tight_layout(pad=0.8)
fig.savefig('/home/sandbox/dataset_paper/figures/sample_radar.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("radar done")

print("All sample figures saved.")
