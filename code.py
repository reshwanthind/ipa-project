import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter


input_folder = 'images/'
output_folder = 'output/'
os.makedirs(output_folder, exist_ok=True)

def get_shadow_data():
    all_intensity_data = []
    shadow_ranges = []
    images = []

    for i in range(1, 8):
        img_path = os.path.join(input_folder, f'L_H_{i}.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: {img_path} not found.")
            continue
            
        images.append(img)
        h, w = img.shape
        centroids = []
        
        for thresh in range(256):
            # Eq. 5: Inverse binary thresholding [cite: 211]
            _, b_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(b_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                best_cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(best_cnt)
                if M["m00"] > 0:
                    centroids.append((M["m10"]/M["m00"]/w, M["m01"]/M["m00"]/h))
                else:
                    centroids.append((0, 0))
            else:
                centroids.append((0, 0))
        
        all_intensity_data.append(centroids)
        
        non_zero = [idx for idx, val in enumerate(centroids) if val != (0, 0)]
        if non_zero:
            shadow_ranges.append(non_zero[-1] - non_zero[0])
        else:
            shadow_ranges.append(0)

    return np.array(all_intensity_data), np.array(shadow_ranges), images

data, s_ranges, raw_images = get_shadow_data()

#  3D Plot of Centroids 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.plasma(np.linspace(0, 1, 7))
for i in range(len(data)):
    valid_idx = np.where(np.any(data[i] != 0, axis=1))[0]
    pts = data[i][valid_idx]
    ax.scatter(pts[:, 0], pts[:, 1], valid_idx, color=colors[i], s=5, label=f'L_H_{i+1}')
ax.set_xlabel('Norm X'); ax.set_ylabel('Norm Y'); ax.set_zlabel('Threshold')
plt.legend(); plt.savefig(f'{output_folder}3d_plot_centroids.png')

# Overlayed Results
plt.figure(figsize=(5, 8))
blues = plt.cm.Blues(np.linspace(0.4, 1, 7))
for i in range(len(data)):
    pts = data[i][np.any(data[i] != 0, axis=1)]
    plt.scatter(pts[:, 0], pts[:, 1], color=blues[i], s=3, alpha=0.6)
plt.gca().invert_yaxis(); plt.title("L_H Centroid Overlay (Different Intensities)")
plt.savefig(f'{output_folder}centroid_overlay_LH.png')

# Centroid overlay grid 
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()
colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
for i in range(len(data)):
    ax = axes[i]
    img = raw_images[i]
    ax.imshow(img, cmap='gray', origin='upper')
    pts = data[i][np.any(data[i] != 0, axis=1)]
    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0] * img.shape[1], pts[:, 1] * img.shape[0],
                   s=10, c=[colors[i]] * len(pts), alpha=0.8)
    ax.set_title(f'L_H_{i+1}', fontsize=10)
    ax.axis('off')
for j in range(len(data), len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig(f'{output_folder}centroids_overlay_grid.png')
plt.close(fig)

# Denoising and Spline Fitting
flat_data = data.reshape(-1, 2)
valid = flat_data[np.any(flat_data != 0, axis=1)]
valid = valid[valid[:, 1].argsort()] # Sort by Y-coordinate

_, unique_indices = np.unique(valid, axis=0, return_index=True)
valid = valid[np.sort(unique_indices)]

# Smoothing 
x_smooth = savgol_filter(valid[:, 0], 51, 3)
y_smooth = savgol_filter(valid[:, 1], 51, 3)

# Fit B-spline 
tck, u_vals = splprep([x_smooth, y_smooth], s=0.01)
curve = splev(np.linspace(0, 1, 100), tck)

# Find point of change S_u/p 
der = np.gradient(curve[1]) / (np.gradient(curve[0]) + 1e-9)
idx_up = np.argmax(np.abs(der))

plt.figure(figsize=(5, 8))
plt.plot(curve[0], curve[1], 'b-', linewidth=2)
plt.axhline(curve[1][idx_up], color='k', linestyle='--')
plt.axvline(curve[0][idx_up], color='k', linestyle='--')
plt.gca().invert_yaxis(); plt.savefig(f'{output_folder}fitted_curve_LH.png')

# Metrics Output 
Uh = 1 - curve[1][idx_up] # Eq. 11: Umbra Height
Pw = abs(curve[0][idx_up] - curve[0][-1]) # Eq. 12: Penumbra Width
H_val = Pw / Uh # Eq. 13: Harshness

# Mock Geometric values
Uh_proj, Pw_proj, H_proj = 0.36, 0.11, 0.31

print(f"\n--- L_H MEASURED METRICS ---")
print(f"Umbra Height (Uh): {Uh:.4f}\nPenumbra Width (Pw): {Pw:.4f}\nHarshness (H): {H_val:.4f}")
print(f"\n--- L_H GEOMETRICAL PROJECTION ---")
print(f"Umbra Height: {Uh_proj}\nPenumbra Width: {Pw_proj}\nHarshness: {H_proj}")

#  Normalized Brightness Ranges 
plt.figure()
source_br = np.linspace(0.1, 1.0, 7) # Mock normalized intensity
plt.plot(range(1, 8), source_br, 'ko--', label='L_H brightness')
plt.plot(range(1, 8), s_ranges/s_ranges.max(), 'bo--', label='shadow range')
plt.legend(); plt.savefig(f'{output_folder}normalized_ranges_LH.png')

# Geometrical Projection Simulation 
geo_sim = np.ones((400, 200), dtype=np.uint8) * 255
cv2.fillPoly(geo_sim, [np.array([[180, 400], [200, 400], [200, 50], [190, 50]])], 0) # Umbra
cv2.fillPoly(geo_sim, [np.array([[140, 400], [180, 400], [190, 50], [170, 50]])], 150) # Penumbra
cv2.imwrite(f'{output_folder}geometrical_projection_13c.png', geo_sim)

print("\nAll images saved in 'output/' folder.")