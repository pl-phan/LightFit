import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
from numpy import pi
import pandas as pd
from astropy.io import fits
from scipy.optimize import least_squares
import plotly.graph_objects as go

from scipy.sparse import csr_matrix


_AUTO = False
_AUTO_FITS = './data/2026-07-20_35342_1997GZ24'
_AUTO_STARS = {
    'REF': {
        1: ['602', '836'],
        2: ['1616', '463'],
    },
    'GUIDE': {
        'UCAC4 453-106307': ['678', '980'],
        'UCAC4 453-106397': ['388', '946'],
        'UCAC4 452-105972': ['604', '835'],
        'UCAC4 452-105938': ['622', '647'],
        'UCAC4 452-106011': ['419', '624'],
        'UCAC4 452-106130': [ '20', '546'],
    },
    'GUIDED': {
        'UCAC4 452-106000': ['373', '432'],
        'UCAC4 452-105869': ['762', '478'],
        'UCAC4 453-106236': ['864', '876'],
        'UCAC4 453-106465': ['182','1123'],
    }
}

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# [1] IMPORT DATA

def import_data() -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Imports FITS frames or loads previous cached data.
    Extracts 'DATE-OBS', 'DATE-AVG', and 'DATE-END' timestamps.
    """
    instr = '[1] --> Enter .FITS files directory'
    instr += '\n\t(leave blank to load previous data):\n'
    if _AUTO:
        fits_path = _AUTO_FITS
    else:
        fits_path = input(instr).strip()

    # If loading previous cached data
    if not fits_path:
        if os.path.exists('frames.npy') and os.path.exists('times.csv'):
            print('[1] ... Loading data from frames.npy and times.csv...')
            frames = np.load('frames.npy')
            times = pd.read_csv('times.csv')
            return frames, times
        raise FileNotFoundError('Previous data not found (frames.npy, times.csv)')

    # If reading new FITS directory
    fits_files = [f for f in Path(fits_path).iterdir() if f.suffix.lower() == '.fits']
    if not fits_files:
        raise FileNotFoundError(f'No .fits files in directory: {fits_path}')

    frames = []
    time_records = []

    for filepath in sorted(fits_files):
        with fits.open(filepath) as hdu:
            header = hdu[0].header
            time_records.append({
                'DATE-OBS': header.get('DATE-OBS', ''),
                'DATE-AVG': header.get('DATE-AVG', ''),
                'DATE-END': header.get('DATE-END', '')
            })
            frames.append(np.flip(hdu[0].data, axis=0).T)

    print('[1] ... Saving data in frames.npy and times.csv')
    frames = np.stack(frames)
    times = pd.DataFrame(time_records)

    np.save('frames.npy', frames)
    times.to_csv('times.csv', index=False)

    return frames, times


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# [2] ESTIMATE DRIFT

def show_image(
    image: np.ndarray, 
    v_min: float = None, 
    v_max: float = None, 
    log: bool = False,
    img_title: str = '',
    offset: Tuple[int, int] = (0, 0), 
) -> None:
    """
    Displays an image using Plotly Heatmap.
    """
    W, H = image.shape
    img_disp = np.clip(image.copy().T, v_min, v_max)
    if log:
        img_disp = np.nan_to_num(np.log10(img_disp), nan=0.)
    x_index = np.arange(W) + offset[0]
    y_index = np.arange(H) + offset[1]

    fig = go.Figure()
    fig.add_heatmap(x=x_index, y=y_index, z=img_disp, colorscale='Greys_r')
    fig.update_layout(
        title=img_title,
        yaxis=dict(scaleanchor='x', scaleratio=1.)
    )
    fig.show()


def select_star_for_drift(
    frame_start: np.ndarray, 
    frame_end: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Displays initial and final frames and prompts for reference star positions.
    """
    if not _AUTO:
        show_image(
            frame_start, v_min=100., v_max=1000., log=False,
            img_title='FIRST Frame: Read (x, y) of reference star'
        )
    print('[2] FIRST frame is displayed : Identify reference star')
    try:
        if _AUTO:
            x0_str, y0_str = _AUTO_STARS['REF'][1][0], _AUTO_STARS['REF'][1][1]
        else:
            x0_str = input('--> FIRST Frame : Enter X-coordinate:\n').strip()
            y0_str = input('--> FIRST Frame : Enter Y-coordinate:\n').strip()
        pos_start = (float(x0_str), float(y0_str))
    except ValueError:
        raise ValueError('Invalid numerical coordinates entered')

    if not _AUTO:
        show_image(
            frame_end, v_min=100., v_max=1000., log=False,
            img_title='LAST Frame: Read (x, y) of reference star'
        )
    print('[2] LAST frame is displayed : Identify reference star')
    try:
        if _AUTO:
            xN_str, yN_str = _AUTO_STARS['REF'][2][0], _AUTO_STARS['REF'][2][1]
        else:
            xN_str = input('--> LAST Frame : Enter X-coordinate:\n').strip()
            yN_str = input('--> LAST Frame : Enter Y-coordinate:\n').strip()
        pos_end = (float(xN_str), float(yN_str))
    except ValueError:
        raise ValueError('Invalid numerical coordinates entered')

    return pos_start, pos_end


@dataclass
class Tracking:
    dx: np.ndarray      # Shape (N,)
    dy: np.ndarray      # Shape (N,)
    theta: np.ndarray   # Shape (N,) [rad]
    rot_center: Tuple[float, float] = (0.0, 0.0)

    def transform(self, x0: float, y0: float, idx: int) -> Tuple[float, float]:
        """Transforms star initial coordinates to guided coordinates at frame idx"""
        dx, dy = self.dx[idx], self.dy[idx]
        th = self.theta[idx]
        xc, yc = self.rot_center
        x_i = xc + ((x0-xc) * np.cos(th) - (y0-yc) * np.sin(th)) + dx
        y_i = yc + ((x0-xc) * np.sin(th) + (y0-yc) * np.cos(th)) + dy
        return float(x_i), float(y_i)

    def transform_all(self, x0: float, y0: float) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms star initial coordinates to guided coordinates"""
        dx, dy = self.dx, self.dy
        th = self.theta
        xc, yc = self.rot_center
        x_s = xc + ((x0-xc) * np.cos(th) - (y0-yc) * np.sin(th)) + dx
        y_s = yc + ((x0-xc) * np.sin(th) + (y0-yc) * np.cos(th)) + dy
        return x_s, y_s


def estimate_drift(
    frames: np.ndarray, 
    times: List[str], 
) -> Tracking:
    """
    Computes linear drift for a coarse Tracking, then stacks frames.
    """

    pos_start, pos_end = select_star_for_drift(frames[0], frames[-1])

    N, W, H = frames.shape
    t = pd.to_datetime(times['DATE-AVG'])
    t = ((t - t.iloc[0]) / pd.to_timedelta('1s')).to_numpy().flatten()
    z = (t - t[0]) / (t[-1] - t[0]) if t[-1] != t[0] else np.zeros_like(t)

    dx = (pos_end[0] - pos_start[0]) * z
    dy = (pos_end[1] - pos_start[1]) * z
    tracking = Tracking(
        dx=dx, dy=dy, theta=np.zeros(N),
        rot_center=(W / 2., H / 2.)
    )

    print('[2] ... Stacking frames...')
    dx = np.round(dx).astype('int')
    dy = np.round(dy).astype('int')
    offset_stack = (int(dx[0] - min(dx)), int(dy[0] - min(dy)))
    W_stack = W - (max(dx) - min(dx))
    H_stack = H - (max(dy) - min(dy))
    stack = np.zeros((W_stack, H_stack), dtype='float')
    for i in range(N):
        x0 = dx[i] - min(dx)
        y0 = dy[i] - min(dy)
        stack += frames[i, x0:x0+W_stack, y0:y0+H_stack].astype('float')
    stack /= N

    print('[2] --> Verify : Stars on stacked image should appear as dots ')
    if not _AUTO:
        show_image(stack, v_min=100, v_max=500, log=True, img_title='Stacked frames', offset=offset_stack)

    return tracking


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# [3] GUIDE STARS

@dataclass
class Star:
    name: str
    x_user: float             # X-coord in ref frame
    y_user: float             # Y-coord in ref frame
    xs: np.ndarray = None # X-coords in all frames (shape: N)
    ys: np.ndarray = None # Y-coords in all frames (shape: N)
    A: np.ndarray = None  # Amplitude in all frames (shape: N)
    B: np.ndarray = None  # Background in all frames (shape: N)
    s: float = None       # PSF width (scalar)

    def __repr__(self) -> str:
        s_str = f'{self.s:.2f}' if self.s is not None else 'None'
        return f'Star({self.name}, pos0=({self.x_user:.1f}, {self.y_user:.1f}), s={s_str})'


def select_stars(
    star_type: str
) -> List[Star]:
    """
    User enters star names and coordinates, or reads them automatically from _AUTO_STARS.
    """
    stars: List[Star] = []

    if _AUTO:
        # Direct key-value iteration over the new dict structure
        for name, coords in _AUTO_STARS[star_type].items():
            try:
                x, y = float(coords[0]), float(coords[1])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid coordinates format for star '{name}'")
            stars.append(Star(name=name, x_user=x, y_user=y))
    else:
        # Interactive prompt mode
        idx = 1
        while True:
            print(f'{star_type} Star #{idx}:')
            print('\t(press ENTER without input when finished)')
            x_str = input('--> Enter X-coordinate:\n').strip()
            if not x_str:
                break
            y_str = input('--> Enter Y-coordinate:\n').strip()
            if not y_str:
                break
            try:
                x, y = float(x_str), float(y_str)
            except ValueError:
                raise ValueError('Invalid numerical coordinates entered')
            instr = f'--> Star Name:\n\t(default: {star_type}_star_{idx})\n'
            name = input(instr).strip() or f'{star_type}_star_{idx}'
            stars.append(Star(name=name, x_user=x, y_user=y))
            idx += 1

    if not stars:
        raise ValueError(f'No stars were entered for {star_type}')

    return stars


def fit_guide_stars(
    frames: np.ndarray,
    stars: List[Star],
    tracking_init: Tracking,
    side: int = 16
) -> List[Star]:
    """
    Joint Astrometry and Photometry fit across all frames with sparse Jacobian.
    """
    N, W, H = frames.shape
    M = len(stars)

    # Coarse estimate of star positions
    for star in stars:
        star.xs, star.ys = tracking_init.transform_all(star.x_user, star.y_user)

    # Extract patches and grids
    patches, x_grids, y_grids = [], [], []
    for i in range(N):
        P_i, x_i, y_i = [], [], []
        for j, star in enumerate(stars):
            xc_p, yc_p = int(round(star.xs[i])), int(round(star.ys[i]))
            x_min, x_max = max(0, xc_p - side//2), min(W, xc_p + side//2 + 1)
            y_min, y_max = max(0, yc_p - side//2), min(H, yc_p + side//2 + 1)

            P_i.append(frames[i, x_min:x_max, y_min:y_max].astype('float'))
            gx, gy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
            x_i.append(gx)
            y_i.append(gy)

        patches.append(P_i)
        x_grids.append(x_i)
        y_grids.append(y_i)

    # Pixel offsets for sparse Jacobian
    patch_sizes = [len(patches[i][j].ravel()) for i in range(N) for j in range(M)]
    row_offsets = np.cumsum([0] + patch_sizes)

    # Build initial params, bounds and scales
    A_init, B_init = np.zeros((N, M)), np.zeros((N, M))
    xc_init, yc_init = np.zeros((N, M)), np.zeros((N, M))
    s_init = np.full(M, 1.5)

    A_min, A_max = np.zeros((N, M)), np.full((N, M), np.inf)
    B_min, B_max = np.zeros((N, M)), np.full((N, M), np.inf)
    xc_min, xc_max = np.zeros((N, M)), np.zeros((N, M))
    yc_min, yc_max = np.zeros((N, M)), np.zeros((N, M))
    s_min, s_max = np.full(M, 0.6), np.full(M, 6.)

    A_scale = np.full((N, M), 100.)
    B_scale = np.full((N, M), 10.)
    xc_scale = np.full((N, M), 1.)
    yc_scale = np.full((N, M), 1.)
    s_scale = np.full(M, 0.2)

    for i in range(N):
        for j, star in enumerate(stars):
            B_init[i, j] = float(np.quantile(patches[i][j], 0.05))
            A_init[i, j] = float(np.quantile(patches[i][j], 0.95) - B_init[i, j])
            xc_init[i, j] = float(star.xs[i])
            yc_init[i, j] = float(star.ys[i])
            xc_min[i, j], xc_max[i, j] = star.xs[i] - side//2, star.xs[i] + side//2
            yc_min[i, j], yc_max[i, j] = star.ys[i] - side//2, star.ys[i] + side//2

    p0 = np.concatenate([A_init.ravel(), B_init.ravel(), xc_init.ravel(), yc_init.ravel(), s_init])
    lower_bounds = np.concatenate([A_min.ravel(), B_min.ravel(), xc_min.ravel(), yc_min.ravel(), s_min])
    upper_bounds = np.concatenate([A_max.ravel(), B_max.ravel(), xc_max.ravel(), yc_max.ravel(), s_max])
    p_scale = np.concatenate([A_scale.ravel(), B_scale.ravel(), xc_scale.ravel(), yc_scale.ravel(), s_scale])

    def unpack_params(params: np.ndarray):
        As = params[0*N*M : 1*N*M].reshape(N, M)
        Bs = params[1*N*M : 2*N*M].reshape(N, M)
        xcs = params[2*N*M : 3*N*M].reshape(N, M)
        ycs = params[3*N*M : 4*N*M].reshape(N, M)
        ss = params[4*N*M :]
        return As, Bs, xcs, ycs, ss

    def residuals(params: np.ndarray) -> np.ndarray:
        As, Bs, xcs, ycs, ss = unpack_params(params)
        res = []
        for i in range(N):
            for j in range(M):
                A, B, xc, yc, s = As[i, j], Bs[i, j], xcs[i, j], ycs[i, j], ss[j]
                gx, gy = x_grids[i][j], y_grids[i][j]
                model = B + A * np.exp(-((gx - xc)**2 + (gy - yc)**2) / (2. * s**2))
                res.append((model - patches[i][j]).ravel())
        return np.concatenate(res)

    def jacobian(params: np.ndarray) -> csr_matrix:
        As, Bs, xcs, ycs, ss = unpack_params(params)
        rows, cols, data = [], [], []
        patch_idx = 0
        for i in range(N):
            for j in range(M):
                A, xc, yc, s = As[i, j], xcs[i, j], ycs[i, j], ss[j]
                gx = x_grids[i][j].ravel()
                gy = y_grids[i][j].ravel()

                dx, dy = gx - xc, gy - yc
                r_sq = dx**2 + dy**2
                G = np.exp(-r_sq / (2. * s**2))

                # Derivative vectors
                dA = G
                dB = np.ones_like(G)
                dxc = A * G * (dx / (s**2))
                dyc = A * G * (dy / (s**2))
                ds = A * G * (r_sq / (s**3))

                # Indices in global parameter vector
                idx_A = i * M + j
                idx_B = N*M + idx_A
                idx_xc = 2*N*M + idx_A
                idx_yc = 3*N*M + idx_A
                idx_s = 4*N*M + j

                r_start = row_offsets[patch_idx]
                r_end = row_offsets[patch_idx + 1]
                r_indices = np.arange(r_start, r_end)

                # Store sparse non-zero elements
                col_indices = [idx_A, idx_B, idx_xc, idx_yc, idx_s]
                deriv_vectors = [dA, dB, dxc, dyc, ds]

                # Append non-zero derivative entries
                for col_idx, d_val in zip(col_indices, deriv_vectors):
                    rows.append(r_indices)
                    cols.append(np.full_like(r_indices, col_idx))
                    data.append(d_val)

                patch_idx += 1

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        return csr_matrix((data, (rows, cols)), shape=(sum(patch_sizes), 4*N*M + M))

    # Optimization with sparse Jacobian
    res = least_squares(
        residuals, p0, x_scale=p_scale,
        jac=jacobian, bounds=(lower_bounds, upper_bounds),
        xtol=1e-4, ftol=1e-8, gtol=1e-8,
        verbose=2
    )
    A_opt, B_opt, xc_opt, yc_opt, s_opt = unpack_params(res.x)

    for j, star in enumerate(stars):
        star.A = A_opt[:, j]
        star.B = B_opt[:, j]
        star.xs = xc_opt[:, j]
        star.ys = yc_opt[:, j]
        star.s = float(s_opt[j])

    return stars


def visualize_star_fit(
    frames: np.ndarray,
    star: Star,
    tracking_init: Tracking,
    side: int = 16,
    save: List[str] = []
) -> None:
    """
    Diagnostic visualizer for a fitted Star across all frames
    """
    N, W, H = frames.shape

    # Extract Patches
    patches, x_grids, y_grids = [], [], []
    xs_init, ys_init = tracking_init.transform_all(star.x_user, star.y_user)
    for i in range(N):
        xc_p, yc_p = int(round(xs_init[i])), int(round(ys_init[i]))
        x_min, x_max = max(0, xc_p - side//2), min(W, xc_p + side//2 + 1)
        y_min, y_max = max(0, yc_p - side//2), min(H, yc_p + side//2 + 1)
        patches.append(frames[i, x_min:x_max, y_min:y_max].astype('float'))
        gx, gy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
        x_grids.append(gx)
        y_grids.append(gy)
    
    all_patch_pixels = np.concatenate([p.ravel() for p in patches])
    v_min_hm, v_max_hm = 0., float(np.quantile(all_patch_pixels, 0.99))
    v_min_sc, v_max_sc = 0., 1.6 * float(np.max(star.A))

    # Setup Canvas Grid Geometry
    n_cols = int(np.ceil(np.sqrt(N)))
    n_rows = int(np.ceil(N / n_cols))
    gap = 2
    cell_w, cell_h = 2 * (side//2) + 1, 2 * (side//2) + 1
    pair_w = 2 * cell_w + gap

    fig = go.Figure()

    circ_dx = star.s * 2.15 * np.cos(np.linspace(0., 2. * pi, num=21))
    circ_dy = star.s * 2.15 * np.sin(np.linspace(0., 2. * pi, num=21))
    r_fit = np.linspace(-side / 2., side / 2., num=41)
    r_scale = cell_w / side

    # Assemble Mosaic Traces
    for i in range(N):
        row_idx, col_idx = i // n_cols, i % n_cols
        x_offset_hm = col_idx * pair_w
        x_offset_sc = x_offset_hm + cell_w + gap
        y_offset = (n_rows - 1 - row_idx) * (cell_h + gap)

        # --- LEFT: HEATMAP PATCH ---
        hm_x = np.arange(patches[i].shape[0]) + x_offset_hm
        hm_y = np.arange(patches[i].shape[1]) + y_offset
        fig.add_heatmap(
            x=hm_x, y=hm_y, z=np.clip(patches[i], v_min_hm, v_max_hm).T,
            zmin=v_min_hm, zmax=v_max_hm,
            colorscale='Greys_r', showscale=False,
        )
        xc_local = star.xs[i] - x_grids[i][0, 0]
        yc_local = star.ys[i] - y_grids[i][0, 0]
        fig.add_scatter(
            x=x_offset_hm + xc_local + circ_dx, 
            y=y_offset + yc_local + circ_dy,
            mode='lines', line=dict(color='#d9381e', width=1.5),
            hoverinfo='skip', showlegend=False
        )

        # --- RIGHT: RADIAL INTENSITY ---
        dx = x_grids[i] - star.xs[i]
        dy = y_grids[i] - star.ys[i]
        r_data = (np.sqrt(dx**2 + dy**2) * np.sign(dx)).ravel()
        I_data = np.clip(patches[i].ravel(), v_min_sc, v_max_sc)
        fig.add_scatter(
            x=x_offset_sc + (cell_w / 2.) + (r_data * r_scale),
            y=y_offset + ((I_data - v_min_sc) / (v_max_sc - v_min_sc)) * cell_h,
            mode='markers', marker=dict(size=2.5, color='rgba(30, 41, 59, 0.45)'),
            hoverinfo='skip', showlegend=False
        )
        I_fit = star.B[i] + star.A[i] * np.exp(-(r_fit**2) / (2. * (star.s**2)))
        I_fit = np.clip(I_fit, v_min_sc, v_max_sc)
        fig.add_scatter(
            x=x_offset_sc + (cell_w / 2.) + (r_fit * r_scale),
            y=y_offset + ((I_fit - v_min_sc) / (v_max_sc - v_min_sc)) * cell_h,
            mode='lines', line=dict(color='#4F46e5', width=2.),
            hoverinfo='skip', showlegend=False
        )

    # Layout Adjustments
    fig.update_layout(
        title=f'{star.name} ({N} frames)',
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', scaleratio=1.),
        margin=dict(l=10., r=10., t=45., b=10.),
        plot_bgcolor='white', paper_bgcolor='white',
    )

    if 'html' in save:
        html_name = f'./outputs/{star.name}_fit.html'.replace(' ', '_')
        fig.write_html(html_name)
        print(f'\tSaved : {html_name}')
    if 'png' in save:
        png_name = f'./outputs/{star.name}_fit.png'.replace(' ', '_')
        fig.write_image(png_name, width=6000, height=3000)
        print(f'\tSaved : {png_name}')
    if 'show' in save:
        fig.show()


def fit_frame_transformations(
    guide_stars: List[Star],
    coarse_tracking: Tracking,
) -> Tuple[Tracking, List[Star]]:
    """
    Fits rigid frame transformations (dx, dy, theta) relative to Frame 0 and 
    refines global guide star coordinates (x0, y0) jointly across all frames.
    """
    N = len(guide_stars[0].xs)
    M = len(guide_stars)
    assert M >= 2, 'At least 2 guide stars required to constrain transformations'
    xc, yc = coarse_tracking.rot_center

    # Observed centroids (N, M, 2)
    x_obs = np.column_stack([star.xs for star in guide_stars])  # (N, M)
    y_obs = np.column_stack([star.ys for star in guide_stars])  # (N, M)

    # Initial estimates
    dx_init = coarse_tracking.dx[1:]
    dy_init = coarse_tracking.dy[1:]
    th_init = np.zeros(N-1)
    x0_init = np.array([star.x_user for star in guide_stars])
    y0_init = np.array([star.y_user for star in guide_stars])

    p0 = np.concatenate([dx_init, dy_init, th_init, x0_init, y0_init])

    # Parameter scaling vector
    dx_scale = np.ones(N - 1)
    dy_scale = np.ones(N - 1)
    th_scale = np.full(N - 1, 1e-4)  # Radians (~0.006 degrees)
    x0_scale = np.ones(M)
    y0_scale = np.ones(M)

    p_scale = np.concatenate([dx_scale, dy_scale, th_scale, x0_scale, y0_scale])

    def unpack_params(params: np.ndarray):
        dx = params[0 : N-1]
        dy = params[N-1 : 2*(N-1)]
        th = params[2*(N-1) : 3*(N-1)]

        dx = np.insert(dx, 0, 0.)
        dy = np.insert(dy, 0, 0.)
        th = np.insert(th, 0, 0.)

        x0 = params[3*(N-1) : 3*(N-1) + M]
        y0 = params[3*(N-1) + M :]

        return dx, dy, th, x0, y0

    def residuals(params: np.ndarray) -> np.ndarray:
        dx, dy, th, x0, y0 = unpack_params(params)

        cos_th = np.cos(th)[:, None]
        sin_th = np.sin(th)[:, None]
        x0_g = x0[None, :]
        y0_g = y0[None, :]

        x_pred = xc + ((x0_g - xc) * cos_th - (y0_g - yc) * sin_th) + dx[:, None]
        y_pred = yc + ((x0_g - xc) * sin_th + (y0_g - yc) * cos_th) + dy[:, None]

        res_x = x_pred - x_obs
        res_y = y_pred - y_obs

        return np.concatenate([res_x.ravel(), res_y.ravel()])

    def jacobian(params: np.ndarray) -> csr_matrix:
        dx, dy, th, x0, y0 = unpack_params(params)
        rows, cols, data = [], [], []

        cos_th, sin_th = np.cos(th), np.sin(th)

        for i in range(N):
            row_x_base = i * M
            row_y_base = N * M + i * M

            for j in range(M):
                rx_row = row_x_base + j
                ry_row = row_y_base + j

                # Frame parameter derivatives (only for i >= 1)
                if i > 0:
                    idx_dx = i - 1
                    idx_dy = (N - 1) + (i - 1)
                    idx_th = 2 * (N - 1) + (i - 1)

                    # dr_x / d(dx), d(th)
                    rows.extend([rx_row, rx_row])
                    cols.extend([idx_dx, idx_th])
                    data.extend([1., -(x0[j]-xc) * sin_th[i] - (y0[j]-yc) * cos_th[i]])

                    # dr_y / d(dy), d(th)
                    rows.extend([ry_row, ry_row])
                    cols.extend([idx_dy, idx_th])
                    data.extend([1., (x0[j]-xc) * cos_th[i] - (y0[j]-yc) * sin_th[i]])

                # Star position derivatives (for all i, including i = 0)
                idx_x0 = 3 * (N - 1) + j
                idx_y0 = 3 * (N - 1) + M + j

                # dr_x / d(x0), d(y0)
                rows.extend([rx_row, rx_row])
                cols.extend([idx_x0, idx_y0])
                data.extend([cos_th[i], -sin_th[i]])

                # dr_y / d(x0), d(y0)
                rows.extend([ry_row, ry_row])
                cols.extend([idx_x0, idx_y0])
                data.extend([sin_th[i], cos_th[i]])

        n_rows = 2 * N * M
        n_cols = 3 * (N - 1) + 2 * M
        return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Optimization with sparse Jacobian
    res = least_squares(
        residuals, p0, x_scale=p_scale,
        jac=jacobian, xtol=1e-6, ftol=1e-8, gtol=1e-8,
        verbose=2
    )
    dx_opt, dy_opt, th_opt, x0_opt, y0_opt = unpack_params(res.x)

    tracking = Tracking(
        dx=dx_opt, dy=dy_opt, theta=th_opt, 
        rot_center=(xc, yc)
    )

    # for j, star in enumerate(guide_stars):
    #     star.xs, star.ys = tracking.transform_all(float(x0_opt[j]), float(y0_opt[j]))

    return tracking


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# [4] GUIDED STAR FITTING

def fit_guided_stars(
    frames: np.ndarray,
    stars: List[Star],
    tracking: Tracking,
    side: int = 16
) -> List[Star]:
    """
    Photometry fit across all frames with sparse Jacobian.
    """
    N, W, H = frames.shape
    M = len(stars)
    xc_rot, yc_rot = tracking.rot_center

    # Coarse estimate of star centroids across all frames
    for star in stars:
        star.xs, star.ys = tracking.transform_all(star.x_user, star.y_user)

    # Extract patches and pixel coordinate grids
    patches, x_grids, y_grids = [], [], []
    for i in range(N):
        P_i, x_i, y_i = [], [], []
        for j, star in enumerate(stars):
            xc_p, yc_p = int(round(star.xs[i])), int(round(star.ys[i]))
            x_min, x_max = max(0, xc_p - side//2), min(W, xc_p + side//2 + 1)
            y_min, y_max = max(0, yc_p - side//2), min(H, yc_p + side//2 + 1)

            P_i.append(frames[i, x_min:x_max, y_min:y_max].astype('float'))
            gx, gy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
            x_i.append(gx)
            y_i.append(gy)

        patches.append(P_i)
        x_grids.append(x_i)
        y_grids.append(y_i)

    # Pixel offsets for sparse Jacobian construction
    patch_sizes = [len(patches[i][j].ravel()) for i in range(N) for j in range(M)]
    row_offsets = np.cumsum([0] + patch_sizes)

    # Build initial params, bounds and scales
    A_init, B_init = np.zeros((N, M)), np.zeros((N, M))
    s_init = np.full(M, 1.5)
    x0_init = np.array([star.x_user for star in stars])
    y0_init = np.array([star.y_user for star in stars])

    A_min, A_max = np.zeros((N, M)), np.full((N, M), np.inf)
    B_min, B_max = np.zeros((N, M)), np.full((N, M), np.inf)
    s_min, s_max = np.full(M, 0.6), np.full(M, 6.)
    x0_min, x0_max = x0_init - side//2, x0_init + side//2
    y0_min, y0_max = y0_init - side//2, y0_init + side//2

    A_scale = np.full((N, M), 100.)
    B_scale = np.full((N, M), 10.)
    s_scale = np.full(M, 0.2)
    x0_scale = np.ones(M)
    y0_scale = np.ones(M)

    for i in range(N):
        for j in range(M):
            B_init[i, j] = float(np.quantile(patches[i][j], 0.05))
            A_init[i, j] = float(np.quantile(patches[i][j], 0.95) - B_init[i, j])

    p0 = np.concatenate([A_init.ravel(), B_init.ravel(), s_init, x0_init, y0_init])
    lower_bounds = np.concatenate([A_min.ravel(), B_min.ravel(), s_min, x0_min, y0_min])
    upper_bounds = np.concatenate([A_max.ravel(), B_max.ravel(), s_max, x0_max, y0_max])
    p_scale = np.concatenate([A_scale.ravel(), B_scale.ravel(), s_scale, x0_scale, y0_scale])

    def unpack_params(params: np.ndarray):
        As = params[0*N*M : 1*N*M].reshape(N, M)
        Bs = params[1*N*M : 2*N*M].reshape(N, M)
        ss = params[2*N*M : 2*N*M + M]
        x0s = params[2*N*M + M : 2*N*M + 2*M]
        y0s = params[2*N*M + 2*M :]
        return As, Bs, ss, x0s, y0s

    def compute_frame_centroids(x0s: np.ndarray, y0s: np.ndarray):
        """Maps base coordinates (x0, y0) to frame-by-frame (xc, yc) using rigid tracking."""
        x0_g, y0_g = x0s[None, :], y0s[None, :]  # Shape (1, M)
        c_th, s_th = np.cos(tracking.theta)[:, None], np.sin(tracking.theta)[:, None]  # Shape (N, 1)
        dx_g, dy_g = tracking.dx[:, None], tracking.dy[:, None]  # Shape (N, 1)

        xcs = xc_rot + ((x0_g - xc_rot) * c_th - (y0_g - yc_rot) * s_th) + dx_g
        ycs = yc_rot + ((x0_g - xc_rot) * s_th + (y0_g - yc_rot) * c_th) + dy_g
        return xcs, ycs  # Shapes (N, M)

    def residuals(params: np.ndarray) -> np.ndarray:
        As, Bs, ss, x0s, y0s = unpack_params(params)
        xcs, ycs = compute_frame_centroids(x0s, y0s)
        res = []
        for i in range(N):
            for j in range(M):
                A, B, s = As[i, j], Bs[i, j], ss[j]
                xc, yc = xcs[i, j], ycs[i, j]
                gx, gy = x_grids[i][j], y_grids[i][j]
                model = B + A * np.exp(-((gx - xc)**2 + (gy - yc)**2) / (2. * s**2))
                res.append((model - patches[i][j]).ravel())
        return np.concatenate(res)

    def jacobian(params: np.ndarray) -> csr_matrix:
        As, Bs, ss, x0s, y0s = unpack_params(params)
        xcs, ycs = compute_frame_centroids(x0s, y0s)
        rows, cols, data = [], [], []
        patch_idx = 0
        for i in range(N):
            c_th_i, s_th_i = np.cos(tracking.theta[i]), np.sin(tracking.theta[i])
            for j in range(M):
                A, s = As[i, j], ss[j]
                xc, yc = xcs[i, j], ycs[i, j]
                gx = x_grids[i][j].ravel()
                gy = y_grids[i][j].ravel()

                dx, dy = gx - xc, gy - yc
                r_sq = dx**2 + dy**2
                G = np.exp(-r_sq / (2.0 * s**2))

                # Derivative vectors
                dA = G
                dB = np.ones_like(G)
                ds = A * G * (r_sq / (s**3))
                dM_dxc = A * G * (dx / (s**2))
                dM_dyc = A * G * (dy / (s**2))
                dx0 = dM_dxc * c_th_i + dM_dyc * s_th_i
                dy0 = -dM_dxc * s_th_i + dM_dyc * c_th_i

                # Global parameter column indices
                idx_A = i * M + j
                idx_B = N*M + idx_A
                idx_s = 2*N*M + j
                idx_x0 = 2*N*M + M + j
                idx_y0 = 2*N*M + 2*M + j

                r_start = row_offsets[patch_idx]
                r_end = row_offsets[patch_idx + 1]
                r_indices = np.arange(r_start, r_end)

                # Store sparse non-zero elements
                col_indices = [idx_A, idx_B, idx_s, idx_x0, idx_y0]
                deriv_vectors = [dA, dB, ds, dx0, dy0]

                for col_idx, d_val in zip(col_indices, deriv_vectors):
                    rows.append(r_indices)
                    cols.append(np.full_like(r_indices, col_idx))
                    data.append(d_val)

                patch_idx += 1

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        return csr_matrix((data, (rows, cols)), shape=(sum(patch_sizes), 2*N*M + 3*M))

    # Optimization with sparse Jacobian
    res = least_squares(
        residuals, p0, x_scale=p_scale,
        jac=jacobian, bounds=(lower_bounds, upper_bounds),
        xtol=1e-4, ftol=1e-8, gtol=1e-8,
        verbose=2
    )
    A_opt, B_opt, s_opt, x0_opt, y0_opt = unpack_params(res.x)
    xc_opt, yc_opt = compute_frame_centroids(x0_opt, y0_opt)

    for j, star in enumerate(stars):
        star.A = A_opt[:, j]
        star.B = B_opt[:, j]
        star.s = float(s_opt[j])
        star.xs, star.ys = tracking.transform_all(float(x0_opt[j]), float(y0_opt[j]))

    return stars


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# [5] LIGHTCURVES EXTRACTION & SAVING

def compute_lightcurves(
    stars: List[Star],
    times: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes integrated flux lightcurves for all stars.
    Flux = 2 * pi * A * s^2
    Visualizes exposures as time intervals (DATE-OBS to DATE-END) with midpoint markers (DATE-AVG).
    """
    data = times[['DATE-OBS', 'DATE-AVG', 'DATE-END']].copy()
    colors = [
        '#b5cf6b', '#17becf', '#bd9e39', '#ff7f0e', '#8c564b', '#636363', '#1f77b4', '#3182bd', '#9467bd', '#e377c2',
        '#2ca02c', '#e6550d', '#756bb1', '#008080', '#31a354', '#ad494a', '#bcbd22', '#a55194', '#8c6d31', '#d62728'
    ]

    fig = go.Figure()
    for idx, star in enumerate(stars):
        flux = 2. * pi * star.A * (star.s**2)
        data[star.name] = flux
        color = colors[idx % len(colors)]

        # Interleave OBS and END timestamps with flux to draw horizontal exposure segments
        xs, ys = [], []
        for t_obs, t_end, f in zip(times['DATE-OBS'], times['DATE-END'], flux):
            xs.extend([t_obs, t_end])
            ys.extend([f, f])
        fig.add_scatter(
            x=xs, y=ys, mode='lines', line=dict(color=color, width=1.5),
            name=star.name, legendgroup=star.name, showlegend=True
        )

        # Exposure Midpoints (DATE-AVG)
        fig.add_scatter(
            x=times['DATE-AVG'], y=flux, mode='markers', marker=dict(color=color, size=5),
            name=star.name, legendgroup=star.name, showlegend=False
        )

    fig.update_layout(
        title='Extracted Lightcurves',
        xaxis_title='Time',
        yaxis_title='Flux',
        template='plotly_white'
    )
    fig.show()

    return data


def save_results(df_lightcurves: pd.DataFrame) -> None:
    """
    Saves lightcurves to CSV.
    """
    output_path = './outputs/lightcurves.csv'
    df_lightcurves.to_csv(output_path, index=False)
    print(f'[5] ... Lightcurves saved to {output_path}')


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# MAIN PIPELINE

if __name__ == '__main__':
    # ==== ==== ==== ==== ==== ==== ==== ====
    print('\n[1] IMPORT DATA')

    frames, times = import_data()

    # ==== ==== ==== ==== ==== ==== ==== ====
    print('\n[2] ESTIMATE DRIFT')

    coarse_tracking = estimate_drift(frames, times)

    # ==== ==== ==== ==== ==== ==== ==== ====
    print('\n[3] GUIDE STARS')

    print('[3] Identify guide stars in stacked image')
    guide_stars = select_stars('GUIDE')
    print('[3] ... Fitting astrometry and photometry of guide stars...')
    guide_stars = fit_guide_stars(frames, guide_stars, coarse_tracking)
    print('[3] ... Fitting refined tracking...')
    refined_tracking = fit_frame_transformations(guide_stars, coarse_tracking)
    
    # ==== ==== ==== ==== ==== ==== ==== ====
    print('\n[4] GUIDED STARS')

    print('[4] Identify guided stars in stacked image')
    guided_stars = select_stars('GUIDED')
    print('[4] ... Fitting photometry of guided stars...')
    guided_stars = fit_guided_stars(frames, guided_stars, refined_tracking)

    # ==== ==== ==== ==== ==== ==== ==== ====
    print('\n[5] LIGHTCURVES')
    all_stars = guide_stars + guided_stars
    lightcurves = compute_lightcurves(all_stars, times)
    save_results(lightcurves)

    for star in guide_stars:
        visualize_star_fit(frames, star, coarse_tracking, save=['png'])
    for star in guided_stars:
        visualize_star_fit(frames, star, refined_tracking, save=['png'])
