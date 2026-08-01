
## LightFit
Least-squares fitting of star tracking and gaussian PSF photometry

LightFit is an open-source pipeline designed to extract high-precision lightcurves from sequences of astronomical image frames (FITS format). It replaces traditional fixed-aperture measurement with a mathematically rigorous, simultaneous optimization approach.

By directly fitting two-dimensional shape models (point spread functions) to star profiles across all images at once—while accounting for field drift and rotation—LightFit delivers stable flux measurements even under poor tracking or crowded field conditions.

---

## Core Methodology & Architecture

The standard way to measure star brightness (aperture photometry) relies on drawing fixed circles around targets and summing pixel values inside them. This approach suffers when the telescope drifts, when images blur, or when stars are close to each other.

LightFit solves this by framing photometrical measurement as a non-linear least-squares optimization problem. Instead of measuring each image independently, it links spatial movement across time and solves for all parameters simultaneously using analytical derivatives (sparse Jacobians).

**Key Advantages:**
* **No Pixel Interpolation:** Stars are measured on the raw pixel grid without applying image warping or resampling, preserving original noise statistics.
* **Low Memory Footprint:** By using sparse matrix representation, thousands of individual frame parameters are solved at once without crashing system memory.
* **Global Rotational Tracking:** Frame movement is decoupled into a shared geometric transform, ensuring background stars anchor the alignment for dim targets.

---

## Pipeline Breakdown

The processing flow is split into five distinct, reproducible steps:

### [1] Data Ingestion (`import_data`)
* **Goal:** Load astronomical frames and normalize coordinate orientation.
* **What happens:** Reads standard FITS files, extracts average observation timestamps from the image headers, and transposes/flips the 2D arrays to ensure spatial coordinates match standard horizontal (X) and vertical (Y) axes. To save processing time on subsequent runs, loaded stacks are cached locally.

### [2] Initial Drift Estimation (`estimate_drift`)
* **Goal:** Establish a baseline path of image movement across time.
* **What happens:** Measures the pixel offset of a bright reference star between the first and last images. Assuming a linear drift rate, it calculates a preliminary frame-by-frame offset map. It then creates a preliminary "stacked" image by shifting and averaging all frames, allowing dim target stars to stand out clearly from background noise.

### [3] Guide Star Astrometry & Field Alignment (`fit_guide_stars` & `fit_frame_transformations`)
* **Goal:** Determine the exact position, size, and brightness of bright reference stars, and calculate precise field rotation and translation for every frame.
* **What happens:** 
  1. Local patches around guide stars are isolated across all images.
  2. A two-dimensional Gaussian profile is fitted to each guide star:
     **Model:** Intensity = Background + Peak Intensity * exp( - Distance^2 / (2 * Width^2) )
  3. The algorithm adjusts the brightness, local background, center position, and star width to minimize the squared difference between the model and raw pixel values.
  4. Using the precise center positions of these guide stars, a global frame transformation (horizontal shift, vertical shift, and rotation angle relative to Frame 0) is derived for every single image in the sequence.

### [4] Guided Target Fitting (`fit_guided_stars`)
* **Goal:** Measure dim target stars reliably by locking their movement to the guide star solution.
* **What happens:** For faint target stars (such as those undergoing occultation), fitting position independently per frame introduces high noise. LightFit locks the movement of target stars to the geometric rotation and translation parameters established in Step 3. The algorithm only fits the star's base reference location, overall profile width, per-frame brightness, and per-frame background.

### [5] Lightcurve Extraction (`compute_lightcurves` & `save_results`)
* **Goal:** Convert fitted profile parameters into absolute light values over time.
* **What happens:** Rather than summing noisy pixels inside a circle, the total light (integrated flux) is computed analytically from the fitted Gaussian parameters:
  **Total Light = 2 * Pi * Peak Intensity * (Width^2)**
  This delivers a clean, continuous measurement of star brightness over time. The results are displayed interactively and exported as a standardized CSV file.

---

## Mathematical Parameter Structure

For a sequence with N images and M stars, LightFit optimizes:

* **Brightness (A):** N x M independent values (one peak value per star per frame).
* **Background (B):** N x M independent values (one background value per star per frame).
* **Profile Width (s):** M shared values (one average shape parameter per star across all frames).
* **Base Position (x0, y0):** M shared reference positions linked through the N frame-transformation parameters (Shift X, Shift Y, Rotation Angle).

Because individual pixel updates only depend on local star patches, the matrix of partial derivatives (Jacobian) is highly sparse. LightFit builds this sparse structure explicitly using compressed sparse row format (`csr_matrix`), allowing rapid convergence via standard non-linear algorithms (`scipy.optimize.least_squares`).

---

## Data Requirements & File Structure

* **Input Files:** Directory containing `.fits` format images with `DATE-AVG` present in headers.
* **Intermediate Cache:** `frames.npy` (3D image array) and `times.csv` (timestamp list).
* **Primary Output:** `lightcurves.csv` containing time series columns for each fitted star.
