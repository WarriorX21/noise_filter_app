import streamlit as st
import cv2
import numpy as np
import scipy.ndimage as nd
from scipy.signal import wiener
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
from skimage.filters import frangi, gabor
from skimage.morphology import white_tophat, black_tophat, disk
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os 

st.set_page_config(layout="wide", page_title="Advanced Noise Filter & Preprocessing Studio")

# -------------------------------------------------------------------------
# Custom Implementations for specialized filters not natively built-in
# -------------------------------------------------------------------------

def lee_filter(img, size=5, cu=0.25):
    """Implements the classical MMSE Lee Filter for multiplicative speckle noise."""
    img_f = img.astype(np.float64)
    img_mean = nd.uniform_filter(img_f, size)
    img_sqr_mean = nd.uniform_filter(img_f**2, size)
    img_var = img_sqr_mean - img_mean**2
    
    # Calculate weights
    overall_var = img_var + (img_mean * cu)**2
    # Avoid division by zero
    overall_var[overall_var == 0] = 1e-5
    w = img_var / overall_var
    
    out = img_mean + w * (img_f - img_mean)
    return np.clip(out, 0, 255).astype(np.uint8)

def frost_filter(img, size=5, k=0.1):
    """Implements the exponentially-weighted adaptive Frost Filter."""
    img_f = img.astype(np.float64)
    pad = size // 2
    padded = np.pad(img_f, pad, mode='reflect')
    out = np.zeros_like(img_f)
    
    # Pre-calculated distance matrix for the window kernels
    x, y = np.mgrid[-pad:pad+1, -pad:pad+1]
    dist = np.sqrt(x**2 + y**2)
    
    for i in range(img_f.shape[0]):
        for j in range(img_f.shape[1]):
            window = padded[i:i+size, j:j+size]
            mean = np.mean(window)
            var = np.var(window)
            cu = var / (mean**2) if mean != 0 else 0
            
            # Kernel weights formula
            w = np.exp(-k * cu * dist)
            w /= np.sum(w) if np.sum(w) != 0 else 1
            out[i, j] = np.sum(window * w)
            
    return np.clip(out, 0, 255).astype(np.uint8)

def anisotropic_diffusion(img, iterations=10, k=20, dt=0.14):
    """Perona-Malik Anisotropic Diffusion implementation."""
    im = img.astype(np.float64)
    for _ in range(iterations):
        # Calculate gradients
        grad_n = np.roll(im, -1, axis=0) - im
        grad_s = np.roll(im, 1, axis=0) - im
        grad_e = np.roll(im, -1, axis=1) - im
        grad_w = np.roll(im, 1, axis=1) - im
        
        # Conductance equations
        c_n = np.exp(-(grad_n / k)**2)
        c_s = np.exp(-(grad_s / k)**2)
        c_e = np.exp(-(grad_e / k)**2)
        c_w = np.exp(-(grad_w / k)**2)
        
        im += dt * (c_n*grad_n + c_s*grad_s + c_e*grad_e + c_w*grad_w)
    return np.clip(im, 0, 255).astype(np.uint8)

def fft_lowpass(img, cutoff=30, order=2, filter_type='Gaussian'):
    """Performs Frequency-Domain filtering using FFT."""
    dft = np.fft.fft2(img.astype(np.float64))
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Generate distance matrix
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    d = np.sqrt(x**2 + y**2)
    
    if filter_type == 'Ideal':
        mask = np.zeros((rows, cols))
        mask[d <= cutoff] = 1
    elif filter_type == 'Butterworth':
        mask = 1 / (1 + (d / cutoff)**(2 * order))
    else:  # Gaussian
        mask = np.exp(-(d**2) / (2 * (cutoff**2)))
        
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

def homomorphic_filter(img, cutoff=30, yl=0.5, yh=1.5, c=1.0):
    """Separates illumination from reflectance components via the log-frequency domain."""
    img_log = np.log1p(img.astype(np.float64))
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    d = np.sqrt(x**2 + y**2)
    
    # Transfer function setup
    mask = (yh - yl) * (1 - np.exp(-c * (d**2) / (cutoff**2))) + yl
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_exp = np.expm1(np.abs(img_back))
    return np.clip(img_exp, 0, 255).astype(np.uint8)

# -------------------------------------------------------------------------
# Filter Logic Switcher Mapping
# -------------------------------------------------------------------------

def apply_filter(img, filter_name, params):
    # Ensure working with Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    if filter_name == "Mean (Box) Filter":
        k = params['Kernel Size']
        return cv2.blur(img, (k, k))
    elif filter_name == "Gaussian Blur":
        k = params['Kernel Size']
        if k % 2 == 0: k += 1
        return cv2.GaussianBlur(img, (k, k), params['Sigma'])
    elif filter_name == "Wiener Filter":
        k = params['Kernel Size']
        if k % 2 == 0: k += 1
        
        # Suppress the divide-by-zero warnings for completely flat regions
        with np.errstate(divide='ignore', invalid='ignore'):
            out = wiener(img.astype(np.float64), (k, k))
            # Safely catch any NaNs produced by the zero-division and set them to 0
            out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
            
        return np.clip(out, 0, 255).astype(np.uint8)
    elif filter_name == "Median Filter":
        k = params['Kernel Size']
        if k % 2 == 0: k += 1
        return cv2.medianBlur(img, k)
    elif filter_name == "Bilateral Filter":
        return cv2.bilateralFilter(img, params['Diameter'], params['Sigma Color'], params['Sigma Space'])
    elif filter_name == "Non-Local Means (NLM)":
        return cv2.fastNlMeansDenoising(img, None, params['Filter Strength'], 7, 21)
    elif filter_name == "Anisotropic Diffusion":
        return anisotropic_diffusion(img, params['Iterations'], params['Edge-Stopping K'], 0.14)
    elif filter_name == "FFT Lowpass (Gaussian)":
        return fft_lowpass(img, params['Cutoff Frequency'], filter_type='Gaussian')
    elif filter_name == "FFT Lowpass (Butterworth)":
        return fft_lowpass(img, params['Cutoff Frequency'], params['Order n'], filter_type='Butterworth')
    elif filter_name == "FFT Lowpass (Ideal)":
        return fft_lowpass(img, params['Cutoff Frequency'], filter_type='Ideal')
    elif filter_name == "Homomorphic Filter":
        return homomorphic_filter(img, params['Cutoff Frequency'], params['Gamma Low'], params['Gamma High'])
    elif filter_name == "Lee Filter (SAR)":
        return lee_filter(img, params['Window Size'], params['Noise Coeff Var'])
    elif filter_name == "Frost Filter (SAR)":
        return frost_filter(img, params['Window Size'], params['Damping Factor K'])
    elif filter_name == "Wavelet Thresholding":
        mode = 'soft' if params['Soft Thresholding'] else 'hard'
        denoised = denoise_wavelet(img, method='BayesShrink', mode=mode, convert2ycbcr=False)
        return (denoised * 255).astype(np.uint8)
    elif filter_name == "Total Variation Denoising":
        denoised = denoise_tv_chambolle(img, weight=params['Weight Lambda'])
        return (denoised * 255).astype(np.uint8)
    elif filter_name == "Sobel Operator":
        k = params['Kernel Size']
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.clip(mag, 0, 255).astype(np.uint8)
    elif filter_name == "Scharr Operator":
        grad_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.clip(mag, 0, 255).astype(np.uint8)
    elif filter_name == "Laplacian of Gaussian (LoG)":
        k = params['Kernel Size']
        if k % 2 == 0: k += 1
        blur = cv2.GaussianBlur(img, (k, k), params['Sigma'])
        lap = cv2.Laplacian(blur, cv2.CV_64F)
        return np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    elif filter_name == "Canny Edge Detector":
        return cv2.Canny(img, params['Low Threshold'], params['High Threshold'])
    elif filter_name == "Morphological Bottom-Hat":
        selem = disk(params['Element Radius'])
        return black_tophat(img, selem)
    elif filter_name == "Otsu Thresholding":
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    elif filter_name == "Frangi Ridge Filter":
        # Ensure max is always greater than min to prevent numpy errors
        min_s, max_s = params['Min Scale'], params['Max Scale']
        if min_s >= max_s: 
            max_s = min_s + 1
            
        out = frangi(
            img, 
            sigmas=np.arange(min_s, max_s, 1), 
            beta=params['Beta'],       # Blobness sensitivity
            gamma=params['Gamma'],     # Structure/Contrast sensitivity 
            black_ridges=params['Black Ridges']
        )
        return (out / np.max(out) * 255).astype(np.uint8) if np.max(out) > 0 else img
    elif filter_name == "CLAHE":
        # OpenCV's CLAHE requires a grayscale image
        clahe = cv2.createCLAHE(
            clipLimit=params['Clip Limit'], 
            tileGridSize=(params['Grid Size'], params['Grid Size'])
        )
        return clahe.apply(img)
    
    return img

# -------------------------------------------------------------------------
# Sidebar UI: Pipeline Management & Dynamic Sliders
# -------------------------------------------------------------------------

st.sidebar.title("Pipeline Pipeline Configurator")

available_filters = [
    "Mean (Box) Filter", "Gaussian Blur", "Wiener Filter", "Median Filter",
    "Bilateral Filter", "Non-Local Means (NLM)", "Anisotropic Diffusion",
    "FFT Lowpass (Gaussian)", "FFT Lowpass (Butterworth)", "FFT Lowpass (Ideal)",
    "Homomorphic Filter", "Lee Filter (SAR)", "Frost Filter (SAR)",
    "Wavelet Thresholding", "Total Variation Denoising", "Sobel Operator",
    "Scharr Operator", "Laplacian of Gaussian (LoG)", "Canny Edge Detector",
    "Morphological Bottom-Hat", "Otsu Thresholding", "Frangi Ridge Filter", "CLAHE"
]

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = []

# Section to append steps
selected_filter = st.sidebar.selectbox("Choose a processing node to append:", ["-- Select --"] + available_filters)

if st.sidebar.button("Apply Filter"):
    if selected_filter != "-- Select --":
        st.session_state.pipeline.append({"name": selected_filter, "params": {}})
        # st.rerun() --> causing errors
    else:
        st.sidebar.warning("Please select a filter from the dropdown first.")

if st.sidebar.button("Clear Processing Chain"):
    st.session_state.pipeline = []
    st.rerun()

# Dynamically construct parameters for active pipeline items
active_pipeline_configs = []
st.sidebar.markdown("---")
st.sidebar.subheader("Active Pipeline Hierarchy")

for index, step in enumerate(st.session_state.pipeline):
    with st.sidebar.expander(f"Step {index+1}: {step['name']}", expanded=True):
        f_name = step['name']
        p = {}
        
        if f_name in ["Mean (Box) Filter", "Wiener Filter", "Median Filter"]:
            p['Kernel Size'] = st.slider(f"Kernel Size (px)", 3, 15, 3, step=2, key=f"k_{index}")
        elif f_name == "Gaussian Blur":
            p['Kernel Size'] = st.slider(f"Kernel Size (px)", 3, 15, 3, step=2, key=f"k_{index}")
            p['Sigma'] = st.slider(f"Standard Deviation (Sigma)", 0.5, 5.0, 1.0, step=0.1, key=f"s_{index}")
        elif f_name == "Bilateral Filter":
            p['Diameter'] = st.slider(f"Pixel Diameter", 1, 15, 5, key=f"d_{index}")
            p['Sigma Color'] = st.slider(f"Sigma Color (Range Threshold)", 10, 150, 25, key=f"sc_{index}")
            p['Sigma Space'] = st.slider(f"Sigma Space (Coordinate Distance)", 10, 150, 25, key=f"ss_{index}")
        elif f_name == "Non-Local Means (NLM)":
            p['Filter Strength'] = st.slider(f"Denoising Factor h", 1, 30, 10, key=f"h_{index}")
        elif f_name == "Anisotropic Diffusion":
            p['Iterations'] = st.slider(f"Diffusion Runs (Iterations)", 1, 30, 10, key=f"it_{index}")
            p['Edge-Stopping K'] = st.slider(f"Gradient Threshold K", 5, 100, 20, key=f"k_ad_{index}")
        elif f_name in ["FFT Lowpass (Gaussian)", "FFT Lowpass (Butterworth)", "FFT Lowpass (Ideal)", "Homomorphic Filter"]:
            p['Cutoff Frequency'] = st.slider(f"Cutoff Horizon (D0)", 5, 200, 30, key=f"co_{index}")
            if f_name == "FFT Lowpass (Butterworth)":
                p['Order n'] = st.slider(f"Filter Order n", 1, 5, 2, key=f"ord_{index}")
            if f_name == "Homomorphic Filter":
                p['Gamma Low'] = st.slider(f"Gamma Low (< 1.0)", 0.1, 0.9, 0.4, step=0.05, key=f"gl_{index}")
                p['Gamma High'] = st.slider(f"Gamma High (> 1.0)", 1.1, 2.5, 1.5, step=0.1, key=f"gh_{index}")
        elif f_name in ["Lee Filter (SAR)", "Frost Filter (SAR)"]:
            p['Window Size'] = st.slider(f"Window Horizon Size", 3, 15, 5, step=2, key=f"ws_{index}")
            if f_name == "Lee Filter (SAR)":
                p['Noise Coeff Var'] = st.slider(f"Coeff of Variation (Cu)", 0.05, 1.0, 0.25, step=0.05, key=f"cu_{index}")
            else:
                p['Damping Factor K'] = st.slider(f"Exponential Damping Factor K", 0.01, 2.0, 0.1, step=0.05, key=f"kf_{index}")
        elif f_name == "Wavelet Thresholding":
            p['Soft Thresholding'] = st.checkbox("Soft Thresholding Mode", value=True, key=f"soft_{index}")
        elif f_name == "Total Variation Denoising":
            p['Weight Lambda'] = st.slider(f"TV Regularizer (Lambda)", 0.01, 0.5, 0.1, step=0.01, key=f"lam_{index}")
        elif f_name == "Sobel Operator":
            p['Kernel Size'] = st.slider(f"Kernel Size Matrix", 3, 7, 3, step=2, key=f"sob_k_{index}")
        elif f_name == "Laplacian of Gaussian (LoG)":
            p['Kernel Size'] = st.slider(f"Smoothing Kernel Width", 3, 15, 5, step=2, key=f"log_k_{index}")
            p['Sigma'] = st.slider(f"Gaussian Distribution Width (Sigma)", 0.5, 5.0, 1.0, step=0.1, key=f"log_s_{index}")
        elif f_name == "Canny Edge Detector":
            p['Low Threshold'] = st.slider(f"Lower Hysteresis Cutoff", 10, 150, 50, key=f"can_l_{index}")
            p['High Threshold'] = st.slider(f"Upper Hysteresis Bound", 100, 250, 150, key=f"can_h_{index}")
        elif f_name == "Morphological Bottom-Hat":
            p['Element Radius'] = st.slider(f"Disk SE Radius Size (px)", 1, 20, 5, key=f"bh_r_{index}")
        elif f_name == "Frangi Ridge Filter":
            p['Min Scale'] = st.slider(f"Min Hessian Scale", 1, 5, 1, key=f"fr_min_{index}")
            p['Max Scale'] = st.slider(f"Max Hessian Scale", 2, 10, 5, key=f"fr_max_{index}")
            p['Beta'] = st.slider(f"Blobness (Beta - lower ignores blobs)", 0.1, 5.0, 0.5, step=0.1, key=f"fr_b_{index}")
            p['Gamma'] = st.slider(f"Contrast/Structure (Gamma - lower finds faint lines)", 1.0, 50.0, 15.0, step=1.0, key=f"fr_g_{index}")
            p['Black Ridges'] = st.checkbox(f"Look for Dark Cracks on Light Bg", value=True, key=f"fr_br_{index}")
        elif f_name == "CLAHE (Contrast Normalization)":
            p['Clip Limit'] = st.slider(f"Contrast Threshold (Clip Limit)", 1.0, 10.0, 2.0, step=0.5, key=f"clahe_cl_{index}")
            p['Grid Size'] = st.slider(f"Tile Grid Size", 2, 16, 8, step=2, key=f"clahe_gs_{index}")  

        if st.button("Delete Element", key=f"del_{index}"):
            st.session_state.pipeline.pop(index)
            st.rerun()
            
        active_pipeline_configs.append({"name": f_name, "params": p})

# -------------------------------------------------------------------------
# Main Execution View Setup
# -------------------------------------------------------------------------

st.title("🎛️ Digital Image Noise Filtering Engine & Preprocessing Playground")

uploaded_file = st.file_uploader("Load Target Inspection Image...", type=["png", "jpg", "jpeg"])

# --- NEW ROBUST IMAGE HANDLING ---
# 1. Store the file in persistent session state the moment it's uploaded
if uploaded_file is not None:
    st.session_state['persisted_image'] = uploaded_file.getvalue()

# 2. Load from session state memory instead of directly from the widget
if 'persisted_image' in st.session_state:
    file_bytes = np.asarray(bytearray(st.session_state['persisted_image']), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
else:
    # Build a synthesized test grid with noise targets
    original_img = np.zeros((400, 400), dtype=np.uint8) + 128
    cv2.putText(original_img, "CRACK SIGNAL 0.4MM", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 20, 2)
    cv2.line(original_img, (50, 220), (350, 240), 10, 2) 
    
    gauss = np.random.normal(0, 15, original_img.shape).astype(np.float64)
    sp_noise = np.random.rand(*original_img.shape)
    noisy_canvas = original_img.astype(np.float64) + gauss
    noisy_canvas[sp_noise < 0.02] = 0
    noisy_canvas[sp_noise > 0.98] = 255
    original_img = np.clip(noisy_canvas, 0, 255).astype(np.uint8)
    st.info("No file uploaded. Displaying synthetic verification pattern containing noise anomalies.")
# ---------------------------------

# Run execution path through pipeline matrix
processed_img = original_img.copy()
for node in active_pipeline_configs:
    processed_img = apply_filter(processed_img, node['name'], node['params'])

# Comparison display layout matrix
col1, col2 = st.columns(2)
with col1:
    st.subheader("Source Payload Frame")
    st.image(original_img, width='stretch', channels="GRAY")
with col2:
    st.subheader("Processed Engine State")
    st.image(processed_img, width='stretch', channels="GRAY")

# -------------------------------------------------------------------------
# Report Export Generation Subsystem
# -------------------------------------------------------------------------

def create_pdf(orig, proc, chain):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Image Pre-Processing Verification Record", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Pipeline Execution Trace Parameters:", ln=True)
    
    for idx, item in enumerate(chain):
        param_str = ", ".join([f"{k}: {v}" for k, v in item['params'].items()])
        pdf.cell(200, 8, txt=f" Step {idx+1}: {item['name']} -> ({param_str})", ln=True)
        
    # Write structural temporary artifacts for image inclusions
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, "orig.png")
        proc_path = os.path.join(tmpdir, "proc.png")
        cv2.imwrite(orig_path, orig)
        cv2.imwrite(proc_path, proc)
        
        pdf.ln(10)
        pdf.cell(200, 10, txt="Source Context:", ln=True)
        pdf.image(orig_path, x=10, w=90)
        
        pdf.ln(5)
        pdf.cell(200, 10, txt="Processed Output State:", ln=True)
        pdf.image(proc_path, x=10, w=90)
        
        return pdf.output()

st.markdown("---")
st.subheader("Export Pipeline Metrics Sheet")
if st.button("Generate Verification Report"):
    if len(active_pipeline_configs) == 0:
        st.warning("Cannot generate data trace reports for empty processing sequences.")
    else:
        pdf_bytes = create_pdf(original_img, processed_img, active_pipeline_configs)
        st.download_button(
            label="Download PDF Analytical Record",
            data=pdf_bytes,
            file_name="filter_pipeline_report.pdf",
            mime="application/pdf"
        )
        st.success("Report successfully generated.")