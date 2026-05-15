import streamlit as st
import torch
import numpy as np
import tifffile
import gc
import tempfile
import time
from pathlib import Path
from models import TopoPreservingUNet3D
from postprocessing import postprocess_v11

PATCH_SIZE = (192,192,192)
FEATURES = [32,64,128,256,320,320]
N_BLOCKS = [1,2,3,4,6,6]

st.set_page_config(page_title='Vesuvius 3D Surface Detection',layout = 'wide')

@st.cache_resource
def load_model(checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TopoPreservingUNet3D(features = FEATURES,n_blocks=N_BLOCKS)
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path,map_location= device,weights_only = False)
        # Remove DataParallel or Module prefixes if they exist
        state = {
            k.replace('_orig_mod.', '').replace('module.',''):v for k,v in ckpt['model_state_dict'].items()
        }
    model.to(device)
    model.eval()
    if device=='cuda':
        model.half()
    return model,device    


def robust_zscore_normalize(img,lower_percentile=0.5,upper_percentile=99.5):
    p_low = np.percentile(img,lower_percentile)
    p_high = np.percentile(img,upper_percentile)
    img_clipped = np.clip(img,p_low,p_high)
    mean = img_clipped.mean()
    std = img_clipped.std()
    img_norm = (img_clipped-mean)/(std+1e-8)
    return img_norm.astype(np.float32)


def create_gaussian_weight(patch_size,sigma=0.125):
    d,h,w = patch_size
    gz = np.exp(-0.5*((np.arange(d)-d/2)/(d*sigma))**2)
    gy = np.exp(-0.5*((np.arange(h)-h/2)/(h*sigma))**2)
    gx = np.exp(-0.5*((np.arange(w)-w/2)/(w*sigma))**2)
    return (gz[:,None,None]*gy[None,:,None]*gx[None,None,:]).astype(np.float32)

def get_patch_positions(volume_shape,patch_size,overlap=0.5):
    D,H,W = volume_shape
    pd,ph,pw = patch_size
    sd,sh,sw = int(pd*(1-overlap)),int(ph*(1-overlap)),int(pw*(1-overlap))

    z_pos = list(range(0,max(1,D-pd+1),sd))
    if len(z_pos) == 0 or z_pos[-1] + pd < D: z_pos.append(max(0, D - pd))
    y_pos = list(range(0, max(1, H-ph+1), sh))
    if len(y_pos) == 0 or y_pos[-1] + ph < H: y_pos.append(max(0, H - ph))
    x_pos = list(range(0, max(1, W-pw+1), sw))
    if len(x_pos) == 0 or x_pos[-1] + pw < W: x_pos.append(max(0, W - pw))
    
    return [(z, y, x) for z in z_pos for y in y_pos for x in x_pos]

@torch.no_grad()
def sliding_window_inference(model,device,volume,patch_size,overlap=0.5,batch_size=2,progress_bar = None):
    D,H,W = volume.shape
    pd,ph,pw  = patch_size

    # pad volume if it's smaller than the patch size
    pad_d ,pad_h , pad_w = max(0,pd-D),max(0,ph-H),max(0,pw-W)
    if pad_d > 0 or pad_h>0 or pad_w >0:
        volume = np.pad(volume,((0,pad_d),(0,pad_h),(0,pad_w)),mode='reflect')
        D,H,W = volume.shape
    pred_sum = np.zeros((D,H,W),dtype = np.float32)
    weight_sum = np.zeros((D,H,W),dtype = np.float32)
    gauss = create_gaussian_weight(patch_size)
    positions = get_patch_positions((D,H,W),patch_size,overlap)

    # Normalize exactly as V11
    vol_norm = robust_zscore_normalize(volume)
    total_patches = len(positions)
    for batch_start in range(0,total_patches,batch_size):
        batch_end = min(batch_start+batch_size,total_patches)
        batch_positions = positions[batch_start:batch_end]

        patches = [vol_norm[z:z+pd,y:y+ph,x:x+pw] for (z,y,x) in batch_positions]
        batch_tensor = torch.from_numpy(np.stack(patches)[:,None]).to(device)
        if device == 'cuda':
            batch_tensor = batch_tensor.half()
        with torch.autocast(device_type=device,dtype=torch.float16 if device=='cuda' else torch.float32):
            batch_pred = torch.sigmoid(model(batch_tensor))
        batch_pred = batch_pred.squeeze(1).float().cpu().numpy()

        for i, (z, y, x) in enumerate(batch_positions):
            pred_sum[z:z+pd, y:y+ph, x:x+pw] += batch_pred[i] * gauss
            weight_sum[z:z+pd, y:y+ph, x:x+pw] += gauss
            
        if progress_bar is not None:
            progress_bar.progress(batch_end / total_patches, text=f"Inference: {batch_end}/{total_patches} patches processed")
            
        del batch_tensor, batch_pred, patches
        if device == 'cuda':
            torch.cuda.empty_cache()
            
    pred = pred_sum / np.maximum(weight_sum, 1e-8)
    
    # Remove padding to return to original shape
    if pad_d > 0: pred = pred[:-pad_d]
    if pad_h > 0: pred = pred[:, :-pad_h]
    if pad_w > 0: pred = pred[:, :, :-pad_w]
    
    return pred


def main():
    st.title("🌋 Vesuvius 3D Surface Detection")
    st.markdown('Surface Detection using TopoPreservingUnet3D model')

    st.sidebar.header('Settings')
    checkpoint_path = st.sidebar.text_input('checkpoint path',value= 'checkpoints_v11/fold_0/best_model.pth')
    overlap = st.sidebar.slider("Patch Overlap",min_value = 0.1,max_value=0.9,value=0.7,step=0.1)
    batch_size = st.sidebar.number_input('Batch Size',min_value=1,max_value=16,value=2)
    threshold = st.sidebar.slider('Post-processing Threshold',min_value=0.1,max_value=0.9,value=0.5,step=0.05)
    use_tta = st.sidebar.checkbox('Use Test time Augmentation (Flip)',value=False)

    # Load model
    with st.spinner('Loading Model...'):
        model,device = load_model(checkpoint_path)
    
    if Path(checkpoint_path).exists():
        st.sidebar.success(f"Model loaded successfully on {device.upper()}")
    else:
        st.sidebar.warning(f"Using untrained weights. Checkpoint not found at {checkpoint_path}")
    
    # Main area file Uploader
    uploaded_file = st.file_uploader('Upload a 3D TIFF Volume', type=['tif', 'tiff'])
    if uploaded_file is not None:
        st.info('Volume Uploaded successfully! Click below to run the v11 inference pipeline')

        if st.button('Run inference',type ='primary'):
            with st.spinner('Loading TIFF volume...'):
                with tempfile.NamedTemporaryFile(delete = False,suffix='.tiff')as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                volume = tifffile.imread(tmp_path).astype(np.float32)
                st.write(f"**volume shape:**{volume.shape}")

            # Step 1: Sliding Window inference
            st.write("### 1. Model Inference")
            progress_bar = st.progress(0,text='Starting sliding window inference')

            start_time = time.time()
            pred_prob = sliding_window_inference(model,device,volume,PATCH_SIZE,overlap,batch_size,progress_bar)

            if use_tta:
                st.write('Running TTA (Z-axis Flip)...')
                progress_bar_tta = st.progress(0,text = 'Staritng TTA inference')
                vol_flip = np.flip(volume ,0).copy()
                pred_flip = sliding_window_inference(model,device,vol_flip,PATCH_SIZE,overlap,batch_size,progress_bar_tta)
                pred_flip = np.flip(pred_flip,0).copy()

                pred_prob = (pred_prob+pred_flip)/2.0
                del vol_flip,pred_flip
                gc.collect()
            
            inf_time = time.time()-start_time
            st.success(f'inference completed in {inf_time:.1f} seconds!')

            # Free original volume memory
            del volume 
            gc.collect()

            st.write()
            # Step 2: Post Processing
            st.write("### 2. Topology-Safe Post-processing")
            with st.spinner("Applying V11 post-processing pipeline..."):
                start_time = time.time()
                # Use the exact postprocess_v11 function from your postprocessing.py
                pred_mask = postprocess_v11(
                    pred_prob,
                    threshold=threshold,
                    min_component_size=50,
                    use_morphology=True,
                    use_hole_fill=True,
                    verbose=False
                )
                pp_time = time.time() - start_time
            st.success(f"Post-processing completed in {pp_time:.1f} seconds!")
            
            # Step 3: Visualization & Download
            st.write("### 3. Results")
            out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            tifffile.imwrite(out_tmp.name, pred_mask, compression=None)
            
            # Provide download button
            with open(out_tmp.name, "rb") as file:
                st.download_button(
                    label="⬇️ Download Predicted Mask (TIFF)",
                    data=file,
                    file_name="predicted_mask.tif",
                    mime="image/tiff"
                )
                
            # Display a preview of the middle slice
            if pred_mask.shape[0] > 0:
                mid_z = pred_mask.shape[0] // 2
                st.write(f"**Middle Slice Preview (Z={mid_z})**")
                # Scale to 0-255 for display
                st.image(pred_mask[mid_z] * 255, caption="Predicted Mask", use_container_width=True)


if __name__=='__main__':
    main()
