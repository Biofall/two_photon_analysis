


# %%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

filename = 'TSeries-20230302-001_channel_2_affMOCOparams.csv'
data_dir = '/Users/mhturner/CurrentData'

df = pd.read_csv(os.path.join(data_dir, filename))

# %%
fh, ax = plt.subplots(1, 1, figsize=(6, 2))
ax.plot(df['MetricPost'])
ax.set_xlabel('Frame')
ax.set_ylabel('MI')

# ax.set_xlim([0, 250])

# %%
fh, ax = plt.subplots(len(df.keys()), figsize=(8, 18))
for k_ind, kk in enumerate(df.keys()):
    ax[k_ind].plot(df[kk])
    ax[k_ind].set_title(kk)

# %%

filename = 'TSeries-20230223-006_channel_1_aff.nii.gz'
data_dir = '/Users/mhturner/CurrentData/krieger/20230223'


filepath = os.path.join(data_dir, filename)

# %%




brain = np.asanyarray(nib.load(filepath).dataobj).astype('uint16')

# save as 16bit
save_path = os.path.join(data_dir, 'TSeries-20230223-006_channel_1_aff_16.nii')

nib.save(nib.Nifti1Image(brain, np.eye(4)), save_path)

# %%

save_path = os.path.join(data_dir, 'TSeries-20230216-005_channel_1_aff_16.nii')

nib.save(nib.Nifti1Image(brain, np.eye(4)), save_path)

# %% save meanbrain

n_frames = 3000


save_path = os.path.join(data_dir, 'TSeries-20221129-001_channel_2_aff_avg.nii')

nib.save(nib.Nifti1Image(np.mean(brain[:, :, :, :n_frames], axis=-1), np.eye(4)), save_path)
# %%
