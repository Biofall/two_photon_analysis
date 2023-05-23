
import subprocess
import os
import time
import nibabel as nib
import numpy as np

imports_dir = '/oak/stanford/groups/trc/data/krave/bruker_data/imports'
date_dir = '20230522'

data_directory = os.path.join(imports_dir, date_dir)

# Key: file name base of reference image
# Value: list of file name bases for target images, to register to reference
# For non-paired moco, put the key and an empty list for value
moco_pairs = {'TSeries-20230522-001_channel_1': [],
              'TSeries-20230522-002_channel_1': ['TSeries-20230522-004_channel_1', 'TSeries-20230522-005_channel_1'],
              'TSeries-20230522-005_channel_1': [],
              'TSeries-20230522-006_channel_1': ['TSeries-20230522-007_channel_1', 'TSeries-20230522-008_channel_1'],
              'TSeries-20230522-011_channel_1': [],
              'TSeries-20230522-012_channel_1': ['TSeries-20230522-013_channel_1', 'TSeries-20230522-014_channel_1'],
             }

# moco_pairs = {
#               'TSeries-20230327-001_channel_1': ['TSeries-20230327-002_channel_1', 'TSeries-20230327-003_channel_1'],
#               'TSeries-20230327-004_channel_1': ['TSeries-20230327-005_channel_1', 'TSeries-20230327-006_channel_1'],
#               'TSeries-20230327-007_channel_1': ['TSeries-20230327-008_channel_1', 'TSeries-20230327-009_channel_1'],
#               'TSeries-20230327-010_channel_1': ['TSeries-20230327-011_channel_1', 'TSeries-20230327-012_channel_1'],
#               'TSeries-20230327-013_channel_1': ['TSeries-20230327-014_channel_1'],
#               }


def saveAs16Bit(filepath_base):

    t0 = time.time()
    filepath = filepath_base + '.nii.gz'
    # Load brain and covert to 16bit
    brain = np.asanyarray(nib.load(filepath).dataobj).astype('uint16')

    # save as 16bit
    save_path = filepath_base + '_16.nii'
    nib.save(nib.Nifti1Image(brain, np.eye(4)), save_path)

    print('Saved as 16bit {} ({:.1f} sec)'.format(filepath_base, time.time()-t0))
    print('-------------------')


for fn_reference in moco_pairs:
    ref_subdir = fn_reference.split('_channel')[0]
    fn_ref = os.path.join(data_directory, ref_subdir, fn_reference + '.nii')
    fn_aff_base = os.path.join(data_directory, ref_subdir, fn_reference + '_aff')
    ref_avg = fn_aff_base + '_avg.nii'


    # Make average of reference brain
    t0 = time.time()
    cmd_make_avg = 'antsMotionCorr -d 3 -a {} -o {}'.format(fn_ref, ref_avg)
    process = subprocess.Popen(cmd_make_avg.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if len(output) > 0:
        print('OUTPUT: {}'.format(output))
    if error is not None:
        print('ERROR: {}'.format(error))
        raise Exception('ERROR in moco for reference brain {}'.format(fn_ref))
    process.wait()
    print('Created average reference brain {} ({:.1f} sec)'.format(fn_ref, time.time()-t0))
    print('-------------------')
    process.terminate()

    # Motion correct reference brain
    t0 = time.time()
    # MI[$t1brain,$template,1,32,Regular,0.25]
    # GC[ {}, {}, 1 , 0, Random, 0.1 ]
    # GC[ {}, {}, 1 , 1, Random, 0.1 ]
    cmd_moco_ref = 'antsMotionCorr  -d 3 -o [ {}, {}.nii.gz, {} ] -m GC[ {}, {}, 1 , 1, Random, 0.1 ] -t Rigid[ 0.005 ] -u 0 -e 1 -s 0 -f 1 -i 30 -n 30'.format(fn_aff_base, fn_aff_base, ref_avg, ref_avg, fn_ref)

    process = subprocess.Popen(cmd_moco_ref.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if len(output) > 0:
        print('OUTPUT: {}'.format(output))
    if error is not None:
        print('ERROR: {}'.format(error))
        raise Exception('ERROR in moco for reference brain {}'.format(fn_ref))
    process.wait()
    print('Motion corrected reference brain {} ({:.1f} sec)'.format(fn_ref, time.time()-t0))
    print('-------------------')
    process.terminate()

    # Save moco'ed reference brain as 16 bit
    saveAs16Bit(filepath_base=os.path.join(data_directory, ref_subdir, fn_aff_base))

    fn_targets = moco_pairs.get(fn_reference)
    for fn_target in fn_targets:
        # # Motion correct target brain
        t0 = time.time()
        targ_subdir = fn_target.split('_channel')[0]
        fn_targ = os.path.join(data_directory, targ_subdir, fn_target + '.nii')
        fn_aff_targ_base = os.path.join(data_directory, targ_subdir, fn_target + '_aff')
        targ_avg = fn_aff_targ_base + '_avg.nii'
        cmd_moco_targ = 'antsMotionCorr  -d 3 -o [ {}, {}.nii.gz, {} ] -m GC[ {}, {}, 1 , 1, Random, 0.1 ] -t Rigid[ 0.005 ] -u 0 -e 1 -s 0 -f 1 -i 30 -n 50'.format(fn_aff_targ_base, fn_aff_targ_base, targ_avg, ref_avg, fn_targ)

        process = subprocess.Popen(cmd_moco_targ.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if len(output) > 0:
            print('OUTPUT: {}'.format(output))
        if error is not None:
            print('ERROR: {}'.format(error))
            raise Exception('ERROR in moco for target brain {}'.format(fn_targ))
        process.wait()
        print('Motion corrected target brain {} ({:.1f} sec)'.format(fn_targ, time.time()-t0))
        print('-------------------')
        process.terminate()

        # Save moco'ed target brain as 16 bit
        saveAs16Bit(filepath_base=os.path.join(data_directory, targ_subdir, fn_aff_targ_base))

    print('####### DONE WITH REF {} #######'.format(fn_reference))

print('######### DONE WITH ALL MOCO ######################')
