# %%
import subprocess
import os
import time

data_directory = '/Users/mhturner/CurrentData/krieger/20230223/'
ants_directory = '/Applications/ants-2.4.3/bin/'


# Key: file name base of reference image
# Value: list of file name bases for target images, to register to reference
moco_pairs = {'TSeries-20230223-005_channel_2': ['TSeries-20230223-006_channel_1'],
              'TSeries-20230223-007_channel_2': ['TSeries-20230223-008_channel_1'],
}

for fn_reference in moco_pairs:
    fn_ref = os.path.join(data_directory, fn_reference + '.nii')
    fn_aff_base = os.path.join(data_directory, fn_reference + '_aff')
    ref_avg = fn_aff_base + '_avg.nii'


    # Make average of reference brain
    t0 = time.time()
    cmd_make_avg = './antsMotionCorr -d 3 -a {} -o {}'.format(fn_ref, ref_avg)
    process = subprocess.Popen(cmd_make_avg.split(), stdout=subprocess.PIPE, cwd=ants_directory)
    output, error = process.communicate()
    if len(output) > 0:
        print('OUTPUT: {}'.format(output))
    if error is not None:
        print('ERROR: {}'.format(error))
    print('Created average reference brain ({:.1f} sec)'.format(time.time()-t0))
    print('-------------------')

    # Motion correct reference brain
    t0 = time.time()
    cmd_moco_ref = './antsMotionCorr  -d 3 -o [ {}, {}.nii.gz, {} ] -m GC[ {}, {}, 1 , 0, Random, 0.1 ] -t Affine[ 0.1 ] -u 1 -e 1 -s 1x0 -f 2x1 -i 15x3 -n 3'.format(fn_aff_base, fn_aff_base, ref_avg, ref_avg, fn_ref)
    process = subprocess.Popen(cmd_moco_ref.split(), stdout=subprocess.PIPE, cwd=ants_directory)
    output, error = process.communicate()
    if len(output) > 0:
        print('OUTPUT: {}'.format(output))
    if error is not None:
        print('ERROR: {}'.format(error))
    print('Motion corrected reference brain {} ({:.1f} sec)'.format(fn_ref, time.time()-t0))
    print('-------------------')

    fn_targets = moco_pairs.get(fn_reference)
    for fn_target in fn_targets:
        # # Motion correct target brain
        t0 = time.time()
        fn_targ = os.path.join(data_directory, fn_target + '.nii')
        fn_aff_targ_base = os.path.join(data_directory, fn_target + '_aff')
        targ_avg = fn_aff_targ_base + '_avg.nii'

        cmd_moco_targ = './antsMotionCorr  -d 3 -o [ {}, {}.nii.gz, {} ] -m GC[ {}, {}, 1 , 0, Random, 0.1 ] -t Affine[ 0.1 ] -u 1 -e 1 -s 1x0 -f 2x1 -i 15x3 -n 3'.format(fn_aff_targ_base, fn_aff_targ_base, targ_avg, ref_avg, fn_targ)
        process = subprocess.Popen(cmd_moco_targ.split(), stdout=subprocess.PIPE, cwd=ants_directory)
        output, error = process.communicate()
        if len(output) > 0:
            print('OUTPUT: {}'.format(output))
        if error is not None:
            print('ERROR: {}'.format(error))
        print('Motion corrected target brain {} ({:.1f} sec)'.format(fn_targ, time.time()-t0))
        print('-------------------')

    print('###############################')
# %%
