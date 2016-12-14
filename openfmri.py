import os
import glob
import shutil

import os.path as op
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.bem import make_flash_bem, convert_flash_mris
from mne.parallel import parallel_func
from mne.preprocessing import ICA, create_eog_epochs
from mne.preprocessing import read_ica, create_ecg_epochs
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)

from mne import io, combine_evoked
from mne.minimum_norm import make_inverse_operator, apply_inverse

mne.set_log_level('INFO')
print(__doc__)

plt.close("all")
use_ica=False
use_ssp=False
inverse_model=True
freesurfer_path='/home/jcasa/freesurfer'
study_path = '/home/jcasa/mne_data/openfmri/ds117'
subjects_dir = os.path.join(study_path, 'subjects')
os.environ["SUBJECTS_DIR"] = subjects_dir
os.environ["FREESURFER_HOME"] = freesurfer_path
meg_dir = os.path.join(study_path, 'MEG')

if not os.path.isdir(subjects_dir):
    os.mkdir(subjects_dir)
if not os.path.isdir(meg_dir):
    os.mkdir(meg_dir)
        
N_JOBS = 1			

spacing = 'oct5'
mindist = 5
startim = 4
stopim = 10
###############################################################################
# The `cross talk file <https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/results/library/ct_sparse.fif>`_
# and `calibration file <https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/results/library/sss_cal.dat>`_
# are placed in the same folder.

#ctc = os.path.join(os.path.dirname(__file__), 'ct_sparse.fif')
#cal = os.path.join(os.path.dirname(__file__), 'sss_cal.dat')

ylim = {'eeg': [-10, 10], 'mag': [-300, 300], 'grad': [-80, 80]}


events_id = {
    'face/famous/first': 5,
    'face/famous/immediate': 6,
    'face/famous/long': 7,
    'face/unfamiliar/first': 13,
    'face/unfamiliar/immediate': 14,
    'face/unfamiliar/long': 15,
    'scrambled/first': 17,
    'scrambled/immediate': 18,
    'scrambled/long': 19,
}

tmin, tmax = -0.2, 0.8
reject = dict(grad=4000e-13, mag=4e-12, eog=180e-6)
baseline = None

map_subjects = {1: 'subject_01', 2: 'subject_02', 3: 'subject_03',
                4: 'subject_05', 5: 'subject_06', 6: 'subject_08',
                7: 'subject_09', 8: 'subject_10', 9: 'subject_11',
                10: 'subject_12', 11: 'subject_14', 12: 'subject_15',
                13: 'subject_16', 14: 'subject_17', 15: 'subject_18',
                16: 'subject_19', 17: 'subject_23', 18: 'subject_24',
                19: 'subject_25'}

def process_subject_anat(subject_id):
    subject = "sub%03d" % subject_id
    print("processing %s" % subject)
    dst_flash = "%s/%s/mri/flash" % (subjects_dir, subject)
    dst_mri = "%s/%s/mri/" % (subjects_dir, subject)
    dst_T1 = "%s/%s/mri/T1" % (subjects_dir, subject)
    dst_sub = "%s/%s/" % (subjects_dir, subject)
    
    if not os.path.isdir(dst_sub):
        os.mkdir(dst_sub)
    if not os.path.isdir(dst_mri):
        os.mkdir(dst_mri)
    if not os.path.isdir(dst_T1):
        os.mkdir(dst_T1)

    if not os.path.isdir(dst_flash):
        os.mkdir(dst_flash)

#    error_log = "%s/%s/scripts/IsRunning.lh+rh" % (subjects_dir, subject)
#    if os.path.exists(error_log):
#        print("removing %s" % error_log)
#        os.remove(error_log)
         
#    t1_fname = "%s/%s/anatomy/highres001.nii.gz" % (study_path, subject)
#    log_fname = "%s/%s/my-recon-all.txt" % (study_path, subject)
#    command = "recon-all -all -s %s -i %s > %s" % (subject,t1_fname,log_fname)
#    print command
#    os.system(command)

    # Move flash data
    meflash="%s/%s/anatomy/FLASH/meflash*" % (study_path, subject)
    fnames = glob.glob(meflash)

    for f_src in fnames:
        f_dst = os.path.basename(f_src).replace("meflash_", "mef")
        f_dst = os.path.join(dst_flash, f_dst)
        print f_src, f_dst
        shutil.copy(f_src, f_dst)

    # Make flash BEM
    print "Convert_flash_mris"
    convert_flash_mris(subject, flash30=True, convert=False, unwarp=False,
                       subjects_dir=subjects_dir)
    print "Make_flash_bem"
    make_flash_bem(subject=subject, subjects_dir=subjects_dir,
                   overwrite=True, show=False)

def run_filter(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    raw_fname_out = op.join(meg_dir, subject, 'run_%02d_filt_sss_raw.fif')
    raw_fname_in = op.join(study_path, subject, 'MEG',
                           'run_%02d_sss.fif')
    for run in range(1, 7):
        raw_in = raw_fname_in % run
        try:
            raw = mne.io.read_raw_fif(raw_in, preload=True, add_eeg_ref=False)
        except AttributeError:
            # Some files on openfmri are corrupted and cannot be read.
            warn('Could not read file %s. '
                 'Skipping run %s from subject %s.' % (raw_in, run, subject))
            continue
        raw_out = raw_fname_out % run
        if not op.exists(op.join(meg_dir, subject)):
            os.mkdir(op.join(meg_dir, subject))

        raw.filter(1, 40, l_trans_bandwidth=0.5, h_trans_bandwidth='auto',
                   filter_length='auto', phase='zero', fir_window='hann')
        raw.save(raw_out, overwrite=True)

def run_events(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    data_path = op.join(meg_dir, subject)
    for run in range(1, 7):
        run_fname = op.join(data_path, 'run_%02d_filt_sss_raw.fif' % run)
        if not os.path.exists(run_fname):
            continue

        raw = mne.io.Raw(run_fname, add_eeg_ref=False)
        mask = 4096 + 256  # mask for excluding high order bits
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive='increasing', mask=mask,
                                 mask_type='not_and', min_duration=0.003,
                                 verbose=True)

        print("S %s - R %s" % (subject, run))

        fname_events = op.join(data_path, 'run_%02d_filt_sss-eve.fif' % run)
        mne.write_events(fname_events, events)
def run_ica(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    data_path = op.join(meg_dir, subject)
    for run in range(1, 7):
        print("Run: %s" % run)
        run_fname = op.join(data_path, 'run_%02d_filt_sss_raw.fif' % run)
        if not os.path.exists(run_fname):
            warn('Could not find file %s. '
                 'Skipping run %s for subject %s.' % (run_fname, run, subject))
            continue
        raw = mne.io.read_raw_fif(run_fname, add_eeg_ref=False)
        ica_name = op.join(meg_dir, subject, 'run_%02d-ica.fif' % run)

        ica = ICA(method='fastica', random_state=42, n_components=0.98)
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                               stim=False, exclude='bads')
        ica.fit(raw, picks=picks, reject=dict(grad=4000e-13, mag=4e-12),
                decim=8)
        ica.save(ica_name)

###############################################################################
# Now we define a function to extract epochs for one subject

def run_epochs(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)

    data_path = op.join(meg_dir, subject)

    all_epochs = list()

    # Get all bad channels
    mapping = map_subjects[subject_id]  # map to correct subject
    all_bads = list()
    for run in range(1, 7):
        bads = list()
        bad_name = op.join('bads', mapping, 'run_%02d_raw_tr.fif_bad' % run)
        if os.path.exists(bad_name):
            with open(bad_name) as f:
                for line in f:
                    bads.append(line.strip())
        all_bads += [bad for bad in bads if bad not in all_bads]

    for run in range(1, 7):
        print " - Run %s" % run
        run_fname = op.join(data_path, 'run_%02d_filt_sss_raw.fif' % run)
        if not os.path.exists(run_fname):
            continue

        raw = mne.io.Raw(run_fname, preload=True, add_eeg_ref=False)

        raw.set_channel_types({'EEG061': 'eog',
                               'EEG062': 'eog',
                               'EEG063': 'ecg',
                               'EEG064': 'misc'})  # EEG064 free floating el.
        raw.rename_channels({'EEG061': 'EOG061',
                             'EEG062': 'EOG062',
                             'EEG063': 'ECG063'})

        eog_events = mne.preprocessing.find_eog_events(raw)
        eog_events[:, 0] -= int(0.25 * raw.info['sfreq'])
        annotations = mne.Annotations(eog_events[:, 0] / raw.info['sfreq'],
                                      np.repeat(0.5, len(eog_events)),
                                      'BAD_blink', raw.info['meas_date'])
        raw.annotations = annotations  # Remove epochs with blinks

        delay = int(0.0345 * raw.info['sfreq'])
        events = mne.read_events(op.join(data_path,
                                         'run_%02d_filt_sss-eve.fif' % run))

        events[:, 0] = events[:, 0] + delay

        raw.info['bads'] = all_bads
        raw.interpolate_bads()
        raw.set_eeg_reference()

        picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                               eog=True)

        # Read epochs
        epochs = mne.Epochs(raw, events, events_id, tmin, tmax, proj=True,
                            picks=picks, baseline=baseline, preload=True,
                            decim=2, reject=reject, add_eeg_ref=False)

        # ICA
        ica_name = op.join(meg_dir, subject, 'run_%02d-ica.fif' % run)
        ica = read_ica(ica_name)
        n_max_ecg = 3  # use max 3 components
        ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
        ecg_inds, scores_ecg = ica.find_bads_ecg(ecg_epochs, method='ctps',
                                                 threshold=0.8)
        ica.exclude += ecg_inds[:n_max_ecg]

        ica.apply(epochs)
        all_epochs.append(epochs)

    epochs = mne.epochs.concatenate_epochs(all_epochs)
    epochs.save(op.join(data_path, '%s-epo.fif' % subject))
def run_evoked(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)

    data_path = op.join(meg_dir, subject)
    epochs = mne.read_epochs(op.join(data_path, '%s-epo.fif' % subject))

    evoked_famous = epochs['face/famous'].average()
    evoked_scrambled = epochs['scrambled'].average()
    evoked_unfamiliar = epochs['face/unfamiliar'].average()

    # Simplify comment
    evoked_famous.comment = 'famous'
    evoked_scrambled.comment = 'scrambled'
    evoked_unfamiliar.comment = 'unfamiliar'

    contrast = mne.combine_evoked([evoked_famous, evoked_unfamiliar,
                                   evoked_scrambled], weights=[0.5, 0.5, -1.])
    contrast.comment = 'contrast'
    faces = mne.combine_evoked([evoked_famous, evoked_unfamiliar], 'nave')
    faces.comment = 'faces'

    mne.evoked.write_evokeds(op.join(data_path, '%s-ave.fif' % subject),
                             [evoked_famous, evoked_scrambled,
                              evoked_unfamiliar, contrast, faces])

    # take care of noise cov
    cov = mne.compute_covariance(epochs, tmax=0, method='shrunk')
    cov.save(op.join(data_path, '%s-cov.fif' % subject))



def run_forward(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    data_path = op.join(meg_dir, subject)

    fname_ave = op.join(data_path, '%s-ave.fif' % subject)
    fname_fwd = op.join(data_path, '%s-meg-%s-fwd.fif' % (subject, spacing))
    fname_trans = op.join(study_path, 'ds117', subject, 'MEG',
                          '%s-trans.fif' % subject)

    src = mne.setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir, overwrite=True,
                                 n_jobs=1, add_dist=False)

    src_fname = op.join(subjects_dir, subject, '%s-src.fif' % spacing)
    mne.write_source_spaces(src_fname, src)

    bem_model = mne.make_bem_model(subject, ico=4, subjects_dir=subjects_dir,
                                   conductivity=(0.3,))
    bem = mne.make_bem_solution(bem_model)
    info = mne.read_evokeds(fname_ave, condition=0).info
    fwd = mne.make_forward_solution(info, trans=fname_trans, src=src, bem=bem,
                                    fname=None, meg=True, eeg=False,
                                    mindist=mindist, n_jobs=1, overwrite=True)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True)
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)

def run_inverse(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    data_path = op.join(meg_dir, subject)

    fname_ave = op.join(data_path, '%s-ave.fif' % subject)
    fname_cov = op.join(data_path, '%s-cov.fif' % subject)
    fname_fwd = op.join(data_path, '%s-meg-%s-fwd.fif' % (subject, spacing))
    fname_inv = op.join(data_path, '%s-meg-%s-inv.fif' % (subject, spacing))

    evokeds = mne.read_evokeds(fname_ave, condition=[0, 1, 2, 3, 4])
    cov = mne.read_cov(fname_cov)

    forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

    # make an M/EEG, MEG-only, and EEG-only inverse operators
    info = evokeds[0].info
    inverse_operator = make_inverse_operator(info, forward, cov, loose=0.2,
                                             depth=0.8)

    write_inverse_operator(fname_inv, inverse_operator)

    # Compute inverse solution
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    for evoked in evokeds:
        stc = apply_inverse(evoked, inverse_operator, lambda2, "dSPM",
                            pick_ori=None)

        stc.save(op.join(data_path, 'mne_dSPM_inverse-%s' % evoked.comment))


###############################################################################
# Let us make the script parallel across subjects

# 19 excluded due to pb with flash images
parallel, run_func, _ = parallel_func(process_subject_anat, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))

parallel, run_func, _ = parallel_func(run_filter, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))

parallel, run_func, _ = parallel_func(run_events, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))

parallel, run_func, _ = parallel_func(run_ica, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))

parallel, run_func, _ = parallel_func(run_epochs, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))

parallel, run_func, _ = parallel_func(run_evoked, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))

for subject_id in range(startim, stopim):
    subject = "sub%03d" % subject_id
    mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir,
                               overwrite=True)

parallel, run_func, _ = parallel_func(run_forward, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))

parallel, run_func, _ = parallel_func(run_inverse, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(startim, stopim))
