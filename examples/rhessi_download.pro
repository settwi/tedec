; see https://sprg.ssl.berkeley.edu/~tohban/wiki/index.php/Creating_a_Spectrum_File_Using_the_HESSI_GUI
; guide on how to do this

; config me
t_int = ['30-Jul-2011T01:50:00.000', '30-Jul-2011 02:20:00.000']
; see here for binning information
; https://hesperia.gsfc.nasa.gov/ssw/hessi/dbase/spec_resp/energy_binning.txt
start_energy = alog10(4)
end_energy = alog10(200)
num = 40
expons = dindgen(num) / (num - 1) * (end_energy - start_energy) + start_energy
energy_bins = [10^expons, [200, 800]]
time_bin_width = 2


name = 'trevor-flare-30-jul-2011-logspace-bkg'
specfile = name + '_spec.fits'
srmfile = name + '_srm.fits'

search_network, /enable

spec_obj = hsi_spectrum()
spec_obj-> set, obs_time_interval=t_int
spec_obj-> set, decimation_correct=1    
spec_obj-> set, rear_decimation_correct= 0    
spec_obj-> set, pileup_correct=0    
spec_obj->set, /sp_semi_calibrated

; "detectors 3, 4, and 9 have the best energy resolution of all the detectors" -- from the wiki page
; use all front segments. whatever.
spec_obj-> set, seg_index_mask=[1B, 1B, 1B, 1B, 0B, 0B, 1B, 1B, 1B, $
			                    0B, 0B, 0B, 0B, 0B, 0B, 0B, 0B, 0B] 
spec_obj-> set, sp_chan_binning=0
spec_obj-> set, sp_chan_max=0
spec_obj-> set, sp_chan_min=0
spec_obj-> set, sp_data_unit='Flux'
spec_obj-> set, sp_energy_binning=energy_bins
spec_obj-> set, sp_semi_calibrated=0B
spec_obj-> set, sp_time_interval=time_bin_width
spec_obj-> set, sum_flag=1
; spec_obj-> set, time_range=[0.0000000D, 0.0000000D]
spec_obj-> set, use_flare_xyoffset=1

spec_obj->filewrite, /buildsrm, all_simplify=0, srmfile=srmfile, specfile=specfile

exit
