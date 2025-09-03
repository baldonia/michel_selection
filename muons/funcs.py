import uproot as up
import awkward as ak
import pandas as pd
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
import time
from pathlib import Path

class Run:
    def __init__(self, file, nevents):
        self.file = Path(file) 
        self.nevents = nevents

        # Root file variables 
        self.meta_data = None
        self.output_data = None

        # Is Online
        self.pmt_is_online = None 

        # Hit cleaning masks
        self.hit_amp_cut = None
        self.hit_charge_cut = None
        self.hit_time_cut = None        

        # Hit cleaning settings
        self.time_cut = [-10, 15]
        self.amp_cut = 3

        # Hit cleaned data
        self.clean_charges = None
        self.clean_nhit = None

        # Event selection masks 
        self.prompt_raw_mask = None
        self.delayed_raw_mask = None 
        self.prompt_clean_mask = None 
        self.delayed_clean_mask = None 

        # Event selected data
        self.delayed_dt = None
        self.delayed_dt_mask = None 
        self.final_event_mask = None 
        self.final_charges = None 

        # Timing variables for plotting
        self.clock_period = 0.5  # us
        self.start_time = 1 #3 * 0.432 + self.clock_period # 0  # e.g., 0.864 us
        self.nclock_cycles_to_plot = 40
        self.stop_time = self.start_time + (self.nclock_cycles_to_plot) * self.clock_period  # e.g., 0.912

        # PMT ids for plotting  
        self.barrel_ids = None 
        self.bottom_ids = None 
        self.dichroic_ids = None 
        self.top_ids = None 
        self.behind_ids = None 

    def print_nevents_in_file(self):
        f = up.open(self.file)
        print(f"There are {f['output'].num_entries} events in {self.file}")

    def load_meta_data(self):
        f = up.open(self.file)
        print(f"Meta Keys: {f['meta'].keys()}")
        start = time.time() 
        self.meta_data = f['meta'].arrays([
            'pmtType',
            'pmtId',
            'pmtChannel',
            'pmtIsOnline',
            'pmtCableOffset',
        ], entry_start=0, entry_stop=self.nevents, library='ak') 
        stop = time.time()
        print(f"Took {stop-start:.2f} seconds to load meta data for {self.nevents} events")

    def load_event_data(self):
        f=up.open(self.file)
        print(f"Output Keys: {f['output'].keys()}")
        start = time.time()
        self.output_data = f['output'].arrays([
            'timestamp',
            'timeSinceLastTrigger_us',
            'digitPMTID',
            'digitTime',
            'digitCharge',
            'digitPeak',
            'digitHitCleaningMask',
            'x_quadfitter',
            'y_quadfitter',
            'z_quadfitter',
            'time_quadfitter',
        ], entry_start=0, entry_stop=self.nevents, library='ak')
        stop = time.time()
        print(f"Took {stop-start:.2f} seconds to load output data for {self.nevents} events")

    # step 1
    def apply_hit_cleaning(self):
        # Make hit amplitude ratio hit cut
        hitcleanmask = self.output_data['digitHitCleaningMask']
        hit_amp_cut = hitcleanmask<1
        
        # Make positive charge hit cut 
        charges = self.output_data['digitCharge']
        hit_charge_cut = charges > 0

        # Make time cut
        times = self.output_data['digitTime']
        event_times = self.output_data['time_quadfitter']
        event_times = ak.broadcast_arrays(times, event_times)[1]
        times = times - event_times
        cut = self.time_cut
        hit_time_cut = (times > cut[0]) & (times < cut[1])

        # Save important stuff
        self.hit_amp_cut = hit_amp_cut
        self.hit_charge_cut = hit_charge_cut
        self.hit_time_cut = hit_time_cut


    # step 2
    def make_nhit_masks(self):

        # Get raw data
        charges = self.output_data['digitCharge']

        # Apply hit cuts individually for diagnostics   
        amp_charges = ak.mask(charges, self.hit_amp_cut)
        charge_charges = ak.mask(charges, self.hit_charge_cut) # don't use this anymore, encapulates in amp cut
        time_charges = ak.mask(charges, self.hit_time_cut)

        # Apply all hit cuts together
        clean_charges = ak.mask(charges, self.hit_amp_cut & self.hit_time_cut)

        # Calculate NHits
        raw_nhit = ak.count(charges, axis=1)
        amp_nhit = ak.count(amp_charges, axis=1)
        time_nhit = ak.count(time_charges, axis=1)
        clean_nhit = ak.count(clean_charges, axis=1)

        # Save important stuff
        self.clean_charges = clean_charges
        self.clean_nhit = clean_nhit

        # Flatten charges for plotting
        charges = ak.to_numpy(ak.flatten(charges))
        amp_charges = ak.to_numpy(ak.flatten(amp_charges))
        time_charges = ak.to_numpy(ak.flatten(time_charges))
        clean_charges = ak.to_numpy(ak.flatten(clean_charges))

        # Make trigger type masks 
        prompt_min_nhit = 125
        delayed_min_nhit = 50
        prompt_raw_mask = raw_nhit >= prompt_min_nhit
        delayed_raw_mask = (raw_nhit >= delayed_min_nhit) & (raw_nhit < prompt_min_nhit)
        prompt_clean_mask = clean_nhit >= prompt_min_nhit
        delayed_clean_mask = (clean_nhit >= delayed_min_nhit) & (clean_nhit < prompt_min_nhit)

        # Save important stuff
        self.prompt_clean_mask = prompt_clean_mask 
        self.delayed_clean_mask = delayed_clean_mask
        self.prompt_raw_mask = prompt_raw_mask 
        self.delayed_raw_mask = delayed_clean_mask

        # For diagnostics and labeling the plot
        raw_prompt = ak.count(ak.mask(raw_nhit, prompt_raw_mask), axis=None)
        raw_delayed = ak.count(ak.mask(clean_nhit, delayed_raw_mask), axis=None)
        clean_prompt = ak.count(ak.mask(clean_nhit, prompt_clean_mask), axis=None)
        clean_delayed = ak.count(ak.mask(clean_nhit, delayed_clean_mask), axis=None)

        # Plot NHits
        max_nhit=200
        fontsize=20
        plt.figure(figsize=(10,5))
        plt.hist(raw_nhit,bins=max_nhit,range=(0,max_nhit),histtype='step',color='black',label=f"Raw data")#: {raw_prompt} prompt {raw_delayed} delayed");
        plt.hist(amp_nhit,bins=max_nhit,range=(0,max_nhit),histtype='step',color='green',label=f"After amplitude hit cut")#: {clean_prompt} prompt {clean_delayed} delayed");
        plt.hist(time_nhit,bins=max_nhit,range=(0,max_nhit),histtype='step',color='orange',label=f"After time hit cut")#: {clean_prompt} prompt {clean_delayed} delayed");
        plt.hist(clean_nhit,bins=max_nhit,range=(0,max_nhit),histtype='step',color='purple',label=f"After amplitude + time hit cut")#: {clean_prompt} prompt {clean_delayed} delayed");
        plt.title('EOS Run 3056 (Water Fill)',fontsize=fontsize)
        plt.xlabel('NHits',fontsize=fontsize)
        plt.ylabel('Number of Events',fontsize=fontsize)
        #plt.axvspan(125,200,color='blue',alpha=0.1,label='Stopping Muon Candidates')
        #plt.axvspan(50,125,color='red',alpha=0.1,label='Michel Electron Candidates')
        plt.semilogy()
        plt.legend(ncol=1,loc='upper right',bbox_to_anchor=(2.5,1))

        # Plot total charge
        charge_range = (-10,10)
        plt.figure(figsize=(10,5))
        plt.hist(charges,bins=max_nhit,range=charge_range,histtype='step',color='black',label=f"Raw data")#: {clean_prompt} prompt {clean_delayed} delayed");
        plt.hist(amp_charges,bins=max_nhit,range=charge_range,histtype='step',color='green',label=f"After amplitude hit cut")#: {clean_prompt} prompt {clean_delayed} delayed");
        plt.hist(time_charges,bins=max_nhit,range=charge_range,histtype='step',color='orange',label=f"After time hit cut")#: {clean_prompt} prompt {clean_delayed} delayed");
        plt.hist(clean_charges,bins=max_nhit,range=charge_range,histtype='step',color='purple',label=f"After amplitude + time hit cut")#: {clean_prompt} prompt {clean_delayed} delayed");
        plt.title('EOS Run 3056 (Water Fill)',fontsize=fontsize)
        plt.xlabel('Digit Charge [pC]',fontsize=fontsize)
        plt.ylabel('Number of Events',fontsize=fontsize)
        plt.semilogy()
        plt.legend(ncol=1,loc='upper right',bbox_to_anchor=(2.5,1))




    # step 3
    def compute_delayed_times_relative_to_prompt_with_ts(self):

        prompt_mask = self.prompt_clean_mask
        delayed_mask = self.delayed_clean_mask

        timestamps = self.output_data['timestamp']
        timestamps = np.asarray(timestamps)
        prompt_times = timestamps[prompt_mask]
        delayed_times = timestamps[delayed_mask]

        delta_ts = []
        i_prompt = 0

        for delayed_time in delayed_times:
            # Advance through prompt_times as long as they're in the past
            while i_prompt + 1 < len(prompt_times) and prompt_times[i_prompt + 1] < delayed_time:
                i_prompt += 1

            if prompt_times[i_prompt] < delayed_time:
                delta_ts.append(delayed_time - prompt_times[i_prompt])

            else:
                # No prior prompt found (e.g., delayed event came before all prompts)
                delta_ts.append(np.nan)

        self.delayed_dt = np.array(delta_ts)

    # step 4
    def make_deliverables(self):

        dt_clean = self.delayed_dt
        delayed_clean_mask = self.delayed_clean_mask
        clean_charges = self.clean_charges

        print(len(dt_clean),dt_clean)
        print(len(delayed_clean_mask),delayed_clean_mask)

        # Apply delay time cut (dt < 3 µs) to delayed events
        time_mask = (self.start_time < dt_clean/1e3) & (dt_clean/1e3 <= self.stop_time)   # [delayed events only]

        print(len(time_mask),time_mask)
        
        # Select only delayed events that passed the time cut
        # delayed_amp_time_mask is [all events] — we now filter that with time_mask
        final_event_mask = ak.Array(delayed_clean_mask)  # shape: [events]
        # Get indices of delayed events
        delayed_indices = ak.where(delayed_clean_mask)[0]  # np.array of indices
        # Apply time cut
        valid_delayed_indices = delayed_indices[time_mask]  # np.array of surviving delayed event indices
        
        # Now build final event-level mask
        final_event_mask = np.zeros(len(delayed_clean_mask), dtype=bool)
        final_event_mask[valid_delayed_indices] = True  # shape: [event]

        print(len(final_event_mask),final_event_mask)
        
        # Final cleaned arrays (hits within selected delayed events)
        digit_charge_final = clean_charges[final_event_mask]


        # Save important stuff
        self.delayed_dt_mask = time_mask
        self.final_event_mask = final_event_mask
        self.final_charges = digit_charge_final

        
    # Optional Plotting
    def plot_final_nhit_charges(self):
 
        digit_charge_final = self.final_charges
 
 
        print(len(digit_charge_final))
        
        # Plot number of valid hits per selected delayed event
        plt.figure(figsize=(6,6))
        plt.hist(ak.sum(digit_charge_final,axis=1),bins=50,range=(0,2000),histtype='step')
        plt.xlabel("Total Digit Charge per delayed event")
        plt.ylabel("Count")
        plt.title("Final Delayed Events After All Cuts")
        plt.yscale('log')
        plt.show()
        
        # Plot number of valid hits per selected delayed event
        plt.figure(figsize=(6,6))
        plt.hist(ak.count(digit_charge_final,axis=1),bins=50,range=(0,200),histtype='step')
        plt.xlabel("Number of hits per delayed event")
        plt.ylabel("Count")
        plt.title("Final Delayed Events After All Cuts")
        plt.yscale('log')
        plt.show()
 
    def fit_muon_lifetime(self):
 
 
        # Construct bin centers 
        bin_centers = np.arange(self.start_time, self.stop_time, self.clock_period)
        pmask = (self.delayed_dt/1e3> self.start_time) & (self.delayed_dt/1e3<= self.stop_time)
 
        # Construct bin edges from centers 
        half_width = self.clock_period / 2
        bin_edges = np.concatenate((
            [bin_centers[0] - half_width],
            bin_centers + half_width
        ))
 
        # print("Bin edges:", bin_edges)
        # print("Bin widths:", np.diff(bin_edges))
 
        dt_clean = self.delayed_dt
 
        result = self.fit_decay(dt_clean/1e3, bins=bin_edges, tau_guess=2)
        bin_centers = result['bin_centers']
        counts = result['counts']
        sigma = result['sigma']
        A, tau, B = result['popt']
        tau_err = result['perr'][1]
 
 
 
        plt.figure(figsize=(18, 10))
        plt.errorbar(bin_centers, counts, yerr=sigma, fmt='o', ms=3, color='black')#, label=f"amp + time cut (τ = {tau:.2f} ± {tau_err:.2f})")
        t_fine = np.linspace(bin_edges[0], bin_edges[-1], 1000)
 
        plt.plot(t_fine, self.decay_model(t_fine, *result['popt']), '-', color='red', lw=4, label=f"$\\tau$ = {tau:.2f} ± {tau_err:.2f}")
        plt.xlabel("$\Delta$t [µs]")
        plt.ylabel(f"Counts / {bin_edges[1] - bin_edges[0]:.3f} µs")
        plt.title(f"{len(dt_clean[pmask])} Michel Candidate Events From Eos Run 3056 (Water Fill)")
        plt.grid(True)
        plt.legend()
        #plt.tight_layout()
        #plt.xlim(0.5,10.5)
        # plt.xlim(range)
        #plt.semilogy()
        plt.show()
 
    def decay_model(self, t, A, tau, B):
        return (A / tau) * np.exp(-t / tau) + B

    def fit_decay(self, data, bins, tau_guess=2):
        # Use np.histogram with explicit bin edges
        counts, bin_edges = np.histogram(data, bins=bins)

        # Calculate bin centers
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Poisson errors, avoid division by zero
        sigma = np.sqrt(counts)
        sigma[sigma == 0] = 1.0

        # Initial parameter guesses
        A_guess = np.max(counts)
        B_guess = np.min(counts)
        p0 = [A_guess, tau_guess, B_guess]

        # Fit the decay model to the data
        popt, pcov = curve_fit(
            self.decay_model, bin_centers, counts,
            sigma=sigma, absolute_sigma=True,
            p0=p0, maxfev=100000
        )

        # Extract fit results
        A_fit, tau_fit, B_fit = popt
        A_err, tau_err, B_err = np.sqrt(np.diag(pcov))

        # Goodness of fit
        residuals = counts - self.decay_model(bin_centers, *popt)
        chi2 = np.sum((residuals / sigma) ** 2)
        dof = len(counts) - len(popt)
        chi2_red = chi2 / dof if dof > 0 else float('inf')

        return {
            "bin_centers": bin_centers,
            "counts": counts,
            "sigma": sigma,
            "popt": popt,
            "perr": [A_err, tau_err, B_err],
            "chi2": chi2,
            "chi2_red": chi2_red,
        }
    
    def plot_nhit_dt(self):

        dt_clean = self.delayed_dt
        delayed_clean_mask = self.delayed_clean_mask 
        clean_nhit = self.clean_nhit

        print(dt_clean)
        print(delayed_clean_mask)
        print(clean_nhit)

        valid_dt_clean_mask = ~np.isnan(dt_clean)

        dt_clean_np = ak.to_numpy(dt_clean[valid_dt_clean_mask])
        clean_nhit_np = ak.to_numpy(clean_nhit[delayed_clean_mask][valid_dt_clean_mask])

        clock_period = 0.016 # us
        nclock_ticks = 100
        follower_start_time = 0.432  # us
   
        plt.hist2d(dt_clean_np/1e3, clean_nhit_np,bins=(nclock_ticks,100),range=((follower_start_time,follower_start_time+clock_period*(nclock_ticks-1)),(0,200)),norm='log');
        plt.axvline(0.864)
        plt.xlabel("$\Delta$t [µs]")
        plt.ylabel('NHits')
        plt.title('Michel Candidates')
        plt.show()

        plt.hist2d(dt_clean_np/1e3, clean_nhit_np,bins=(100,100),range=((0,20),(0,200)),norm='log');
        plt.axvline(0.864)
        plt.xlabel("$\Delta$t [µs]")
        plt.ylabel('NHits')
        plt.title('Michel Candidates')
        plt.show()

        tts = np.unique(dt_clean_np)
        # for t in tts[:50]:
        #     print(t)
        # print(np.unique(np.diff(np.unique(dt_clean_np))))

    def plot_timing_cut(self):

        # Make time cut
        times = self.output_data['digitTime']
        event_times = self.output_data['time_quadfitter']
        event_times = ak.broadcast_arrays(times, event_times)[1]
        times = times - event_times
        cut = self.time_cut 
        hit_time_cut = (times > cut[0]) & (times < cut[1])
        # Apply time cut
        cut_times = ak.mask(times, hit_time_cut)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.hist(ak.flatten(times), bins=400, range=(-150, 150), histtype='step',label='Raw', color='tab:blue')
        plt.hist(ak.flatten(cut_times), bins=400, range=(-150, 150),label='Filtered', color='tab:blue')
        plt.axvspan(cut[0], cut[1], color='grey', alpha=0.2)
        plt.xlabel('Hit Time [ns]')
        plt.ylabel('Total Counts')
        plt.title('Digit Times Across All Channels and Events')
        plt.legend()
        plt.tight_layout()
        #plt.semilogy()
        #plt.ylim(1e5.1e7)
        plt.show()

    def create_michel_file(self, ofname):
        start = time.time()
        final_event_mask = self.final_event_mask
        f=up.open(self.file)
        output_tree = f['output']
        meta_tree = f['meta']
        meta_arrays = meta_tree.arrays(filter_name=['pmtX', 'pmtY', 'pmtZ', 'pmtU', 'pmtV', 'pmtW'], library='np')
        
        # Get event times
        event_times = output_tree['time_quadfitter'].array(entry_start=0, entry_stop=self.nevents, library='ak')
   
        selected_output_arrays = {}
        for key, branch in output_tree.items():
            try:
                array = branch.array(entry_start=0, entry_stop=self.nevents, library='ak')
            except ValueError:
                print(key, "was bad")
                continue
            # Correct hit times
            if key == 'digitTime':
                array = array - event_times
            #FIXME better to do hit cleaning to all hit variables
            if key in ['digitPMTID', 'digitTime', 'digitCharge']:
                array = ak.mask(array, self.hit_amp_cut & self.hit_time_cut)
                array = ak.drop_none(array)
            selected_output_arrays[key] = array[final_event_mask]
        
        with up.recreate(ofname) as out_file:
            out_file['output'] = selected_output_arrays
            out_file['meta'] = meta_arrays
            
        stop = time.time()
        print(stop-start)


class SimRun:
    def __init__(self, file, pmt_is_online, nevents):
        self.file = Path(file) 
        self.nevents = nevents
        self.pmt_is_online = pmt_is_online

        # Root file variables 
        self.meta_data = None
        self.output_data = None

    
        # Hit cleaning masks
        # self.hit_amp_cut = None
        self.hit_id_cut = None
        self.hit_charge_cut = None
        self.hit_time_cut = None        

        # Hit cleaned data
        self.clean_charges = None
        self.clean_nhit = None

        # Event selection masks 
        self.prompt_raw_mask = None
        self.delayed_raw_mask = None 
        self.prompt_clean_mask = None 
        self.delayed_clean_mask = None 

        # Event selected data
        self.delayed_dt = None
        self.delayed_dt_mask = None 
        self.final_event_mask = None 
        self.final_charges = None 

        # Timing variables for plotting
        self.clock_period = 0.016  # us
        self.start_time = 3 * 0.432 + self.clock_period # 0  # e.g., 0.864 us
        self.nclock_cycles_to_plot = 500
        self.stop_time = self.start_time + (self.nclock_cycles_to_plot) * self.clock_period  # e.g., 0.912

        # PMT ids for plotting  
        self.barrel_ids = None 
        self.bottom_ids = None 
        self.dichroic_ids = None 
        self.top_ids = None 
        self.behind_ids = None 
        
    def get_valid_out_key(self, file):
        all_out_keys = [out_key for out_key in file.keys() if out_key.startswith('output')]  # filter metas
        all_out_num = np.array([int(out_key[-1]) for out_key in all_out_keys])
        index = np.argmax(all_out_num)
        return all_out_keys[index]

    def print_nevents_in_file(self):
        f = up.open(self.file)
        print(f"There are {f['output'].num_entries} events in {self.file}")

    def load_meta_data(self):
        f = up.open(self.file)
        print(f"Meta Keys: {f['meta'].keys()}")
        start = time.time() 
        self.meta_data = f['meta'].arrays([
            'pmtType',
            'pmtId',
            'pmtChannel',
            'pmtIsOnline',
            'pmtCableOffset',
        ], entry_start=0, entry_stop=1, library='ak') 
        stop = time.time()
        print(f"Took {stop-start:.2f} seconds to load meta data for {self.nevents} events")

    def load_event_data(self):
        f=up.open(self.file)
        outkey = self.get_valid_out_key(f)
        print(f"Output Keys: {f[outkey].keys()}")
        start = time.time()
        self.output_data = f[outkey].arrays([
            'timestamp',
            'timeSinceLastTrigger_us',
            'digitPMTID',
            'digitTime',
            'digitCharge',
            'digitPeak',
            'x_quadfitter',
            'y_quadfitter',
            'z_quadfitter',
            'time_quadfitter',
        ], entry_start=0, entry_stop=self.nevents, library='ak')
        stop = time.time()
        print(f"Took {stop-start:.2f} seconds to load output data for {self.nevents} events")

    # step 1
    def apply_hit_cleaning(self):
        # # Make hit amplitude ratio hit cut
        # digit_peak = self.output_data['digitPeak']
        # digit_neg_peak = self.output_data['digitNegativePeak']
        # ratio = digit_peak / digit_neg_peak
        # hit_amp_cut = ratio > 2

        # Check that the PMT is online

        # get boolean mask array from metadata
        #pmt_is_online = np.asarray(self.meta_data['pmtIsOnline'][0])  # shape (241,), bool
        #pmt_is_online = np.asarray(self.pmt_is_online)  # shape (241,), bool

        # Step 2: get jagged array of PMT IDs
        ids = self.output_data['digitPMTID']  # shape [event][hit]

        # Step 3: apply element-wise mapping using NumPy boolean mask and `ak.Array`
        # We do this by converting to flat NumPy, indexing, and re-wrapping into the same structure
        #flat_ids = ak.flatten(ids)                                # shape [total_hits]
        #flat_mask = pmt_is_online[flat_ids]                        # shape [total_hits], bool
        #hit_id_cut = ak.unflatten(flat_mask, ak.num(ids, axis=1))  # shape [event][hit]

        # Make positive charge hit cut 
        charges = self.output_data['digitCharge']
        hit_charge_cut = charges > 0

        # Make time cut
        times = self.output_data['digitTime']
        event_times = self.output_data['time_quadfitter']
        event_times = ak.broadcast_arrays(times, event_times)[1]
        times = times - event_times
        cut = [-10, 15]
        hit_time_cut = (times > cut[0]) & (times < cut[1])

        # Save important stuff
        # self.hit_amp_cut = hit_amp_cut
        #self.hit_id_cut = hit_id_cut
        self.hit_charge_cut = hit_charge_cut
        self.hit_time_cut = hit_time_cut


    # step 2
    def make_nhit_masks(self):

        # Get raw data
        charges = self.output_data['digitCharge']

        # Apply hit cuts
        #clean_charges = ak.mask(charges, self.hit_amp_cut & self.hit_charge_cut & self.hit_time_cut)
        #clean_charges = ak.mask(charges, self.hit_id_cut & self.hit_charge_cut & self.hit_time_cut)
        clean_charges = ak.mask(charges, self.hit_charge_cut & self.hit_time_cut)

        # Calculate NHits
        raw_nhit = ak.count(charges, axis=1)
        clean_nhit = ak.count(clean_charges, axis=1)

        # Make trigger type masks 
        prompt_min_nhit = 125
        delayed_min_nhit = 50

        prompt_raw_mask = raw_nhit >= prompt_min_nhit
        delayed_raw_mask = (raw_nhit >= delayed_min_nhit) & (raw_nhit < prompt_min_nhit)

        prompt_clean_mask = clean_nhit >= prompt_min_nhit
        delayed_clean_mask = (clean_nhit >= delayed_min_nhit) & (clean_nhit < prompt_min_nhit)

        raw_prompt = ak.count(ak.mask(raw_nhit, prompt_raw_mask), axis=None)
        raw_delayed = ak.count(ak.mask(clean_nhit, delayed_raw_mask), axis=None)

        clean_prompt = ak.count(ak.mask(clean_nhit, prompt_clean_mask), axis=None)
        clean_delayed = ak.count(ak.mask(clean_nhit, delayed_clean_mask), axis=None)

        # Plot NHits
        max_nhit=200
        fontsize=20
        plt.figure(figsize=(10,5))
        plt.hist(raw_nhit,bins=max_nhit,range=(0,max_nhit),histtype='step',color='black',linestyle='--',label=f"Before: {raw_prompt} prompt {raw_delayed} delayed");
        plt.hist(clean_nhit,bins=max_nhit,range=(0,max_nhit),histtype='step',color='black',label=f"After: {clean_prompt} prompt {clean_delayed} delayed");
        plt.title('EOS Water Fill Run 3046',fontsize=fontsize)
        plt.xlabel('NHits',fontsize=fontsize)
        plt.ylabel('Number of Events',fontsize=fontsize)
        plt.axvspan(125,200,color='blue',alpha=0.3,label='Stopping Muon Candidates')
        plt.axvspan(50,125,color='red',alpha=0.3,label='Michel Electron Candidates')
        plt.semilogy()
        plt.legend(ncol=1,loc='upper center',bbox_to_anchor=(1.25,1),fontsize=15)

        # Save important stuff
        self.prompt_raw_mask = prompt_raw_mask 
        self.delayed_raw_mask = delayed_clean_mask

        self.clean_charges = clean_charges
        self.clean_nhit = clean_nhit
        self.prompt_clean_mask = prompt_clean_mask 
        self.delayed_clean_mask = delayed_clean_mask

    # step 3 
    def make_deliverables(self):

        dt_clean = self.delayed_dt
        delayed_clean_mask = self.delayed_clean_mask
        clean_charges = self.clean_charges

        # 8. Apply delay time cut (dt < 3 µs) to delayed events
        
        # 9. Select only delayed events that passed the time cut
        # delayed_amp_time_mask is [all events] — we now filter that with time_mask
        final_event_mask = ak.Array(delayed_clean_mask)  # shape: [events]
        # Get indices of delayed events
        delayed_indices = ak.where(delayed_clean_mask)[0]  # np.array of indices
        
        # Now build final event-level mask
        final_event_mask = np.zeros(len(delayed_clean_mask), dtype=bool)
        final_event_mask[delayed_indices] = True  # shape: [event]
        
        # 10. Final cleaned arrays (hits within selected delayed events)
        digit_charge_final = clean_charges[final_event_mask]


        # Save important stuff
        self.final_event_mask = final_event_mask
        self.final_charges = digit_charge_final

    # Optional Plotting
    def plot_final_nhit_charges(self):
 
        digit_charge_final = self.final_charges
 
 
        print(len(digit_charge_final))
        
        # 11. Plot number of valid hits per selected delayed event
        plt.figure(figsize=(6,6))
        plt.hist(ak.sum(digit_charge_final,axis=1),bins=100,range=(0,2000),histtype='step')
        plt.xlabel("Total Digit Charge per delayed event")
        plt.ylabel("Count")
        plt.title("Final Delayed Events After All Cuts")
        plt.yscale('log')
        plt.show()
        
        # 11. Plot number of valid hits per selected delayed event
        plt.figure(figsize=(6,6))
        plt.hist(ak.count(digit_charge_final,axis=1),bins=100,range=(0,200),histtype='step')
        plt.xlabel("Number of hits per delayed event")
        plt.ylabel("Count")
        plt.title("Final Delayed Events After All Cuts")
        plt.yscale('log')
        plt.show()
 
    def plot_timing_cut(self):

        # Make time cut
        times = self.output_data['digitTime']
        event_times = self.output_data['time_quadfitter']
        event_times = ak.broadcast_arrays(times, event_times)[1]
        times = times - event_times
        cut = [-10, 15]
        hit_time_cut = (times > cut[0]) & (times < cut[1])
        # Apply time cut
        cut_times = ak.mask(times, hit_time_cut)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.hist(ak.flatten(times), bins=400, range=(-200, 200), histtype='step',label='Raw', color='tab:blue')
        plt.hist(ak.flatten(cut_times), bins=400, range=(-200, 200),label='Filtered', color='tab:blue')
        plt.axvspan(cut[0], cut[1], color='grey', alpha=0.2)
        plt.xlabel('Hit Time [ns]')
        plt.ylabel('Total Counts')
        plt.title('Digit Times Across All Channels and Events')
        plt.legend()
        plt.tight_layout()
        #plt.semilogy()
        #plt.ylim(1e5.1e7)
        plt.show()
        
    def create_michel_file(self, ofname):
        start = time.time()
        final_event_mask = self.final_event_mask
        f=up.open(self.file)
        outkey = self.get_valid_out_key(f)
        output_tree = f[outkey]
        meta_tree = f['meta']
        meta_arrays = meta_tree.arrays(filter_name=['pmtX', 'pmtY', 'pmtZ', 'pmtU', 'pmtV', 'pmtW'], library='np')
   
        # Get trigger times
        trig_times = output_tree['triggerTime'].array(entry_start=0, entry_stop=self.nevents, library='ak')

        selected_output_arrays = {}
        for key, branch in output_tree.items():
            try:
                array = branch.array(entry_start=0, entry_stop=self.nevents, library='ak')
            except ValueError:
                print(key, "was bad")
                continue
            if key == 'digitTime':
                array = array + trig_times
            #FIXME better to do hit cleaning to all hit variables
            if key in ['digitPMTID', 'digitTime', 'digitCharge']:
                array = ak.mask(array, self.hit_charge_cut & self.hit_time_cut)
                array = ak.drop_none(array)
            selected_output_arrays[key] = array[final_event_mask]
        
        with up.recreate(ofname) as out_file:
            out_file['output'] = selected_output_arrays
            out_file['meta'] = meta_arrays
            
        stop = time.time()
        print(stop-start)


def get_nhits_charges_times(fname):
    f = up.open(fname)
    out = f['output']
    meta = f['meta']
    
    digitPMTID = out['digitPMTID'].array(library='np')
    nhit = np.array([len(ev) for ev in digitPMTID])
    charge = out['digitCharge'].array(library='np')
    total_charge = np.array([np.sum(charges) for charges in charge])
    times = out['digitTime'].array(library='ak')
    all_times = np.concatenate(times)
    quad_times = out['time_quadfitter'].array(library='ak')
    quad_times = ak.broadcast_arrays(times, quad_times)[1]
    quad_times = np.concatenate(quad_times)
    corrected_times = all_times - quad_times
    #all_corrected_times = np.concatenate(corrected_times)
    trig_times = out['triggerTime'].array(library='ak')
    trig_times = ak.broadcast_arrays(times, trig_times)[1]
    trig_times = np.concatenate(trig_times)
    local_trig_times = out['digitLocalTriggerTime'].array(library='ak')
    all_local_trig_times = np.concatenate(local_trig_times)
    #corrected_trig_times = corrected_times - all_local_trig_times
    
    return nhit, total_charge, all_times, corrected_times, quad_times, trig_times, all_local_trig_times
