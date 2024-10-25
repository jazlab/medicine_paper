"""Run medicine."""

import numpy as np
import torch
from medicine import plotting
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix


def _kernelD(x, y, sig=1):
    """Forked from Kilosort4 repo."""
    ds = x[:, np.newaxis] - y
    Kn = np.exp(-(ds**2) / (2 * sig**2))
    return Kn


def _bin_spikes(ops, st):
    """Forked from Kilosort4 repo.

    For each batch, the spikes in that batch are binned to a 2D matrix by
    amplitude and depth.
    """
    # The bin edges are based on min and max of channel y positions
    ymin = ops["yc"].min()
    ymax = ops["yc"].max()
    dd = ops["binning_depth"]  # binning width in depth

    # Start 1um below the lowest channel
    dmin = ymin - 1

    # How many bins to use
    dmax = 1 + np.ceil((ymax - dmin) / dd).astype("int32")

    Nbatches = ops["Nbatches"]

    batch_id = st[:, 4].copy()

    # Always use 20 bins for amplitude binning
    F = np.zeros((Nbatches, dmax, 20))
    for t in range(ops["Nbatches"]):
        # Consider only spikes from this batch
        ix = (batch_id == t).nonzero()[0]
        sst = st[ix]

        # Their depth relative to the minimum
        dep = sst[:, 1] - dmin

        # The amplitude binnning is logarithmic, goes from the Th_universal
        # minimum value to 100.
        amp = np.log10(np.minimum(99, sst[:, 2])) - np.log10(
            ops["Th_universal"]
        )

        # amplitudes get normalized from 0 to 1
        amp = amp / (np.log10(100) - np.log10(ops["Th_universal"]))

        # Rows are divided by the vertical binning depth
        rows = (dep / dd).astype("int32")

        # Columns are from 0 to 20
        cols = (1e-5 + amp * 20).astype("int32")

        # For efficient binning, use sparse matrix computation in scipy
        cou = np.ones(len(ix))
        M = coo_matrix((cou, (rows, cols)), (dmax, 20))

        # The 2D histogram counts are transformed to logarithm
        F[t] = np.log2(1 + M.todense())

    # Center of each vertical sampling bin
    ysamp = dmin + dd * np.arange(dmax) - dd / 2

    return F, ysamp


def _align_block2(F, ysamp, ops):
    """Forked from Kilosort4 repo."""
    Nbatches = ops["Nbatches"]

    n = 15  # maximum vertical shift allowed, in units of bins
    dc = np.zeros((2 * n + 1, Nbatches))
    dt = np.arange(-n, n + 1, 1)

    # Batch fingerprints are mean subtracted along depth
    Fg = torch.from_numpy(F).float()
    Fg = Fg - Fg.mean(1).unsqueeze(1)

    # The template fingerprint is initialized with batch 300 if that exists
    F0 = Fg[np.minimum(300, Nbatches // 2)]

    niter = 10
    dall = np.zeros((niter, Nbatches))

    # At each iteration, align each batch to the template fingerprint
    # Fg is incrementally modified, and cumulative shifts are accumulated over
    # iterations
    for iter in range(niter):
        # For each vertical shift in the range -n to n, compute the dot product
        for t in range(len(dt)):
            Fs = torch.roll(Fg, dt[t], 1)
            dc[t] = (Fs * F0).mean(-1).mean(-1).cpu().numpy()

        # For all but the last iteration, align the batches
        if iter < niter - 1:
            # The maximum dot product is the best match for each batch
            imax = np.argmax(dc, 0)

            for t in range(len(dt)):
                # For batches which have the maximum at dt[t]
                ib = imax == t

                # Roll the fingerprints for those batches by dt[t]
                Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
                dall[iter, ib] = dt[t]

        # Take the mean of the aligned batches. This will be the new fingerprint
        # template.
        F0 = Fg.mean(0)

    # Divide the vertical bins into nblocks non-overlapping segments, and then
    # consider also the segments which half-overlap these segments
    nblocks = ops["nblocks"]
    nybins = F.shape[1]
    yl = nybins // nblocks
    ifirst = np.round(np.linspace(0, nybins - yl, 2 * nblocks - 1)).astype(
        "int32"
    )
    ilast = ifirst + yl

    # The new nblocks is 2*nblocks - 1 due to the overlapping blocks
    nblocks = len(ifirst)
    yblk = np.zeros(
        nblocks,
    )

    # Consider much smaller ranges for the fine drift correction
    n = 5
    dt = np.arange(-n, n + 1, 1)
    dcs = np.zeros((2 * n + 1, Nbatches, nblocks))

    # For each block in each batch, recompute the dot products with the template
    for j in range(nblocks):
        isub = np.arange(ifirst[j], ilast[j], 1)
        yblk[j] = ysamp[isub].mean()

        Fsub = Fg[:, isub]

        for t in range(len(dt)):
            Fs = torch.roll(Fsub, dt[t], 1)
            dcs[t, :, j] = (Fs * F0[isub]).mean(-1).mean(-1).cpu().numpy()

    # Upsamples the dot-product matrices by 10 to get finer estimates of vertica
    # ldrift
    dtup = np.linspace(-n, n, 2 * n * 10 + 1)

    # Get 1D upsampling matrix
    Kn = _kernelD(dt, dtup, 1)

    # Smooth the dot-product matrices across correlation, batches, and vertical
    # offsets
    dcs = gaussian_filter(dcs, ops["drift_smoothing"])

    # For each block, upsample the dot-product matrix and find new max
    imin = np.zeros((Nbatches, nblocks))
    for j in range(nblocks):
        dcup = Kn.T @ dcs[:, :, j]
        imax = np.argmax(dcup, 0)

        # The new max gets added to the last iteration of dall
        dall[niter - 1] = dtup[imax]

        # The cumulative shifts in dall represent the total vertical shift for
        # each batch
        imin[:, j] = dall.sum(0)

    # Fg gets reinitialized with the un-corrected F without subtracting the mean
    # across depth.
    Fg = torch.from_numpy(F).float()
    imax = dall[: niter - 1].sum(0)

    # Fg gets aligned again to compute the non-mean subtracted fingerprint
    for t in range(len(dt)):
        ib = imax == dt[t]
        Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
    F0m = Fg.mean(0)

    return imin, yblk, F0, F0m


def run_kilosort(
    recording_dir,
    write_dir,
    sampling_frequency=32000,
    batch_steps=60000,
    random_seed=0,
):
    print(
        f"\n\nRunning kilosort datashift on recording {recording_dir.name}\n"
    )

    # Set random seed
    np.random.seed(random_seed)

    # Load peak depths data
    peak_locations = np.load(recording_dir / "peak_locations.npy")
    peak_locations = np.array([x.tolist() for x in peak_locations])
    peak_locations = dict(
        x=peak_locations[:, 0],
        y=peak_locations[:, 1],
        z=peak_locations[:, 2],
        alpha=peak_locations[:, 3],
    )
    peak_depths = peak_locations["y"]

    # Load peaks data
    peaks = np.load(recording_dir / "peaks.npy")
    peaks = np.array([x.tolist() for x in peaks])
    peaks = dict(
        sample_index=peaks[:, 0],
        channel_index=peaks[:, 1],
        amplitude=peaks[:, 2],
        segment_index=peaks[:, 3],
    )
    peak_amplitudes = peaks["amplitude"]
    peak_times = peaks["sample_index"] / sampling_frequency

    # Create ops dictionary
    batch_duration = float(batch_steps) / sampling_frequency
    ops = {
        "nblocks": 1,
        "drift_smoothing": [0.5, 0.5, 0.5],
        "binning_depth": 5,
        "yc": peak_depths,
        "Nbatches": 1 + int(np.max(peak_times) / batch_duration),
        "Th_universal": 9,
    }

    # Create st array [_, depth, amplitude, _, batch]
    batch_index = np.floor(peak_times / batch_duration).astype(int)
    peak_amplitudes *= -1
    peak_amplitudes -= np.min(peak_amplitudes)
    peak_amplitudes /= np.max(peak_amplitudes)
    peak_amplitudes = 10 + 90 * peak_amplitudes
    st = np.stack(
        [
            np.zeros_like(peak_depths),
            peak_depths,
            peak_amplitudes,
            np.zeros_like(peak_depths),
            batch_index,
        ],
        axis=1,
    )

    # spikes are binned by amplitude and y-position to construct a "fingerprint" for each batch
    F, ysamp = _bin_spikes(ops, st)

    # the fingerprints are iteratively aligned to each other vertically
    imin, _, _, _ = _align_block2(F, ysamp, ops)

    # imin contains the shifts for each batch, in units of discrete bins
    # multiply back with binning_depth for microns
    dshift = -1 * imin * ops["binning_depth"]

    # Plot raster
    _ = plotting.plot_raster_and_amplitudes(
        peak_times,
        peak_depths,
        np.log(peak_amplitudes),
        figure_dir=write_dir,
    )

    # Plot motion correction
    times = batch_duration * np.arange(ops["Nbatches"])
    fig = plotting.plot_motion_correction(
        peak_times=peak_times,
        peak_depths=peak_depths,
        peak_amplitudes=peak_amplitudes,
        time_bins=times,
        depth_bins=np.array([0]),
        motion=dshift,
    )
    _ = fig.suptitle("Motion correction", fontsize=24)
    save_path = write_dir / "corrected_motion_raster.png"
    print(f"Saving figure to {save_path}")
    fig.savefig(save_path)

    # Save outputs
    print(f"Saving outputs to {write_dir}")
    np.save(write_dir / "time_bins.npy", times)
    np.save(write_dir / "depth_bins.npy", np.array([0]))
    np.save(write_dir / "motion.npy", dshift[:, None])

    print("\n\nDONE\n\n")

    return
