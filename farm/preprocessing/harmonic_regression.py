"""Local sinusoidal regression for explicitly selected residual spectral lines."""
import numpy as np


def refine_frequencies(x, fs, frequencies, search_width):
    """Refine coarse detections on the full Hann-windowed spectrum."""
    values = np.asarray(x, dtype=float)
    power = np.abs(np.fft.rfft((values-values.mean())*np.hanning(len(values))))**2
    df = fs/len(values)
    result = []
    for frequency in frequencies:
        lo = max(1, int(np.floor((frequency-search_width)/df)))
        hi = min(len(power)-2, int(np.ceil((frequency+search_width)/df)))
        peak = lo+int(np.argmax(power[lo:hi+1]))
        logp = np.log(np.maximum(power[peak-1:peak+2], 1e-300))
        curvature = logp[0]-2*logp[1]+logp[2]
        offset = .5*(logp[0]-logp[2])/curvature if abs(curvature)>1e-20 else 0.
        result.append(float((peak+np.clip(offset,-.5,.5))*df))
    return result


def regress_lines(x, fs, selected, window_seconds=32.):
    """Subtract overlapping Hann-weighted sine/cosine fits on original samples.

    Unlike an IIR notch, this estimates slow amplitudes of selected sinusoids.
    Genuine physiology near a selected frequency can still be attenuated.
    Partial edge windows use measured samples, without zero filling.
    """
    signal = np.asarray(x)
    if (signal.ndim != 1 or not len(signal) or not np.issubdtype(signal.dtype, np.floating)
            or not np.isfinite(signal).all() or not np.isfinite(fs) or fs <= 0
            or not np.isfinite(window_seconds) or window_seconds < 16):
        raise ValueError('Invalid signal, rate or regression window')
    for candidate in selected:
        frequency, confidence = candidate['frequency_hz'], candidate.get('regression_weight', 1.)
        if not np.isfinite(frequency) or not 0 < frequency < fs/2 or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid frequency or confidence in regression plan')
    if not selected:
        return signal.copy(), 0
    length = max(16, int(round(window_seconds*fs)))
    hop = max(1, length//4)
    t = np.arange(length)/fs
    window = np.hanning(length)
    correction = np.zeros(len(signal), dtype=float)
    weights = np.zeros(len(signal), dtype=float)
    bases = [(np.cos(2*np.pi*c['frequency_hz']*t), np.sin(2*np.pi*c['frequency_hz']*t))
             for c in selected]
    for start in range(-length//2, len(signal), hop):
        left, right = max(0, start), min(len(signal), start+length)
        if right-left < 8:
            continue
        first, last = left-start, right-start
        w = window[first:last]
        values = signal[left:right].astype(float)
        fitted = np.zeros(len(values), dtype=float)
        for candidate, (cosine, sine) in zip(selected, bases):
            c, s = cosine[first:last], sine[first:last]
            cc, ss, cs = np.dot(w,c*c), np.dot(w,s*s), np.dot(w,c*s)
            determinant = cc*ss-cs*cs
            if determinant <= 1e-20:
                continue
            yc, ys = np.dot(w*values,c), np.dot(w*values,s)
            a, b = (yc*ss-ys*cs)/determinant, (ys*cc-yc*cs)/determinant
            fitted += candidate.get("regression_weight", 1.)*(a*c+b*s)
        correction[left:right] += w*fitted
        weights[left:right] += w
    correction /= np.maximum(weights, 1e-30)
    return (signal-correction).astype(signal.dtype), length


def coherence_at_frequencies(reference, witness, fs, length, frequencies):
    """Magnitude-squared cross-channel coherence at exact candidate frequencies.

    Direct Hann-weighted projections avoid interpolating coherence ratios from
    neighboring FFT bins. The same bounded, non-overlapping windows are used.
    """
    count = len(reference)//length
    blocks = np.unique(np.linspace(0,count-1,min(count,64)).astype(int))
    a = np.asarray([reference[k*length:(k+1)*length] for k in blocks], dtype=float)
    b = np.asarray([witness[k*length:(k+1)*length] for k in blocks], dtype=float)
    a -= a.mean(axis=1,keepdims=True)
    b -= b.mean(axis=1,keepdims=True)
    t = np.arange(length)/fs
    basis = np.hanning(length)[:,None]*np.exp(-2j*np.pi*t[:,None]*np.asarray(frequencies)[None,:])
    ca,cb = a@basis,b@basis
    numerator = np.abs(np.mean(ca*np.conj(cb),axis=0))**2
    denominator = np.mean(np.abs(ca)**2,axis=0)*np.mean(np.abs(cb)**2,axis=0)
    return np.clip(numerator/np.maximum(denominator,1e-30),0,1)
