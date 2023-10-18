
import scipy.fftpack as sfft
import numpy as np





def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


def fourier_filter_2d(
    image: np.ndarray,
    ret: str = "filtered",
) -> np.ndarray:
    """Function to Fourier filter an image for removal of regular pattern artefacts,
       e.g. grid lines.

    Args:
        image: the input image
        peaks: list of dicts containing the following information about a "peak" in the
               Fourier image:
               'pos_x', 'pos_y', sigma_x', sigma_y', 'amplitude'. Define one entry for
               each feature you want to suppress in the Fourier image, where amplitude
               1 corresponds to full suppression.
        ret: flag to indicate which data to return. Possible values are:
             'filtered', 'fft', 'mask', 'filtered_fft'

    Returns:
        The chosen image data. Default is the filtered real image.
    """
    peaks_fft = {}
    peaks_fft['pos_x'] = [442, 407, 447, 520, 553, 510, 395, 364, 336, \
        379, 487, 560, 593, 625, 585, 469, 357, 322, 291, 258, 602, 638, 665, 696]
    peaks_fft['pos_y'] = [277, 354, 404, 384, 303, 255, 236, 310, 383, \
        427, 450, 424, 351, 275, 230, 208, 187, 261, 334, 406, 466, 394, 321, 255]
    peaks_fft['amplitude'] = np.ones(np.shape(peaks_fft['pos_x']))
    peaks_fft['sigma_x'] = 12*np.ones(np.shape(peaks_fft['pos_x']))
    peaks_fft['sigma_y'] = 28*np.ones(np.shape(peaks_fft['pos_x']))

    peak = peaks_fft # for PSI case

    # Do Fourier Transform of the (real-valued) image
    image_fft = sfft.fftshift(sfft.fft2(image))
    mask = np.ones(image_fft.shape)
    xgrid, ygrid = np.meshgrid(
        range(image_fft.shape[0]),
        range(image_fft.shape[1]),
        indexing="ij",
        sparse=True,
    )

    for i in range(len(peak['pos_x'])):
            try:
                mask -= peak["amplitude"][i] * gauss2d(
                    xgrid,
                    ygrid,
                    peak["pos_x"][i],
                    peak["pos_y"][i],
                    peak["sigma_x"][i],
                    peak["sigma_y"][i],
                )
            except KeyError as exc:
                raise KeyError(
                    f"The peaks input is supposed to be a list of dicts with the\
    following structure: pos_x, pos_y, sigma_x, sigma_y, amplitude. The error was {exc}.",
                ) from exc

    # apply mask to the FFT, and transform back
    # apply mask to the FFT, and transform back
    filtered = sfft.ifft2(sfft.ifftshift(image_fft * mask))
    # strip negative values

    filtered = np.abs(filtered.clip(min=0))
    # strip negative values

    if ret == "filtered":
        return filtered
    if ret == "fft":
        return image_fft
    if ret == "mask":
        return mask
    if ret == "filtered_fft":
        return image_fft * mask
    return filtered  # default return