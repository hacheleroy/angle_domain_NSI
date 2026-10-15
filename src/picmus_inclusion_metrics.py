"""Measure the existing PICMUS bright inclusion with fixed, shared ROIs.

The primary estimator is the existing revision pipeline's normalized_db and
compute_gcnr implementation, with its established 120-dB numerical floor and
100 bins. No image is reconstructed or modified for quantitative analysis.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
sys.path.insert(0, str(SOURCE))
from conventional_baseline_comparison import compute_gcnr, normalized_db
from method_names import METHODS

CACHE = ROOT / 'results/generated/revision/picmus_full_phantom/picmus_full_phantom_cache.npz'
INCLUSION_CENTER_MM = (-10.5, 27.5)
BACKGROUND_CENTER_MM = (10.5, 27.5)
RADIUS_MM = 2.0
BINS = 100


def masks(x_mm, z_mm, radius=RADIUS_MM, dx=0.0, dz=0.0):
    x, z = np.meshgrid(x_mm, z_mm, indexing='ij')
    left = INCLUSION_CENTER_MM[0] + dx
    depth = INCLUSION_CENTER_MM[1] + dz
    inclusion = (x-left)**2 + (z-depth)**2 <= radius**2
    background = (x+left)**2 + (z-depth)**2 <= radius**2
    assert inclusion.any() and background.any() and not np.any(inclusion & background)
    assert left-radius > x_mm.min() and -left+radius < x_mm.max()
    return inclusion, background


def probability_histograms(signal, background, bins):
    edges = np.linspace(min(signal.min(), background.min()),
                        max(signal.max(), background.max()), bins+1)
    first = np.histogram(signal, bins=edges)[0].astype(float)
    second = np.histogram(background, bins=edges)[0].astype(float)
    return edges, first/first.sum(), second/second.sum()


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Measure the fixed PICMUS inclusion/background ROIs used in main Table 2 and supplementary Figure S10.')
    parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/generated/revision/phantom_inclusion')
    args=parser.parse_args()
    cache=args.cache.expanduser().resolve()
    output=args.output_dir.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    with np.load(cache,allow_pickle=False) as archive:
        x = archive['x_axis_m'].astype(float)*1000
        z = archive['z_axis_m'].astype(float)*1000
        images = {method:archive['image_'+method.lower().replace('-','_')].astype(float)
                  for method in METHODS}
    inside, outside = masks(x,z)
    primary, checks, arrays = [], [], {'x_mm':x, 'z_mm':z,
                                     'inclusion_mask':inside,'background_mask':outside}
    for method, image in images.items():
        assert np.isfinite(image).all() and (image >= 0).all() and image.max() > 0
        db = normalized_db(image, floor_db=-120)
        signal, background = db[inside], db[outside]
        value = compute_gcnr(signal, background, BINS)
        edges, first, second = probability_histograms(signal,background,BINS)
        assert np.isclose(value,1-np.minimum(first,second).sum(),rtol=0,atol=1e-14)
        assert 0 <= value <= 1
        primary.append({'method':method,'gcnr':value,'normalization_reference_peak':float(image.max()),
                        'inclusion_pixel_count':int(inside.sum()),
                        'background_pixel_count':int(outside.sum())})
        slug = method.lower().replace('-','_')
        arrays.update({slug+'_inclusion_envelope':image[inside],
                       slug+'_background_envelope':image[outside],
                       slug+'_edges_db':edges,slug+'_inclusion_probability':first,
                       slug+'_background_probability':second})
        for radius in [1.5,1.75,2.0,2.25]:
            a,b = masks(x,z,radius)
            for bins in [50,100,200]:
                checks.append({'method':method,'check':'radius_and_bins','radius_mm':radius,
                               'bins':bins,'dx_mm':0,'dz_mm':0,
                               'gcnr':compute_gcnr(db[a],db[b],bins)})
        for dx in [-.25,0,.25]:
            for dz in [-.25,0,.25]:
                a,b = masks(x,z,dx=dx,dz=dz)
                checks.append({'method':method,'check':'paired_center_shift',
                               'radius_mm':RADIUS_MM,'bins':BINS,'dx_mm':dx,'dz_mm':dz,
                               'gcnr':compute_gcnr(db[a],db[b],BINS)})
        # Verify the conclusion without the log-domain histogram convention.
        for mode in ['linear_envelope','pooled_quantile_bins']:
            s,b = image[inside],image[outside]
            pooled = np.r_[s,b]
            edges = (np.linspace(pooled.min(),pooled.max(),BINS+1) if mode=='linear_envelope'
                     else np.unique(np.quantile(pooled,np.linspace(0,1,BINS+1))))
            sh=np.histogram(s,bins=edges)[0];bh=np.histogram(b,bins=edges)[0]
            alternate=1-np.minimum(sh/sh.sum(),bh/bh.sum()).sum()
            checks.append({'method':method,'check':mode,'radius_mm':RADIUS_MM,'bins':BINS,
                           'dx_mm':0,'dz_mm':0,'gcnr':float(alternate)})
        print(method,f'{value:.9f}')
    for filename, rows in [('inclusion_gcnr.csv',primary),('inclusion_gcnr_sensitivity.csv',checks)]:
        with (output/filename).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'inclusion_gcnr_samples.npz',**arrays)
    summary={'source_cache':str(cache),'source_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
             'inclusion_center_xz_mm':INCLUSION_CENTER_MM,'background_center_xz_mm':BACKGROUND_CENTER_MM,
             'roi_radius_mm':RADIUS_MM,'primary_bins':BINS,'histogram_domain':'log envelope, shared equal-width edges per method',
             'numerical_floor_db':-120,'display_floor_db':-60,
             'estimator':'1 - sum_j min(p_inclusion[j], p_background[j])',
             'gcnr_function_sha256':hashlib.sha256((SOURCE/'conventional_baseline_comparison.py').read_bytes()).hexdigest(),
             'sampling':'Existing common overview grid, 0.05 mm lateral by 0.10 mm axial; native MV/DMAS grids as in Supplement S10.',
             'replication':'One inclusion/background ROI pair per method; these are not seven-target averages.',
             'primary':primary,'sensitivity':checks}
    (output/'inclusion_gcnr_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Display-only crops show the complete predefined masks on all six methods.
    fig,axes=plt.subplots(2,3,figsize=(11.2,4.2),layout='constrained')
    values={r['method']:r['gcnr'] for r in primary}
    for ax,(method,image) in zip(axes.flat,images.items()):
        shown=normalized_db(image,floor_db=-60)
        im=ax.imshow(shown.T,extent=[x[0],x[-1],z[-1],z[0]],cmap='gray',vmin=-60,vmax=0,aspect='equal')
        for center,color,label in [(INCLUSION_CENTER_MM,'#f59e0b','Inclusion'),
                                   (BACKGROUND_CENTER_MM,'#00bfc4','Background')]:
            ax.add_patch(Circle(center,RADIUS_MM,fill=False,color=color,lw=1.5))
        ax.set_xlim(x[0],x[-1]);ax.set_ylim(33,22)
        ax.set_xticks([-10,0,10]);ax.set_yticks([24,28,32])
        ax.set_xlabel('Lateral position [mm]');ax.set_ylabel('Depth [mm]')
        ax.set_title(f'{method}   gCNR = {values[method]:.3f}',fontsize=10)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Normalized amplitude [dB]',shrink=.8,pad=.015)
    fig.suptitle('Inclusion (orange) and background (cyan): identical 2 mm radius regions',fontsize=11)
    fig.savefig(output/'figureS13_picmus_inclusion_rois.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    # This is a separation result for the existing maps, not a population test.
    worst_nsi=max(r['gcnr'] for r in checks if r['method'] in ['Receive-NSI','Angle-NSI'])
    best_lower_bound=min(r['gcnr'] for r in checks if r['method'] not in ['Receive-NSI','Angle-NSI'])
    assert worst_nsi < best_lower_bound
    print('Across all checks: max NSI',worst_nsi,'min other methods',best_lower_bound)


if __name__=='__main__':main()
