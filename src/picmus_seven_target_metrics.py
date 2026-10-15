"""Run the unchanged NumPy/CuPy beamforming algebra on seven local CPU windows.

The small backend adapter replaces device transfers/synchronization with CPU
operations and uses SciPy's single-precision FFT. Physical settings, delays,
apertures, CF, Capon MV, and filtered DMAS are imported from the revision code.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
import scipy.fft
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import conventional_baseline_comparison as pipeline
from adaptive_beamforming import MvConfiguration, FdmasFilterConfiguration
from method_names import METHODS, DMAS
from picmus_experimental_psf import build_target_grid, metrics_for_method, demodulate_rf
from picmus_io import load_picmus_dataset,load_picmus_phantom

class CpuBackend:
    fft=scipy.fft
    cuda=SimpleNamespace(Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda:None)))
    def __getattr__(self,name): return getattr(np,name)
    def asnumpy(self,values): return np.asarray(values)
    def get_default_memory_pool(self): return SimpleNamespace(free_all_blocks=lambda:None)

BACKEND=CpuBackend()
DATASET=None
PHANTOM=None
IQ=None
FC=5.208333e6
ANGLE_INDICES=np.arange(75,dtype=np.int64)
OUT=ROOT/'results/generated/revision/seven_target_metrics'
DATA_DIR=ROOT/'data/PICMUS/resolution_distorsion'
REFERENCE=ROOT/'results/reported/paper_inputs/conventional_baselines/conventional_baseline_experimental_psf_profiles.npz'
INPUT_DIGEST=''
MV_CONFIG=MvConfiguration(temporal_half_window_samples=0,chunk_pixels=32)
DMAS_CONFIG=FdmasFilterConfiguration()

def timestamp(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def log(message): print(timestamp(),message,flush=True)

def load_inputs():
    global DATASET,PHANTOM,IQ,INPUT_DIGEST
    data=DATA_DIR
    DATASET=load_picmus_dataset(data/'resolution_distorsion_expe_dataset_rf.hdf5')
    PHANTOM=load_picmus_phantom(data/'resolution_distorsion_expe_phantom.hdf5')
    digest=hashlib.sha256()
    for path in sorted(data.glob('resolution_distorsion_expe_*.hdf5')):
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    INPUT_DIGEST=digest.hexdigest()
    log(f'Loaded verified RF: {DATASET.data.shape}; preparing complex64 IQ on CPU.')
    IQ=demodulate_rf(DATASET.data,DATASET.sampling_frequency_hz,FC,BACKEND)
    assert IQ.dtype==np.complex64 and np.isfinite(IQ).all()
    # Every task uses the same fixed 75 transmissions; cache the identical IQ.
    pipeline.demodulate_rf=lambda *args:IQ
    log('Shared complex64 IQ prepared.')

def iq_reconstruction(points,label):
    return pipeline.reconstruct_iq_methods(
        DATASET,ANGLE_INDICES,points,grid_shape=(len(points),1),
        carrier_frequency_hz=FC,f_number=1.0,nsi_c=.05,
        mv_configuration=MV_CONFIG,cp=BACKEND,label=label,
        focus_chunk_pixels=1024)

def dmas_reconstruction(x_mm,z_mm,x0,z0,label):
    # Preserve the existing representative-PSF filter extent and padding.
    depth,filter_meta=pipeline.fdmas_padded_axis(
        (z0-1.5)*1e-3,(z0+1.5)*1e-3,.02e-3,
        sound_speed_m_s=DATASET.sound_speed_m_s,
        carrier_frequency_hz=FC,filter_configuration=DMAS_CONFIG)
    images,metadata=pipeline.reconstruct_fdmas(
        DATASET,ANGLE_INDICES,(x_mm*1e-3).astype(np.float32),depth,
        carrier_frequency_hz=FC,f_number=1.0,
        filter_configuration=DMAS_CONFIG,cp=BACKEND,label=label,x_chunk_lines=16)
    return images,depth,metadata

def validate_reference():
    old=REFERENCE
    with np.load(old) as archive:
        xa=archive['lateral_x_mm'];za=archive['axial_z_mm']
        x0,_,z0=(PHANTOM.positions_m[3].astype(float)*1000)
        points=np.concatenate([
            np.column_stack([xa,np.zeros(xa.size),np.full(xa.size,z0)]),
            np.column_stack([np.full(za.size,x0),np.zeros(za.size),za])])
        log('Validating the five IQ methods against saved central-target GPU profiles.')
        values=iq_reconstruction((points*1e-3).astype(np.float32),'CPU validation')
        audit=[]
        for method,y in values.items():
            slug=method.lower().replace('-','_')
            for name,y0 in [('lateral',y[:xa.size]),('axial',y[xa.size:])]:
                reference=archive[f'{slug}_{name}'].astype(float)
                relative=float(np.max(abs(y0-reference))/np.max(reference))
                audit.append({'method':method,'profile':name,'maximum_relative_amplitude_error':relative})
                log(f'Validation {method} {name}: max relative amplitude error {relative:.6g}')
                if relative>0.015: raise RuntimeError(f'CPU/GPU profile agreement failed: {method} {name}')
        log('Validating filtered DMAS against saved central-target GPU profiles.')
        dmas,depth,metadata=dmas_reconstruction(xa,za,x0,z0,'CPU DMAS validation')
        yy=pipeline.resample_regular_image(dmas,(xa*1e-3).astype(np.float32),depth,
            xa*1e-3,np.asarray([z0*1e-3]))[:,0]
        reference=archive['dmas_lateral'].astype(float)
        relative=float(np.max(abs(yy-reference))/np.max(reference))
        audit.append({'method':DMAS,'profile':'lateral','maximum_relative_amplitude_error':relative})
        log(f'Validation DMAS lateral: max relative amplitude error {relative:.6g}')
        if relative>0.015: raise RuntimeError('CPU/GPU DMAS profile agreement failed.')
        (OUT/'cpu_gpu_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    log('Central-target CPU/GPU validation passed.')

def compute_target(tid):
    with threadpool_limits(limits=1):
        started=time.monotonic()
        out=OUT/f'target_{tid}_metrics.json'
        if out.is_file():
            cached=json.loads(out.read_text())
            if cached.get('computation_signature')==signature():
                log(f'Reusing validated target {tid}.');return cached
        position=PHANTOM.positions_m[tid-1].astype(float)*1000
        x0,_,z0=position
        grid=build_target_grid(x0,z0,map_half_width_mm=.8,map_spacing_mm=.02,
            profile_half_width_mm=1.0,lateral_spacing_mm=.002,axial_spacing_mm=.005)
        log(f'Target {tid} IQ reconstruction started at ({x0:.2f},{z0:.1f}) mm; {len(grid.points_m)} points.')
        images=iq_reconstruction(grid.points_m,f'target {tid}')
        log(f'Target {tid} IQ reconstruction complete; starting DMAS.')
        # The broad DMAS response uses directly focused 0.005-mm lateral lines.
        # Its depth sampling is 0.02 mm, with the same fixed bandpass/padding.
        dmas_x=np.linspace(x0-1,x0+1,401)
        dmas,depth,filter_metadata=dmas_reconstruction(dmas_x,grid.map_z_mm,x0,z0,f'target {tid} DMAS')
        map_values=pipeline.resample_regular_image(dmas,(dmas_x*1e-3).astype(np.float32),depth,
            grid.map_x_mm*1e-3,grid.map_z_mm*1e-3)
        lat=pipeline.resample_regular_image(dmas,(dmas_x*1e-3).astype(np.float32),depth,
            grid.lateral_x_mm*1e-3,np.asarray([z0*1e-3]))[:,0]
        axial=pipeline.resample_regular_image(dmas,(dmas_x*1e-3).astype(np.float32),depth,
            np.asarray([x0*1e-3]),grid.axial_z_mm*1e-3)[0]
        images[DMAS]=np.concatenate([map_values.ravel(),lat,axial]).astype(np.float32)
        records=[];payload={'lateral_x_mm':grid.lateral_x_mm,'axial_z_mm':grid.axial_z_mm,
                            'map_x_mm':grid.map_x_mm,'map_z_mm':grid.map_z_mm}
        for method in METHODS:
            metrics,arrays=metrics_for_method(images[method],grid,x0,z0)
            for key in ['lateral_width_mm','axial_width_mm','localization_error_mm','peak_to_median_background_db']:
                if metrics[key] is None or not np.isfinite(metrics[key]): raise RuntimeError(f'Invalid {method} {key}')
            records.append({'target_id':tid,'target_x_mm':x0,'target_z_mm':z0,'method':method,**metrics})
            slug=method.lower().replace('-','_')
            for name,array in arrays.items():payload[f'{slug}_{name}']=array
        np.savez_compressed(OUT/f'target_{tid}_arrays.npz',**payload)
        result={'computation_signature':signature(),'target_id':tid,'position_mm':position.tolist(),
                'elapsed_seconds':time.monotonic()-started,'metrics':records,'dmas_filter':filter_metadata}
        out.write_text(json.dumps(result,indent=2)+'\n')
        log(f'Target {tid} checkpoint saved in {result["elapsed_seconds"]:.1f}s.')
        return result

def signature():
    paths=sorted((ROOT/'src').glob('*.py'))
    source_digests=''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)
    environment=np.__version__+scipy.__version__
    return hashlib.sha256((source_digests+INPUT_DIGEST+environment).encode()).hexdigest()

def main():
    global OUT,DATA_DIR,REFERENCE
    parser=argparse.ArgumentParser(description='Reconstruct all seven PICMUS targets with the six paper methods on the validated CPU backend.')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--reference-profiles',type=Path,default=REFERENCE)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    OUT=args.output_dir.expanduser().resolve();OUT.mkdir(parents=True,exist_ok=True)
    DATA_DIR=args.data_dir.expanduser().resolve();REFERENCE=args.reference_profiles.expanduser().resolve()
    with threadpool_limits(limits=1):
        load_inputs()
        if not (OUT/'cpu_gpu_validation.json').is_file():validate_reference()
        if args.validate_only:return
        tasks={}
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for tid in range(1,8):tasks[pool.submit(compute_target,tid)]=tid
            results=[]
            for task in as_completed(tasks):results.append(task.result())
    records=[row for result in sorted(results,key=lambda r:r['target_id']) for row in result['metrics']]
    aggregate=[]
    for method in METHODS:
        rows=[r for r in records if r['method']==method]
        assert len(rows)==7
        stats={'method':method,'n':7}
        for metric in ['lateral_width_mm','axial_width_mm','localization_error_mm','peak_to_median_background_db']:
            values=np.asarray([r[metric] for r in rows],dtype=float)
            stats[metric]={'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),
                           'minimum':float(values.min()),'maximum':float(values.max())}
        aggregate.append(stats)
    summary={'computed_utc':timestamp(),'signature':signature(),'backend':'NumPy/SciPy CPU, complex64','input_sha256':INPUT_DIGEST,
             'nsi_c':.05,'angles':75,'f_number':1.0,'carrier_frequency_hz':FC,
             'map_extent_mm':[1.6,1.6],'map_spacing_mm':.02,'lateral_profile_spacing_mm':.002,
             'axial_profile_spacing_mm':.005,'dmas_native_lateral_spacing_mm':.005,
             'dmas_native_axial_spacing_mm':.02,'background_exclusion_radius_mm':.35,
             'peak_background_definition':'20 log10(local map maximum / median background outside exclusion disk)',
             'standard_deviation':'sample SD across seven target-wise scalar metrics, ddof=1',
             'mv_configuration':MV_CONFIG.to_dict(),'dmas_configuration':DMAS_CONFIG.to_dict(),
             'records':records,'aggregate':aggregate}
    import csv
    columns=['target_id','target_x_mm','target_z_mm','method','lateral_width_mm','axial_width_mm','localization_error_mm','peak_to_median_background_db']
    with (OUT/'seven_target_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    (OUT/'seven_target_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    log('All 42 method-target cases complete; mean and sample SD saved.')
    for row in aggregate:log(str(row))

if __name__=='__main__':main()
