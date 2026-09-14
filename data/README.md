# External data placement

The experimental data are not redistributed in this repository.

All three inputs can be downloaded, safely extracted, and checksum-verified
from their public source locations with:

```bash
python scripts/fetch_datasets.py --dataset all
```

Individual choices are `picmus-in-vivo`, `picmus-resolution`, and `mbtrace`.
Interrupted downloads use a resumable `.part` file when the server supports
byte ranges. The download script records a local `data/dataset_manifest.json`;
all downloaded material remains ignored by Git.

## PICMUS carotid acquisitions

The original data are available from the
[PICMUS download page](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download).

Set `PICMUS_DATA_DIR` to the directory containing this layout:

```text
in_vivo/
  carotid_cross/carotid_cross_expe_dataset_rf.hdf5
  carotid_long/carotid_long_expe_dataset_rf.hdf5
```

If the environment variable is omitted, `src/bmode_picmus.py` looks under
`data/PICMUS/in_vivo/` relative to the repository root. The script also accepts
the file-specific variables `PICMUS_CC_FILE` and `PICMUS_CL_FILE`.

## Open-NSI MBTrace acquisition

The acquisition is distributed through the
[Open-NSI repository](https://github.com/ZhengchangKou/Open-NSI), whose README
provides the example-data download link.

Set `OPEN_NSI_MBTRACE_FILE` to the downloaded file, or place it at:

```text
data/Open-NSI/Basic/data/MBTrace.mat
```

The script may create `MBTrace.npy` beside the MAT file as a local loading
cache. Both external datasets and the cache are ignored by Git.

## PICMUS experimental resolution acquisition

The point-target RF acquisition, phantom definition, and scan definition are
contained in the official PICMUS `archive_to_download.zip`. The fetch script
extracts only the three required files to:

```text
data/PICMUS/resolution_distorsion/
  resolution_distorsion_expe_dataset_rf.hdf5
  resolution_distorsion_expe_phantom.hdf5
  resolution_distorsion_expe_scan.hdf5
```

This acquisition is used by `src/picmus_experimental_psf.py` for five near-axis
depths and the two off-axis points at approximately 37.5 mm.
