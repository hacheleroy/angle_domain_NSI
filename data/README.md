# External data placement

The experimental data are not redistributed in this repository.

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

