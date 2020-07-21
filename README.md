# protodune-neutron-hit-ana

Simple tools for analyzing hit dumps taken continuously from ProtoDUNE-SP runs with neutron source. Tested with Python 3.7.4. Required python packages:
* `numpy`
* `matplotlib`
* `scikit-learn`
* `arrow`

The main script is `cluster.py` which runs DBSCAN clustering to identify hits in tracks. It does some counting analysis on the results. Input files are numpy arrays of hits in format:

```
channel timestamp adcsum time_over_threshold
```

