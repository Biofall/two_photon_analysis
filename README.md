# two_photon_analysis

This project is intended to serve as a complete package for analyzing visual areas of the drosophila brain using two-photon imaging.
A majority of the backbone of the analysis comes from visanalysis, a code package written primarily by Max Turner: https://github.com/ClandininLab/visanalysis. Go into the directory level that has "setup.py" file
in it and run the following command to add these tools to your virtual environment.

Written by Avery Krieger - contact @ krave@stanford.edu

```python
(your venv) >> pip install -e .
```

For Waaaaayyyy more information go here: [ouroboros](https://github.com/tgfisher/ouroboros).

Currently, the medulla_analysis.py file contains functions called by the scripts>plotting_responses.py file. 

ToDo:
-Finish writing backbone code to turn this into an installable package
-create separate utility functions for things like motion correction 
