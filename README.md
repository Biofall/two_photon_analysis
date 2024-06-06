# Two Photon Analysis Project

## Overview
This project is designed as a comprehensive toolkit for analyzing the visual areas of the Drosophila brain using two-photon imaging. It integrates essential tools from the `visanalysis` package (by Max Turner: https://github.com/ClandininLab/visanalysis), adding a suite of custom scripts and notebooks for motion correction, optogenetic response analysis, data visualization, and experimental data processing tailored to specific research needs.

## Key Features
- **Motion Correction**: Align imaging sequences to correct for motion artifacts.
- **Optogenetic and Flash Response Analysis**: Analyze and quantify neural responses to optogenetic and sensory stimuli.
- **Visualization**: Utilize advanced plotting tools for detailed examination and presentation of data.
- **Data Processing**: Custom pipelines for processing and analyzing experimental data.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Biofall/two_photon_analysis.git
   cd two_photon_analysis
   ```
2. Install the `visanalysis` package:
   ```bash
   git clone https://github.com/ClandininLab/visanalysis
   cd visanalysis
   pip install -e .
   cd ..
   ```
3. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Navigate to the directory containing the `setup.py` file and install the package locally:
```bash
pip install -e .
```

### Detailed Description of Components
#### Scripts
- **`batch_common_moco.py`**: Automates motion correction across multiple datasets using a common reference.
- **`moco_scratch.py`**: Experimental script for testing new motion correction algorithms.
- **`flash_w_opto_step.py`**: Processes data from experiments combining uniform flash stimuli with optogenetic steps to study neural response dynamics.
- **`plotting_responses.py`**: Generates plots to visualize neural activity responses.

#### Notebooks
- **`q_vis_response_plotter.ipynb`**: Visualizes quantitative data from visual response experiments.
- **`opto_step_series_plotter.ipynb`**: Plots detailed time-series data from optogenetic step response experiments.
- **`imaging_params_finder.ipynb`**: Helps identify optimal imaging parameters for experiments.
- **`detailed_response_analysis.ipynb`**: Conducts in-depth analysis of neural responses to various stimuli.
- **`fig_generator.ipynb`**: Generates high-quality figures for publications and presentations.
- **`perfusion_analysis.ipynb`**: Analyzes data from perfusion experiments to assess drug effects.
- **`trf_collecter.ipynb`**: Collects and processes temporal response fields from neural data.
- **`trace_plotting.ipynb`**: Plots time-series traces from neural recordings.
- **`asta_addition.ipynb`**: Explores additional analysis specific to AstA-neuron responses.
- **`moco_avery.ipynb`**: Avery's custom notebook for applying and refining motion correction techniques.
- **`plotting_party.ipynb`**: Interactive session notebook for collaborative data visualization.
- **`process_imports.ipynb`**: Manages imports and packages required for the analysis.
- **`noise_analysis_mht.ipynb`**: Analyzes noise characteristics in imaging data.
- **`filter_reconstruction.ipynb`**: Reconstructs imaging data using various filters to improve clarity and resolution.

## Contributing
Contributions to enhance or expand the toolkit are welcome. Please fork the repository and submit pull requests or reach out directly to Avery Krieger at krave@stanford.edu.

## License
Specify the license under which your project is made available.

## Contact
For any inquiries or support, contact Avery Krieger at krave@stanford.edu.
