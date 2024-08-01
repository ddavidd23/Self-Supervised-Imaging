# Code release

<p align="center">
  ![unet](https://github.com/user-attachments/assets/f4313292-7e28-4b98-8c39-8c1ba4548509)
  ![pipeline](https://github.com/user-attachments/assets/860dd65e-6d9a-4f58-ab36-edc0c8accbea)
</p>

These are companion scripts to the manuscript "Self-Supervised Maize Kernel Classification and Segmentation for Embryo Identification", which has been submitted to the journal Frontiers in Plant Science.

* evaluate.py takes an existing model and formatted data directory to evaluate model performance.
* pipeline.py is a vision pipeline which takes a backbone trained by self-supervision or one which is preloaded, and trains it on labeled data.

The trained models can be found at https://zenodo.org/record/7577017, along with the accompanying dataset.
