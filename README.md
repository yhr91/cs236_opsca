Course project for aligning data from multi-modal single-cell experiments using generative models. (CS236)

- Further information about the dataset and how to download it is available at:
https://openproblems.bio/neurips_docs/data/dataset/ 

- Please read about AnnData class (adata) here for understanding how single cell data is normally stored and manipulated in Python:
https://anndata.readthedocs.io/en/latest/.
  These are essentially data matrices (adata.X) where each row is a cell and each column is a feature. Along with the data matrix, the AnnData object also contains a dataframe (adata.obs) with metadata for each cell as well as another dataframe (adata.var)with metadata for each feature.

Final results were produced using:
- VAE: model/VAE_trainer.py
- c-GAN: notebooks/conditional_gan.py
