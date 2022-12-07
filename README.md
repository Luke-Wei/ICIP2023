# ICIP2023
## 0 Enviroments

pip install -r requirements.txt

## 1 To use the demo
refer to TLGAN https://github.com/SummitKwan/transparent_latent_gan

Decompress the downloaded files and put it in project directory as the following format, which can be decompressed in transparent_latent_gan_ICIP.7z.

    ```text
    root(d):
      asset_model(d):
        karras2018iclr-celebahq-1024x1024.pkl   # pretrained GAN from Nvidia
        network-final_yitu.pkl
        network-snapshot-007286.pkl
        network-snapshot-008486.pkl
        cnn_face_attr_celeba(d):
          model_20180927_032934.h5              # trained feature extractor network
        dnnlib(d)
      asset_results(d):
        pg_gan_celeba_feature_direction_40(d):
          feature_direction_20181002_044444.pkl # feature axes
        pg_gan_celeba_feature_direction_celebA
        pg_gan_celeba_feature_direction_celebA_givens
        pg_gan_celeba_feature_direction_celebA_house
        pg_gan_celeba_feature_direction_celebA_smitt
        pg_gan_celeba_feature_direction_yitu_givens
        pg_gan_celeba_feature_direction_yitu_house
        pg_gan_celeba_feature_direction_yitu_smitt
        stylegan_ffhq_explore
        stylegan_ffhq_feature_direction_40
        stylegan_ffhq_feature_direction_retrained
        
        
    ```
    
## 2 To use the demo

run the interactive demo from the Jupyter Notebook at ./src/notebooks/tl_gan_ipywidgets_gui_celebA.ipynb

feature direction can be changed by change "path_feature_direction"

