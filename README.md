How to Run Locally

Test Inference Immediately
You can test the trained model on an image immediately after cloning without needing to download the full 4.5 GB feature dataset, as the optimized weights are already tracked in the repository.

```bash
# Clone the repository
git clone [https://github.com/anushkaa-17/Image_Caption_generation.git](https://github.com/anushkaa-17/Image_Caption_generation.git)
cd Image_Caption_generation

# Install dependencies
pip install -r requirements.txt

# Run inference using an image from your local dataset or any custom image path
python test.py --image Flickr8k_Dataset/Flicker8k_Dataset/3710468717_c051d96a5f.jpg
```
* **Pre-extracted Spatial Features & Processed Artifacts:** [Download All Generated Files from Google Drive](https://drive.google.com/drive/folders/1tByitIRH7GsnnqaFJG379El2K3tmYtme?usp=sharing)
* **Raw Dataset:** [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
