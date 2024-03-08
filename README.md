# Nevise: A Bert-Based Spell-Checker for Persian

Nevise is a Persian spelling-checker developed by Dadmatech  Co based on deep learning. Nevise is available in two versions. The second version has greater accuracy, the ability to correct errors based on spaces, and a better understanding of special characters like half space. These versions can be accessed via web services and as demos. We provide public access to the code and model checkpoint of the first version here.

## Quick Start

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Dadmatech/Nevise.git
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r Nevise/requirements.txt
    ```

3. **Download Model Checkpoint and Vocab:**
   Create a directory for the model and the vocab:
    ```bash
    mkdir Nevise/model
    ```

    Download the model and the vocab: 
    ```bash
    gdown -O Nevise/model/model.pth.tar "https://drive.google.com/uc?id=1Ki5WGR4yxftDEjROQLf9Br8KHef95k1F"
    gdown -O Nevise/model/vocab.pkl "https://drive.google.com/uc?id=1nKeMdDnxIJpOv-OeFj00UnhoChuaY5Ns"
    ```

    In case of download failure, you can obtain the download links from your download manager and use the following commands:
    ```bash
    wget -O Nevise/model/model.pth.tar "[DOWNLOAD URL OF model.pth.tar]"
    wget -O Nevise/model/vocab.pkl "[DOWNLOAD URL OF vocab.pkl]"
    ```

5. **Run Spell-Checking:**
    You can then use the model for spelling correction using the following command in which you must replace `"input.txt"` with the path to your input text and `"output.txt"` with the path to the file for writing the corrected text.

    ```bash
    python Nevise/nevise.py --input-file "input.txt" --output-file "output.txt" --vocab-path "Nevise/model/vocab.pkl" --model-checkpoint-path "Nevise/model/model.pth.tar"
    ```

**NOTE:** For a more expressive output of the model including the pairs of wrong and corrected sentences along with the words that had an error and their corrected form, see `main.py`.

# Demo

[Nevise(both versions)](https://dadmatech.ir/#/products/SpellChecker)

# Results on [Nevise Dataset](https://github.com/Dadmatech/Nevise-Dataset/tree/main/nevise-news-title-539)

</br>

| Algorithm | Wrong detection rate | Wrong correction rate | Correct to wrong rate | Precision |
| -- | -- | -- | -- | -- |
| Nevise 2 | **0.8314** | **0.7216** | 0.003 | 0.968 |
| Paknevis | 0.7843 | 0.6706 | 0.228 | 0.7921 |
| Nevise 1 | 0.7647 | 0.6824 | **0.0019** | **0.9774** |
| Google | 0.7392 | 0.702 | 0.0045 | 0.9449 |
| Virastman | 0.6 | 0.5 | 0.0032 | 0.9533 |
