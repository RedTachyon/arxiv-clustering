# arXiv abstract clustering

## Important note

This implementation is meant as a proof of concept. 
Due to the time constraints, both in terms of the deadline and my other obligations; 
as well as the computational cost of embedding all the abstracts, I restricted my analysis to an MVP-like state.

## Literature review and Methodology

(Note: I take some liberties in the formatting of this report by combining the literature review with the methods, considering it's not an official publication.)

My approach consists of three main parts:

- Feature extraction via BERT
- Clustering via K-Means
- Clustering evaluation via Adjusted Rand Index and the Within-Cluster Sum of Squares


BERT (Devlin et al., 2018) is a transformer-based model commonly used for natural language processing tasks. 
It is pre-trained on the task of filling in missing tokens, can be fine-tuned for a wide variety of new tasks. 

In this case, I used the pre-trained model to extract features from the abstracts. 
Specifically, I used the `prajjwal1/bert-small` model from the Hugging Face hub, so that the embeddings are faster to compute, and take up less memory (512-dimensional instead of the default 768).

K-Means (MacQueen, 1967) is a simple clustering algorithm which starts with a random set of cluster centers, and iteratively updates them to minimize the within-cluster sum of squares.
Here, I used the implementation from scikit-learn, with the minibatch variant for computational efficiency.

Finally, I evaluate the resulting clusterings using the Adjusted Rand Index (ARI) (Hubert and Arabie, 1985) and the Within-Cluster Sum of Squares (WCSS).
With ARI, I use some additional information, namely the top-level arXiv categories of the papers. 
The ARI value indicates how well a given clustering matches the categories, with 1 being a perfect match, and 0 being a random clustering.
WCSS on the other hand does not require any additional information, and is simply a measure of how well the clustering fits the data. 

While I did not expect a full match due to the unsupervised nature of the clustering, there should be at least some correlation between categories and clusters.

### References
Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193–218. https://doi.org/10.1007/BF01908075

MacQueen, J. (1965). Some methods for classiﬁcation and analysis of multivariate observations. In Proc. 5th Berkeley Symposium on Math., Stat., and Prob (p. 281).

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171–4186. https://doi.org/10.18653/v1/N19-1423



## Running the code

1. After initializing a virtual environment, install the dependencies with `pip install -r requirements.txt`
2. Run `bash download_data.sh` to download the data from kaggle
3. Run `python process_data.py` to extract the abstracts from the downloaded data, and save them in a separate file
4. Run `python generate_embeddings.py` to generate the BERT-based embeddings for the abstracts. This will likely take a long time and require a decent GPU. I can provide the embeddings on request.
5. Go through the `clustering.ipynb` notebook to see the results of the clustering and the evaluation.


## Results

From the analysis in the notebook, we can see that the clustering does not match the categories very well. 
Nevertheless, the ARI index peaks at 0.267, which is not too bad considering that the clustering is unsupervised. 
Considering this, as well as the shape of the WCSS curve, I propose that the **optimal number of clusters is 9**.


## Limitations

Here I want to address a few limitations, as well as comments on my interpretation of the original task.

1. For brevity, I integrated the literature review with the description of my method, since both are very closely tied together.
2. Exploratory data analysis happened mostly in a few ad-hoc notebooks, which I did not include in the repository.
3. I did not exactly "train an NLP" model -- I am not sure if this was a mistake in the task description, or if I misunderstood it. Training a classification algorithm from scratch would be unsuitable for the unsupervised task at hand. Following the suggestion of a BERT model, it would also be relatively computationally expensive.
4. I did not perform a full hyperparameter search except for the number of clusters, which is the core of this study. From my experience, K-Means tends to be relatively robust to hyperparameter changes, and the number of clusters is the most important one.
5. The model I used is based on PyTorch, as that is what I exclusively use in my own research at the moment. Using a similar TF-based model would be straight-forward using the transformers library, but it would also be additional time spent on a task that is not the focus of this study. 
6. There are many visualizations and analyses that could be done to better understand the data, and the clustering results. Examples include -- a t-SNE visualization of the abstracts, their categories and clustering; a matrix similar to a confusion matrix, correlating categories with clusters; other metrics of clustering quality. However, at some point my regular life had to take priority, and I had to stop somewhere.