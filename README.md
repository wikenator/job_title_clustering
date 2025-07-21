# Job Title Clustering
Clustering job titles for go-to-market roles, campaigns, or touchpoints from customer journey data.

## How to set up the project
All commands can be run from the root directory.

1. Install packages from `requirements.txt`.
```
pip install -r src/requirements.txt
```
2. Run `setup.py` to install additional supporting packages.
```
python src/setup.py
```

## Basic Usage

Import the Cluster class and load your data:
```
import pandas as pd
from src.job_title_clustering.cluster import Cluster

cl = new Cluster()
df = pd.read_csv('./path/to/data.csv')
```

Pass in your raw data to extract features.
```
feats = cl.get_features(data, use_bigrams=False)
```

Configure your processing pipeline.

See scikit-learn's documentation for algorithm-specific arguments.

```
cluster_config = {
    "kmeans_kwargs": {
        "init": "k-means++", # random, k-means++
        "n_init": 50,
        "max_iter": 500,
        # "random_state": 1,
    },
    "hdbscan_kwargs": {
        'min_cluster_size': 15,
        'min_samples': 4,
        'metric': 'euclidean',
    },
    "agglom_kwargs": {
        'metric': 'euclidean',
        'linkage': 'ward',
    },
    "cluster": "", # kmeans, hdbscan, agglom
    "scaler": "", # std, minmax
    "dim_reduce": "", # pca, umap
    "n_clusters": 3,
}
```

Fit your features to the pipeline and run the pipeline.
```
pipe = cl.setup_pipeline(**cluster_config)
pipe.fit(feats)
df = cl.run_pipeline(pipe, feats, labels=None)
```

Continue exploring!

```
# show a simple scatterplot of your reduced dimensionality data
cl.show_simple_scatterplot(df)

# run various metrics on the clustering algorithm you chose
cl.find_best_n_components(feats, cluster_config, labels=None)

# if you have labeled data, show a scatterplot that overlays
# cluster assignments over your reduced dimensionality data
df = cl.run_pipeline(pipe, feats, labels=labels)
cl.show_scatterplot(df)
```