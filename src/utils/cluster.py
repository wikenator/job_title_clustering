#!/usr/bin/env python

from time import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from feature_builder import FeatureBuilder as FB

class Cluster(FB):
    def __init__(self):
        super().__init__()

        self.algo = None
        self.label_encoder = LabelEncoder()
        self.n_components = 2
        self.component_columns = ['component_1', 'component_2']

    def setup_pipeline(self, *args, **kwargs):
        pipeline = []

        if kwargs['scaler']:
            if kwargs['scaler'] == 'std':
                pipeline.append(("preprocessor", Pipeline(
                    [
                        ("scaler", StandardScaler())
                    ]
                )))

            elif kwargs['scaler'] == 'minmax':
                pipeline.append(("preprocessor", Pipeline(
                    [
                        ("scaler", MinMaxScaler())
                    ]
                )))

        if kwargs['dim_reduce']:
            if kwargs['dim_reduce'] == 'pca':
                pipeline.append(("dim_reducer", Pipeline(
                    [
                        ("pca", PCA(n_components=self.n_components, random_state=2901))
                    ]
                )))

            elif kwargs['dim_reduce'] == 'umap':
                pipeline.append(("dim_reducer", Pipeline(
                    [
                        ("umap", UMAP(n_components=self.n_components, n_neighbors=5, random_state=2901))
                    ]
                )))

        if not kwargs['cluster']:
            return 'No clustering config defined.'
        
        else:
            if kwargs['cluster'] == 'kmeans' and kwargs['kmeans_kwargs']:
                self.algo = 'kmeans'
                kwargs['kmeans_kwargs']['n_clusters'] = kwargs['n_clusters']
                clusterer = Pipeline(
                    [
                        (
                            "kmeans",
                            KMeans(
                                **kwargs['kmeans_kwargs']
                            )
                        )
                    ]
                )
                pipeline.append(("clusterer", clusterer))

            elif kwargs['cluster'] == 'agglom' and kwargs['agglom_kwargs']:
                self.algo = 'agglom'
                kwargs['agglom_kwargs']['n_clusters'] = kwargs['n_clusters']
                clusterer = Pipeline(
                    [
                        (
                            "agglom",
                            AgglomerativeClustering(
                                **kwargs['agglom_kwargs']
                            ),
                        ),
                    ]
                )
                pipeline.append(("clusterer", clusterer))

            elif kwargs['cluster'] == 'hdbscan' and kwargs['hdbscan_kwargs']:
                self.algo = 'hdbscan'
                clusterer = Pipeline(
                    [
                        (
                            "hdbscan",
                            HDBSCAN(
                                **kwargs['hdbscan_kwargs']
                            ),
                        ),
                    ]
                )
                pipeline.append(("clusterer", clusterer))

        pipe = Pipeline(pipeline)

        return pipe
    
    def run_pipeline(self, pipe, feats, labels):
        '''
        Runs all pipeline steps on the given data.

        @param pipe: sklearn pipeline to run
        @param feats: features to use in the clustering algorithm
        @param labels: output labels for each example
        @return: dataframe of results containing principal components, predicted clusters, and true labels
        '''
        true_labels = self.label_encoder.fit_transform(labels)
        preprocessed_data = pipe["preprocessor"].transform(feats)
        reduced_data = pipe["dim_reducer"].transform(preprocessed_data)
        predicted_labels = pipe["clusterer"][self.algo].labels_

        df = pd.DataFrame(
            reduced_data,
            columns=self.component_columns,
        )
        df["predicted_cluster"] = predicted_labels
        df["true_label"] = self.label_encoder.inverse_transform(true_labels)

        self.pipe_data = preprocessed_data
        self.pred_lbl = predicted_labels
        self.true_lbl = labels

        return df
    
    def calculate_scores(self):
        '''
        Calculate different performance metrics.
        '''

        return {
            'ari': adjusted_rand_score(self.true_lbl, self.pred_lbl),
            'hcv': homogeneity_completeness_v_measure(self.true_lbl, self.pred_lbl),
            'sil': silhouette_score(self.pipe_data, self.pred_lbl),
        }
    
    def find_best_n_components(self, feats, labels, cluster_config):
        sil_scores = []
        ari_scores = []
        hom_scores = []
        com_scores = []
        v_scores = []
        sse = []
        pipe = self.setup_pipeline(**cluster_config)

        for n in range(2, 15):
            pipe["dim_reducer"][cluster_config['dim_reduce']].n_components = n
            self.n_components = n
            self.component_columns = ['component_' + str(i) for i in range(n)]
            pipe.fit(feats)
            self.run_pipeline(pipe, feats, labels)

            scores = self.calculate_scores()
            sil_scores.append(scores['sil'])
            ari_scores.append(scores['ari'])
            hom_scores.append(scores['hcv'][0])
            com_scores.append(scores['hcv'][1])
            v_scores.append(scores['hcv'][2])

            if self.algo == 'kmeans':
                sse.append(pipe[2][0].inertia_)

        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(6, 6))
        plt.plot(range(2, 15), sil_scores, c="blue", label="Silhouette")
        plt.plot(range(2, 15), ari_scores, c='red', label="ARI")
        plt.plot(range(2, 15), hom_scores, c='orange', label="Homogeneity")
        plt.plot(range(2, 15), com_scores, c='green', label="Completeness")
        plt.plot(range(2, 15), v_scores, c='purple', label="V-measure")

        plt.xlabel("n_components")
        plt.legend(loc='best')
        plt.title(f"{cluster_config['cluster']} Performance as a Function of n_components")
        plt.tight_layout()
        plt.show()

        if self.algo == 'kmeans':
            kl = KneeLocator(
                range(2, 15), sse, curve="convex", direction="decreasing"
            )
            print(f"Elbow clusters computed by KneeLocator: {int(kl.elbow)}")

    def show_simple_scatterplot(self, df):
        '''
        Show a simple scatterplot to visualize dimensionality reduction.

        @param df: dataframe containing principal components
        @return: None
        '''

        plt.scatter(df[self.component_columns[0]], df[self.component_columns[1]])
        plt.show()

    def show_scatterplot(self, df):
        '''
        Scatterplot to show predicted clusterings overlaid with true labels.

        @param df: dataframe containing principal components, predicted clusters, and true labels
        @return: None
        '''

        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(6, 6))

        scat = sns.scatterplot(
            x=self.component_columns[0],
            y=self.component_columns[1],
            s=50,
            data=df,
            hue="predicted_cluster",
            style="true_label",
            palette="Set2",
        )

        scat.set_title(f"Clustering results from {self.algo}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

    def topic_analysis(self, data, feats):
        # Authors: The scikit-learn developers
        # SPDX-License-Identifier: BSD-3-Clause
        # Source: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

        n_samples = feats.shape[0]
        n_features = feats.shape[1]
        n_components = 14
        n_top_words = 15
        init = "nndsvda"

        def plot_top_words(model, feature_names, n_top_words, title):
            fig, axes = plt.subplots(3, 5, figsize=(30, 15), sharex=True)
            axes = axes.flatten()
            
            for topic_idx, topic in enumerate(model.components_):
                top_features_ind = topic.argsort()[-n_top_words:]
                top_features = feature_names[top_features_ind]
                weights = topic[top_features_ind]

                ax = axes[topic_idx]
                ax.barh(top_features, weights, height=0.7)
                ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
                ax.tick_params(axis="both", which="major", labelsize=20)
                for i in "top right left".split():
                    ax.spines[i].set_visible(False)
                fig.suptitle(title, fontsize=40)

            plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
            plt.show()

        data = data.apply(lambda x: ' '.join(self.regex_tokenizer(x)))

        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.95, min_df=2, max_features=n_features, tokenizer=self.regex_tokenizer
        )
        t0 = time()
        tfidf = tfidf_vectorizer.fit_transform(data)
        print("done in %0.3fs." % (time() - t0))

        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        tf_vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, max_features=n_features, tokenizer=self.regex_tokenizer
        )
        t0 = time()
        tf = tf_vectorizer.fit_transform(data)
        print("done in %0.3fs." % (time() - t0))
        print()

        # Fit the NMF model
        print(
            "Fitting the NMF model (Frobenius norm) with tf-idf features, "
            "n_samples=%d and n_features=%d..." % (n_samples, n_features)
        )
        t0 = time()
        nmf = NMF(
            n_components=n_components,
            random_state=1,
            init=init,
            beta_loss="frobenius",
            alpha_W=0.00005,
            alpha_H=0.00005,
            l1_ratio=1,
        ).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))

        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        plot_top_words(
            nmf, tfidf_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)"
        )

        # Fit the NMF model
        print(
            "\n" * 2,
            "Fitting the NMF model (generalized Kullback-Leibler "
            "divergence) with tf-idf features, n_samples=%d and n_features=%d..."
            % (n_samples, n_features),
        )
        t0 = time()
        nmf = NMF(
            n_components=n_components,
            random_state=1,
            init=init,
            beta_loss="kullback-leibler",
            solver="mu",
            max_iter=1000,
            alpha_W=0.00005,
            alpha_H=0.00005,
            l1_ratio=0.5,
        ).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))

        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        plot_top_words(
            nmf,
            tfidf_feature_names,
            n_top_words,
            "Topics in PLSA model",
        )

        print(
            "\n" * 2,
            "Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
            % (n_samples, n_features),
        )
        lda = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )
        t0 = time()
        lda.fit(tf)
        print("done in %0.3fs." % (time() - t0))

        tf_feature_names = tf_vectorizer.get_feature_names_out()
        plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")
