import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def conduct_pca(
    data_path: str,
    save_path: str,
    n_components: int = None,
    save_scores: bool = True,
    save_loadings: bool = True,
    save_explained_variance: bool = True,
    save_scaling_params: bool = True,
    save_scree_plot: bool = True,
    save_biplot: bool = True,
    save_pc_scatter: bool = True
):

    df = pd.read_csv(data_path, index_col=0)
    print(f"読み込んだデータの形状: {df.shape}")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    os.makedirs(save_path, exist_ok=True)

    if save_scores:
        scores = pd.DataFrame(pca.transform(scaled_data),
                              index=df.index,
                              columns=[f'PC{i+1}' for i in range(pca.n_components_)])
        scores.to_csv(os.path.join(save_path, 'pca_scores.csv'))
        print("✅ 主成分得点（scores.csv）を保存しました。")

    if save_loadings:
        loadings = pd.DataFrame(pca.components_.T,
                                index=df.columns,
                                columns=[f'PC{i+1}' for i in range(pca.n_components_)])
        loadings.to_csv(os.path.join(save_path, 'pca_loadings.csv'))
        print("✅ 主成分負荷量（loadings.csv）を保存しました。")

    if save_explained_variance:
        explained = pd.DataFrame({
            'ExplainedVarianceRatio': pca.explained_variance_ratio_,
            'CumulativeVarianceRatio': np.cumsum(pca.explained_variance_ratio_)
        }, index=[f'PC{i+1}' for i in range(pca.n_components_)])
        explained.to_csv(os.path.join(save_path, 'pca_explained_variance.csv'))
        print("✅ 寄与率（explained_variance.csv）を保存しました。")

    if save_scaling_params:
        scaling_df = pd.DataFrame({'Mean': scaler.mean_, 'Std': np.sqrt(scaler.var_)}, index=df.columns)
        scaling_df.to_csv(os.path.join(save_path, 'scaling_params.csv'))
        print("✅ 標準化パラメータ（scaling_params.csv）を保存しました。")

    if save_scree_plot:
        plt.figure()
        plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_ * 100)
        plt.plot(range(1, pca.n_components_ + 1),
                 np.cumsum(pca.explained_variance_ratio_) * 100,
                 marker='o', color='red', label='Cumulative')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance (%)')
        plt.title('Scree Plot')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'scree_plot.png'), dpi=300)
        plt.close()
        print("✅ スクリープロット（scree_plot.png）を保存しました。")

    if save_pc_scatter:
        scores_df = pd.DataFrame(pca.transform(scaled_data),
                                 index=df.index,
                                 columns=[f'PC{i+1}' for i in range(pca.n_components_)])
        plt.figure()
        plt.scatter(scores_df['PC1'], scores_df['PC2'])
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title('PC1 vs PC2 Scatter Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pc1_pc2_scatter.png'), dpi=300)
        plt.close()
        print("✅ PC1 vs PC2 散布図（pc1_pc2_scatter.png）を保存しました。")

    if save_biplot and pca.n_components_ >= 2:
        scores_2d = pca.transform(scaled_data)[:, :2]
        loadings_2d = pca.components_[:2, :].T
        plt.figure(figsize=(7, 6))
        plt.scatter(scores_2d[:, 0], scores_2d[:, 1], alpha=0.7)
        for i, var in enumerate(df.columns):
            plt.arrow(0, 0, loadings_2d[i, 0]*3, loadings_2d[i, 1]*3,
                      color='r', alpha=0.5, head_width=0.05)
            plt.text(loadings_2d[i, 0]*3.2, loadings_2d[i, 1]*3.2, var, color='r', ha='center', fontsize=8)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title("Biplot (PC1 vs PC2)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'biplot.png'), dpi=300)
        plt.close()
        print("✅ バイプロット（biplot.png）を保存しました。")

    return pca

conduct_pca(data_path='Park.csv', save_path='PCA results')