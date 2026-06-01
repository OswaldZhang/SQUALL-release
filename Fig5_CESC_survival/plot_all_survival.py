from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, mannwhitneyu
import os
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.family'] = 'DejaVu Sans'

# Base path settings
data_path = '/data200T/SQUALL/SQUALL_cls'
clustering_results_path = '/lustre1/zxzeng/bwqin/SQUALL_main/clustering/OV/clustering_results'

def get_dominant_pathotype_per_cluster(adata, cluster_key='hc_clusters', patho_key='annotation'):
    """
    计算每个 cluster 对应的主要病理类型（pathotype）
    
    Parameters:
    -----------
    adata : AnnData
        含有 cluster 和 pathotype 信息的 AnnData 对象
    cluster_key : str
        表示聚类结果的 obs 列名（默认 'hc_clusters'）
    patho_key : str
        表示病理注释的 obs 列名（默认 'pathotype'）
    
    Returns:
    --------
    cluster_to_pathotype : dict
        每个 cluster 的主导病理类型（最多的那个）
    """
    df = adata.obs[[cluster_key, patho_key]].dropna()

    # 确保 cluster 是字符串（便于一致性）
    df[cluster_key] = df[cluster_key].astype(str)

    # 分组并找出每个 cluster 中数量最多的 pathotype
    dominant = df.groupby(cluster_key)[patho_key] \
                 .agg(lambda x: x.value_counts().idxmax())

    cluster_to_pathotype = dominant.to_dict()
    return cluster_to_pathotype


def load_oc_data(fdr_threshold=0.05,n_clusters = 50):
    """
    Load OC tumor data and retain only clusters significantly enriched in 'tumor' pathotype.

    Parameters:
    -----------
    fdr_threshold : float
        FDR cutoff for defining enrichment significance

    Returns:
    --------
    adata : AnnData
        Filtered AnnData object with only tumor-enriched clusters
    clinical_df : DataFrame
        Clinical metadata
    cluster_props : DataFrame
        Cluster proportions per sample (Pathology_ID) after filtering
    """
    import scanpy as sc
    import pandas as pd
    import numpy as np
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import fisher_exact
    def simplify_annotation(x):
        if x in ['Calcification"', 'Paramesonephric duct cyst', 'Benign cyst']:
            return 'Other pathological conditions'
        elif "Calcification" in x:
            return 'Other pathological conditions'
        elif x in ['Normal ovary', 'Normal glandular epithelium', 'Normal fallopian tube']:
            return 'Normal parenchyma'
        elif x in ['Simple stroma', 'Complex stroma']:
            return 'Stroma'
        elif x == 'Unknown':
            return None
        else:
            return x
    
    # Load data
    #adata = sc.read_h5ad("oc_with_kmeans_and_annotation.h5ad")
    #adata = sc.read_h5ad("oc_kmeans_5_26.h5ad")
    #adata = sc.read_h5ad("oc_with_kmeans_40_annotatioins.h5ad")
    adata = sc.read_h5ad(f"cluster_outputs/oc_with_hc_{n_clusters}_annotation.h5ad")
    clinical_df = pd.read_csv("hm_GC.csv")
    clinical_df = pd.read_csv("hm_GC_with_term.csv")
    adata.obs["hc_clusters"] = adata.obs[f"hc_clusters_{n_clusters}"].astype(str)
    adata.obs["Pathotype"] = adata.obs["annotation"].map(simplify_annotation)  # assume annotation column contains pathotype info
    adata = adata[~adata.obs["Pathotype"].isna()].copy()
    # Build contingency table
    print("adata.obs['Pathotype']",adata.obs["Pathotype"])
    cross_tab = pd.crosstab(adata.obs["Pathotype"], adata.obs["hc_clusters"])
    pathotypes = cross_tab.index
    clusters = cross_tab.columns

    # Compute Fisher's exact test for tumor vs. all others
    results = []
    for cluster in clusters:
        for patho in pathotypes:
            a = cross_tab.loc[patho, cluster]
            b = cross_tab.loc[patho].sum() - a
            c = cross_tab.loc[:, cluster].sum() - a
            d = cross_tab.values.sum() - (a + b + c)

            table = [[a, b], [c, d]]
            try:
                odds, p = fisher_exact(table)
            except:
                odds, p = np.nan, 1.0

            results.append({
                "cluster": cluster,
                "pathotype": patho,
                "odds_ratio": odds,
                "pval": p,
                "a": a
            })

    # Combine and adjust p-values
    results_df = pd.DataFrame(results)
    results_df["fdr"] = multipletests(results_df["pval"], method="fdr_bh")[1]
    tumor_enriched_clusters = results_df[
        (results_df["pathotype"].str.lower() == "tumor") &
        (results_df["fdr"] < fdr_threshold)
    ]["cluster"].unique().tolist()

    print(f"✅ Tumor-enriched clusters (FDR < {fdr_threshold}): {tumor_enriched_clusters}")

    # Filter adata to keep only tumor-enriched clusters
    adata = adata[adata.obs["hc_clusters"].isin(tumor_enriched_clusters)].copy()

    # Recompute cluster proportions
    cluster_counts = (
        adata.obs
        .groupby(["pathology_id", "hc_clusters"])
        .size()
        .unstack(fill_value=0)
    )
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100

    return adata, clinical_df, cluster_props

'''
def load_oc_data():
    """
    Load OC tumor data from updated files:
    - AnnData with hc_clusters and annotation
    - Metadata with clinical information
    
    Returns:
    --------
    adata : AnnData
        AnnData object with clustering and annotation
    clinical_df : DataFrame
        Clinical metadata from hm_GC.csv
    cluster_props : DataFrame
        Cluster proportions per sample (Pathology_ID)
    """
    # Load AnnData
    adata = sc.read_h5ad("oc_with_kmeans_and_annotation.h5ad")
    
    # Load metadata
    clinical_df = pd.read_csv("hm_GC.csv")
    
    # Ensure clusters are string type
    adata.obs["hc_clusters"] = adata.obs["hc_clusters"].astype(str)
    
    # Compute cluster proportions per pathology_id
    cluster_counts = (
        adata.obs
        .groupby(["pathology_id", "hc_clusters"])
        .size()
        .unstack(fill_value=0)
    )
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100
    
    return adata, clinical_df, cluster_props
'''

def create_meta_features(cluster_props):
    """
    Create meta-features such as ratios, indicators, and squares from cluster proportions.
    
    Parameters:
    -----------
    cluster_props : DataFrame
        DataFrame of cluster proportions per sample.
    
    Returns:
    --------
    meta_features : DataFrame
        Enhanced feature matrix with meta-features.
    """
    meta_features = cluster_props.copy()
    
    # List of cluster column names (assumed as strings)
    clusters = list(cluster_props.columns)
    
    # Create ratios of adjacent clusters
    for i, cluster_i in enumerate(clusters):
        for j in range(i+1, min(i+3, len(clusters))):
            cluster_j = clusters[j]
            ratio_col = f"ratio_{cluster_i}_{cluster_j}"
            meta_features[ratio_col] = cluster_props[cluster_i] / (cluster_props[cluster_j] + 0.1)
    
    # Indicator if the cluster is max per row
    for cluster in clusters:
        is_max_col = f"is_max_{cluster}"
        meta_features[is_max_col] = (cluster_props[cluster] == cluster_props[clusters].max(axis=1)).astype(float)
    
    # Square terms to capture non-linearity
    for cluster in clusters:
        sq_col = f"sq_{cluster}"
        meta_features[sq_col] = cluster_props[cluster] ** 2
    
    return meta_features


def analyze_oc_platinum_resistance(n_clusters = 50,correlation_results_path = None):
    """Analyze platinum resistance correlation for OC tumors"""
    print(f"\n{'='*50}")
    print(f"Analyzing platinum resistance correlation for OC tumors")
    print(f"{'='*50}")

    # Create output directory
    output_dir = os.path.join(correlation_results_path, 'OC', 'Platinum')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    adata, clinical_df, cluster_props = load_oc_data(n_clusters = n_clusters)
    cluster_to_pathotype = get_dominant_pathotype_per_cluster(adata)
    if adata is None or clinical_df is None or cluster_props is None:
        print("Unable to load OC data, skipping analysis")
        return

    # Filter samples with platinum resistance data
    clinical_platinum = clinical_df[clinical_df['Platinum_Resistance'].isin(['sensitive', 'resistance'])].copy()
    print("clinical_platinum",clinical_platinum)
    print(len(clinical_platinum))
    print(f"Platinum resistance data distribution: {clinical_platinum['Platinum_Resistance'].value_counts().to_dict()}")

    # Perform enhanced correlation analysis
    #results_df = analyze_correlation_enhanced(cluster_props, clinical_platinum, 'Platinum_Resistance')
    results_df = analyze_cluster_vs_recurrence(cluster_props, clinical_platinum, 'Platinum_Resistance')
    if results_df.empty:
        print("Unable to perform platinum resistance correlation analysis for OC")
        return

    # Save results
    results_path = os.path.join(output_dir, 'OC_platinum_correlation.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    # Create forest plot
    #forest_path = os.path.join(output_dir, 'OC_platinum_forest_plot.png')
    #create_forest_plot(results_df, 'OC', 'Platinum_Resistance', forest_path,cluster_to_pathotype = cluster_to_pathotype)
    dot_path = os.path.join(output_dir, f'{tumor_type}_platinum_dotplot.png')

    create_dotplot(results_df, tumor_type, 'Platinum_Resistance', dot_path,cluster_to_pathotype = cluster_to_pathotype)
    # Get significant features (p < 0.1)
    sig_features = results_df[results_df['P_Value_Min'] < 0.1]['Feature'].tolist()

    if sig_features:
        # Create heatmap
        heatmap_path = os.path.join(output_dir, 'OC_platinum_heatmap.png')
        create_heatmap(cluster_props, clinical_platinum, 'OC', 'Platinum_Resistance', sig_features, heatmap_path)

        # Create boxplots
        create_boxplots(cluster_props, clinical_platinum, 'OC', 'Platinum_Resistance', sig_features, output_dir)

    # Create ROC curves
    roc_path = os.path.join(output_dir, 'OC_platinum_roc.png')
    create_roc_plot(cluster_props, clinical_platinum, 'OC', 'Platinum_Resistance', results_df, roc_path)

    # Print significant findings summary
    sig_results = results_df[results_df['P_Value_Min'] < 0.05]
    print(f"\nSignificant platinum resistance correlations for OC (p < 0.05):")

    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            if row['Feature_Type'] == 'cluster':
                print(f"- Cluster {row['Feature']}: Min P-value={row['P_Value_Min']:.3f}, "
                      f"OR={row['Odds_Ratio']:.2f}, AUC={row['AUC']:.3f}, "
                      f"Mean Diff={row['Mean_Diff']:.2f}%")
    else:
        print("No significant correlations found.")
def simplify_annotation(x):
    if x in ['Calcification"', 'Paramesonephric duct cyst', 'Benign cyst']:
        return 'Other pathological conditions'
    elif "Calcification" in x:
        return 'Other pathological conditions'
    elif x in ['Normal ovary', 'Normal glandular epithelium', 'Normal fallopian tube']:
        return 'Normal parenchyma'
    elif x in ['Simple stroma', 'Complex stroma']:
        return 'Stroma'
    elif x == 'Unknown':
        return None
    else:
        return x

def get_cluster_dominant_pathotype(cluster_props, meta_info, cluster_columns, pathotype_col="Pathology_Check"):
    """
    Return a dict mapping cluster -> most dominant pathotype
    """
    print("cluster_props",cluster_props.keys())
    print("meta_info['Pathology_Check']",meta_info['Pathology_Check'])
    meta_info['Pathology_Check'] = meta_info['Pathology_Check'].apply(simplify_annotation)
    cluster_pathotype_map = {}
    for cluster in cluster_columns:
        df = pd.DataFrame({
            "value": cluster_props[cluster],
            "pathotype": meta_info[pathotype_col]
        })
        # 取该 cluster 中 value 的总和，按 pathotype 分组
        dominant = df.groupby("pathotype")["value"].sum().sort_values(ascending=False)
        if not dominant.empty:
            cluster_pathotype_map[cluster] = dominant.index[0]
        else:
            cluster_pathotype_map[cluster] = "UNK"
    return cluster_pathotype_map


def create_forest_plot(results_df, tumor_type, clinical_var, output_path,cluster_to_pathotype):
    """Create forest plot showing odds ratios and confidence intervals"""
    # Sort by significance (most significant at the top)
    #results_df = results_df.sort_values('P_Value_Min')
    results_df = results_df.sort_values('P_Value_Min_adj')
    print("results_df.keys()",results_df.keys())
    # Only keep basic clusters (not meta-features)
    cluster_results = results_df[results_df['Feature_Type'] == 'cluster'].copy()
    annotation_colors = {
    "Adipose tissue": "#fdb863",
    "Debris": "#e66101",
    "Immune cells": "#5e3c99",
    "Normal parenchyma": "#b2abd2",
    "Other pathological conditions": "#80cdc1",
    "Stroma": "#018571",
    "Tumor": "#d7191c"
    }
    if len(cluster_results) == 0:
        print(f"No basic cluster results to plot")
        return
    '''
    # Take top 15 most significant clusters
    if len(cluster_results) > 15:
        cluster_results = cluster_results.head(15)
    '''

    plt.figure(figsize=(12, max(6, len(cluster_results) * 0.4)))
    
    # Limit very large upper confidence intervals for better visualization
    cluster_results['CI_Upper'] = np.minimum(cluster_results['CI_Upper'], 20)
    # Plot each cluster
    for i, (_, row) in enumerate(cluster_results.iterrows()):
        # Plot confidence interval as horizontal line
        plt.plot(
            [max(0.01, row['CI_Lower']), row['CI_Upper']], 
            [i, i], 
            'b-', 
            alpha=0.6
        )
        
        # Plot odds ratio as point
        #marker_color = 'red' if row['P_Value_Min'] < 0.05 else ('orange' if row['P_Value_Min'] < 0.1 else 'blue')
        #marker_color = 'red' if row['P_Value_Min_adj'] < 0.05 else ('orange' if row['P_Value_Min_adj'] < 0.1 else 'blue')
        cluster_label = row['Feature']
        dom_patho = cluster_to_pathotype.get(str(cluster_label), "Other pathological conditions")
        marker_color = annotation_colors.get(dom_patho, 'gray')
        plt.plot(row['Odds_Ratio'], i, 'o', color=marker_color, markersize=8)
        
        # Add p-value and cluster label
        if row['P_Value_Min_adj'] < 0.05:
            p_text = f"p={row['P_Value_Min_adj']:.3f} *"
        elif row['P_Value_Min_adj'] < 0.1:
            p_text = f"p={row['P_Value_Min_adj']:.3f} †"
        else:
            p_text = f"p={row['P_Value_Min_adj']:.3f}"
        
        plt.text(
            max(row['CI_Upper'] + 0.5, 2.5), 
            i, 
            p_text, 
            va='center'
        )
    
    # Add vertical reference line (OR=1)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # Customize plot
    plt.yticks(range(len(cluster_results)), [f"Cluster {c}" for c in cluster_results['Feature']])
    plt.xlabel('Odds Ratio (95% Confidence Interval)')
    plt.ylabel('Cluster')
    
    var_label = "Recurrence" if clinical_var == "Recurrence_Status" else "Platinum Resistance"
    plt.title(f'{tumor_type} - {var_label}\nCluster Correlations (Odds Ratio)')
    
    # Set reasonable x-axis limits
    plt.xlim(0, min(20, cluster_results['CI_Upper'].max() * 1.2))
    
    plt.grid(axis='x', alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=ptype) for ptype, color in annotation_colors.items()]
    plt.legend(handles=legend_elements, title="Dominant Pathotype", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Forest plot saved to: {output_path}")

def create_heatmap(cluster_props, clinical_df, tumor_type, clinical_var, sig_features, output_path):
    """Create heatmap showing distribution of significant clusters across samples"""
    # Prepare heatmap data
    if clinical_var == 'Platinum_Resistance':
        # Map to binary (0=sensitive, 1=resistance)
        clinical_df['target'] = clinical_df[clinical_var].map({'sensitive': 0, 'resistance': 1})
        labels = ['Sensitive', 'Resistant']
    else:
        # Already binary
        clinical_df['target'] = clinical_df[clinical_var]
        labels = ['No Recurrence', 'Recurrence']
    
    # Merge data
    merged_df = clinical_df.merge(
        cluster_props, 
        left_on='Pathology_ID', 
        right_index=True, 
        how='inner'
    )
    
    # Filter to keep only basic clusters (not meta-features)
    #cluster_features = [f for f in sig_features if f.isdigit()]
    cluster_features = [col for col in cluster_props.columns if col.isdigit()]
    # Ensure we have valid significant clusters in the data
    valid_clusters = [c for c in cluster_features if c in merged_df.columns]
    
    if not valid_clusters:
        print(f"{tumor_type} - {clinical_var} No significant clusters found for heatmap")
        return
    cluster_means = merged_df[valid_clusters].mean().sort_values(ascending=False)
    valid_clusters_sorted = cluster_means.index.tolist()
    # Sort by clinical variable
    merged_df = merged_df.sort_values('target')
    merged_df_z = merged_df.copy()
    #merged_df_z[valid_clusters] = merged_df[valid_clusters].apply(zscore, axis=0)
    merged_df_z[valid_clusters_sorted] = merged_df[valid_clusters_sorted].apply(zscore, axis=0)
    merged_df_z[valid_clusters_sorted] = pd.DataFrame(
    zscore(merged_df_z[valid_clusters_sorted], axis=1),
    index=merged_df_z.index,
    columns=valid_clusters_sorted
    )
    # Create heatmap
    plt.figure(figsize=(12, max(8, len(merged_df) * 0.3)))
    row_colors = merged_df['target'].map({0: 'blue', 1: 'red'})
    # Main heatmap
    '''
    ax = sns.heatmap(
        merged_df_z[valid_clusters_sorted],
        #merged_df_z[valid_clusters],
        cmap='viridis',
        yticklabels=merged_df['Pathology_ID'],
        cbar_kws={'label': 'Cluster Percentage (%)'},
        row_cluster=False,  # 如果不想对 sample 聚类
        col_cluster=True,   # Cluster 列聚类
        )
    
    ax = sns.clustermap(
    merged_df_z[valid_clusters_sorted],
    cmap='vlag',
    center=0,
    figsize=(14, 10),
    row_cluster=False,  # 如果不想对 sample 聚类
    col_cluster=True,   # Cluster 列聚类
    yticklabels=merged_df['Pathology_ID']
    )
    '''
    g = sns.clustermap(
    merged_df_z[valid_clusters_sorted],
    cmap='vlag',
    center=0,
    figsize=(14, 10),
    row_cluster=True,
    row_colors=row_colors,
    col_cluster=True,
    yticklabels=merged_df['Pathology_ID']
    )
    # Add clinical variable annotation
    ax2 = plt.axes([0.92, 0.1, 0.02, 0.8])
    ax2.imshow(
        merged_df['target'].values.reshape(-1, 1),
        cmap='bwr',
        aspect='auto',
        vmin=0,
        vmax=1
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label=labels[0]),
        Patch(facecolor='red', label=labels[1])
    ]
    ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0, -0.05))
    
    # Title
    var_label = "Recurrence" if clinical_var == "Recurrence_Status" else "Platinum Resistance"
    plt.suptitle(f'{tumor_type} - Significant Clusters Associated with {var_label}', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Heatmap saved to: {output_path}")

def create_boxplots(cluster_props, clinical_df, tumor_type, clinical_var, sig_features, output_dir):
    """Create boxplots for each significant feature"""
    # Prepare boxplot data
    if clinical_var == 'Platinum_Resistance':
        # Map values
        y_dict = {'sensitive': 0, 'resistance': 1}
        clinical_df['target'] = clinical_df[clinical_var].map(y_dict)
        labels = ['Sensitive', 'Resistant']
    else:
        # Already binary
        clinical_df['target'] = clinical_df[clinical_var]
        labels = ['No Recurrence', 'Recurrence']
    
    # Merge data
    merged_df = clinical_df.merge(
        cluster_props, 
        left_on='Pathology_ID', 
        right_index=True, 
        how='inner'
    )
    
    # Filter to keep only basic clusters (not meta-features)
    cluster_features = [f for f in sig_features if f.isdigit()]
    
    # Create boxplot for each significant cluster
    for feature in cluster_features:
        if feature not in merged_df.columns:
            continue
        
        plt.figure(figsize=(8, 6))
        
        # Group by clinical variable
        data_0 = merged_df[merged_df['target'] == 0][feature]
        data_1 = merged_df[merged_df['target'] == 1][feature]
        
        # Create boxplot
        boxplot_data = [data_0, data_1]
        bp = plt.boxplot(
            boxplot_data,
            labels=labels,
            patch_artist=True,
            medianprops={'color': 'black'}
        )
        
        # Add scatter points
        for i, data in enumerate(boxplot_data):
            plt.scatter(
                [i+1] * len(data),
                data,
                alpha=0.7,
                s=30,
                color='red' if i == 1 else 'blue',
                edgecolor='black'
            )
        
        # Customize colors
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        # Add statistical test
        if len(data_0) > 0 and len(data_1) > 0:
            # t-test
            if len(data_0) > 1 and len(data_1) > 1:
                t_stat, p_value_t = ttest_ind(data_0, data_1, equal_var=False)
            else:
                p_value_t = 1.0
            
            # Mann-Whitney U test
            try:
                _, p_value_mw = mannwhitneyu(data_0, data_1, alternative='two-sided')
            except:
                p_value_mw = 1.0
            
            # Use minimum p-value
            p_value = min(p_value_t, p_value_mw)
        else:
            p_value = 1.0
        
        # Calculate mean percentage difference
        mean_diff = data_1.mean() - data_0.mean() if len(data_0) > 0 and len(data_1) > 0 else 0
        
        var_label = "Recurrence" if clinical_var == "Recurrence_Status" else "Platinum Resistance"
        plt.title(f'{tumor_type} - Cluster {feature} vs {var_label}\np-value = {p_value:.3f}, Difference = {mean_diff:.2f}%')
        
        plt.ylabel('Cluster Percentage (%)')
        plt.grid(axis='y', alpha=0.3)
        
        output_path = os.path.join(output_dir, f'{tumor_type}_{clinical_var}_cluster{feature}_boxplot.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Boxplot saved to: {output_path}")

def create_roc_plot(cluster_props, clinical_df, tumor_type, clinical_var, results_df, output_path):
    """Create ROC curves for predicting binary clinical outcomes"""
    # Prepare binary target variable
    if clinical_var == 'Platinum_Resistance':
        # Map to binary (0=sensitive, 1=resistance)
        y_dict = {'sensitive': 0, 'resistance': 1}
        clinical_df['target'] = clinical_df[clinical_var].map(y_dict)
        pos_label = "Platinum Resistant"
        neg_label = "Platinum Sensitive"
    else:
        # Already binary
        clinical_df['target'] = clinical_df[clinical_var]
        pos_label = "Recurrence"
        neg_label = "No Recurrence"
    
    # Filter out samples with missing target
    clinical_filtered = clinical_df[clinical_df['target'].notna()].copy()
    
    # Merge cluster data
    merged_df = clinical_filtered.merge(
        cluster_props, 
        left_on='Pathology_ID', 
        right_index=True, 
        how='inner'
    )
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot random classifier reference line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
    
    # Sort by AUC and get top 10 features
    if not results_df.empty:
        # Filter out basic clusters (not meta-features)
        cluster_results = results_df[results_df['Feature_Type'] == 'cluster'].copy()
        
        # Sort by AUC
        top_aucs = cluster_results.sort_values('AUC', ascending=False)
        
        # Select up to 10 best-performing clusters
        top_features = top_aucs.head(min(10, len(top_aucs)))
        
        # Plot ROC curves
        auc_values = {}
        
        for _, row in top_features.iterrows():
            feature = row['Feature']
            X = merged_df[feature]
            y = merged_df['target']
            
            try:
                fpr, tpr, _ = roc_curve(y, X)
                roc_auc = auc(fpr, tpr)
                
                if roc_auc >= 0.5:  # Only show curves better than random guessing
                    plt.plot(
                        fpr, 
                        tpr, 
                        lw=2, 
                        label=f'Cluster {feature} (AUC = {roc_auc:.3f})'
                    )
                    auc_values[feature] = roc_auc
            except Exception as e:
                print(f"Error plotting ROC curve for cluster {feature}: {e}")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    var_label = "Recurrence" if clinical_var == "Recurrence_Status" else "Platinum Resistance"
    plt.title(f'{tumor_type} - ROC Curves for Predicting {var_label}')
    plt.legend(loc="lower right")
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"ROC curves saved to: {output_path}")
    
    # Create boxplot of highest AUC clusters
    if auc_values:
        # Sort clusters by AUC
        sorted_clusters = sorted(auc_values.items(), key=lambda x: x[1], reverse=True)
        top_clusters = [c for c, _ in sorted_clusters[:min(5, len(sorted_clusters))]]
        
        # Create boxplot for top clusters
        plt.figure(figsize=(12, 6))
        
        pos = []
        labels = []
        colors = []
        
        for i, cluster in enumerate(top_clusters):
            # Group by target
            data_0 = merged_df[merged_df['target'] == 0][cluster]
            data_1 = merged_df[merged_df['target'] == 1][cluster]
            
            positions = [i*3, i*3+1]
            pos.extend(positions)
            labels.extend([f'{neg_label}\nCluster{cluster}', f'{pos_label}\nCluster{cluster}'])
            
            box_data = [data_0, data_1]
            
            bp = plt.boxplot(
                box_data, 
                positions=positions, 
                widths=0.6, 
                patch_artist=True,
                medianprops={'color': 'black'}
            )
            
            # Customize colors
            for j, patch in enumerate(bp['boxes']):
                color = 'lightblue' if j == 0 else 'lightcoral'
                patch.set_facecolor(color)
                colors.append(color)
            
            # Add scatter points
            for j, data in enumerate(box_data):
                x_pos = positions[j]
                plt.scatter(
                    [x_pos] * len(data), 
                    data, 
                    alpha=0.7, 
                    s=30, 
                    color='blue' if j == 0 else 'red', 
                    edgecolor='black'
                )
            
            # Add AUC value as text
            plt.text(
                i*3 + 0.5, 
                merged_df[cluster].max() * 1.1,
                f'AUC = {auc_values[cluster]:.3f}',
                ha='center'
            )
        
        plt.xticks(pos, labels)
        plt.ylabel('Cluster Percentage (%)')
        plt.title(f'{tumor_type} - Top Clusters for Predicting {var_label}')
        plt.grid(axis='y', alpha=0.3)
        
        # Save image
        output_dir = os.path.dirname(output_path)
        output_path = os.path.join(output_dir, f'{tumor_type}_{clinical_var}_top_clusters_boxplot.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Top clusters boxplot saved to: {output_path}")



def analyze_tumor_recurrence(tumor_type,n_clusters=50,correlation_results_path = None):
    """分析特定肿瘤类型的复发相关性"""
    print(f"\n{'='*50}")
    print(f"分析 {tumor_type} 肿瘤的复发相关性")
    print(f"{'='*50}")
    
    # 创建输出目录
    output_dir = os.path.join(correlation_results_path, tumor_type, 'Recurrence')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    adata, clinical_df, cluster_props = load_oc_data(n_clusters=n_clusters)
    
    if adata is None or clinical_df is None or cluster_props is None:
        print(f"无法加载 {tumor_type} 的数据，跳过分析")
        return
    
    # 过滤有复发数据的样本
    clinical_recurrence = clinical_df[~clinical_df['Recurrence_Status'].isna()].copy()
    cluster_to_pathotype = get_dominant_pathotype_per_cluster(adata)
    print(f"复发数据分布: {clinical_recurrence['Recurrence_Status'].value_counts().to_dict()}")
    
    # 执行增强的相关性分析
    #results_df = analyze_correlation_enhanced(cluster_props, clinical_recurrence, 'Recurrence_Status')
    results_df = analyze_cluster_vs_recurrence(cluster_props, clinical_recurrence, 'Recurrence_Status')
    if results_df.empty:
        print(f"无法为 {tumor_type} 执行复发相关性分析")
        return
    
    # 保存结果
    results_path = os.path.join(output_dir, f'{tumor_type}_recurrence_correlation.csv')
    results_df.to_csv(results_path, index=False)
    print(f"结果保存到: {results_path}")
    
    # 创建森林图
    forest_path = os.path.join(output_dir, f'{tumor_type}_recurrence_forest_plot.png')
    #create_forest_plot(results_df, tumor_type, 'Recurrence_Status', forest_path,cluster_to_pathotype = cluster_to_pathotype)
    dot_path = os.path.join(output_dir, f'{tumor_type}_recurrence_dotplot.png')
    
    create_dotplot(results_df, tumor_type, 'Recurrence_Status', dot_path,cluster_to_pathotype = cluster_to_pathotype)
    # 获取显著特征 (p < 0.1)
    sig_features = results_df[results_df['P_Value_Min'] < 0.1]['Feature'].tolist()
    
    if sig_features:
        # 创建热图
        heatmap_path = os.path.join(output_dir, f'{tumor_type}_recurrence_heatmap.png')
        create_heatmap(cluster_props, clinical_recurrence, tumor_type, 'Recurrence_Status', sig_features, heatmap_path)
        
        # 创建箱线图
        create_boxplots(cluster_props, clinical_recurrence, tumor_type, 'Recurrence_Status', sig_features, output_dir)
    
    # 创建ROC曲线
    roc_path = os.path.join(output_dir, f'{tumor_type}_recurrence_roc.png')
    create_roc_plot(cluster_props, clinical_recurrence, tumor_type, 'Recurrence_Status', results_df, roc_path)
    
    # 打印显著发现摘要
    sig_results = results_df[results_df['P_Value_Min'] < 0.05]
    print(f"\n{tumor_type} 的显著复发相关性 (p < 0.05):")
    
    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            if row['Feature_Type'] == 'cluster':
                print(f"- 聚类 {row['Feature']}: 最小P值={row['P_Value_Min']:.3f}, "
                      f"OR={row['Odds_Ratio']:.2f}, AUC={row['AUC']:.3f}, "
                      f"平均差异={row['Mean_Diff']:.2f}%")
    else:
        print("未找到显著相关性。")

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def analyze_cluster_vs_recurrence(cluster_props, clinical_df, label_col='Recurrence_Status', min_samples=2):
    """
    对每个 cluster 的占比在 recurrence vs. non-recurrence 样本中进行比较，返回统计显著性和方向。
    
    Parameters
    ----------
    cluster_props : DataFrame
        样本 × cluster 的占比表（index 为 sample ID，columns 为 cluster ID）
    clinical_df : DataFrame
        包含 sample 的临床标签，至少要有 ['Pathology_ID', label_col]
    label_col : str
        指定用于比较的二分类列名（如 'Recurrence_Status'）
    min_samples : int
        每组最少样本数
    
    Returns
    -------
    result_df : DataFrame
        含 p 值、FDR、方向（正负）、均值差等的结果表
    """
    results = []

    for cluster in cluster_props.columns:
        merged = cluster_props[[cluster]].merge(clinical_df[['Pathology_ID', label_col]], 
                                                left_index=True, right_on='Pathology_ID')

        if merged[label_col].nunique() != 2:
            continue

        group_0 = merged[merged[label_col] == 0][cluster]
        group_1 = merged[merged[label_col] == 1][cluster]

        if len(group_0) < min_samples or len(group_1) < min_samples:
            continue

        try:
            stat, pval = mannwhitneyu(group_0, group_1, alternative='two-sided')
            direction = 'Positive' if group_1.mean() > group_0.mean() else 'Negative'
            results.append({
                'Cluster': cluster,
                'Mean_0': group_0.mean(),
                'Mean_1': group_1.mean(),
                'Mean_Diff': group_1.mean() - group_0.mean(),
                'Direction': direction,
                'P_Value': pval
            })
        except Exception as e:
            print(f"Error in cluster {cluster}: {e}")

    result_df = pd.DataFrame(results)
    result_df["Feature_Type"] = "cluster"
    if not result_df.empty:
        result_df["P_Value_adj"] = multipletests(result_df["P_Value"], method="fdr_bh")[1]
        result_df["log10P_adj"] = -np.log10(result_df["P_Value_adj"] + 1e-10)
    result_df["P_Value_Min_adj"] = result_df["P_Value_adj"]
    return result_df





def analyze_correlation_enhanced(cluster_props, clinical_df, clinical_var):
    """
    Analyze correlations between cluster proportions and clinical variables using multiple methods
    
    Parameters:
    -----------
    cluster_props : DataFrame
        Cluster proportions (%) for each sample
    clinical_df : DataFrame
        Clinical data
    clinical_var : str
        Clinical variable to analyze (e.g., 'Recurrence_Status', 'Platinum_Resistance')
    
    Returns:
    --------
    results_df : DataFrame
        DataFrame containing correlation results
    """
    # Prepare target variable
    if clinical_var == 'Platinum_Resistance':
        # Convert 'sensitive' to 0, 'resistance' to 1
        y_dict = {'sensitive': 0, 'resistance': 1}
        clinical_df['target'] = clinical_df[clinical_var].map(y_dict)
    else:
        # Assume it's already numeric
        clinical_df['target'] = clinical_df[clinical_var]
    
    # Merge cluster data with clinical data
    merged_df = clinical_df.merge(
        cluster_props, 
        left_on='Pathology_ID', 
        right_index=True, 
        how='inner'
    )
    
    print(f"Number of samples after merging: {len(merged_df)}")
    
    # Create meta-features
    meta_features = create_meta_features(cluster_props)
    merged_meta_df = clinical_df.merge(
        meta_features,
        left_on='Pathology_ID',
        right_index=True,
        how='inner'
    )
    
    # Prepare result container
    results = []
    
    # Get all feature columns
    all_feature_cols = [col for col in merged_meta_df.columns if col != 'target' and 
                        col != 'Pathology_ID' and col != clinical_var]
    
    # First, perform multiple correlation tests for each basic cluster
    cluster_columns = [col for col in cluster_props.columns if col.isdigit()]
    
    for cluster in cluster_columns:
        try:
            X = merged_df[cluster]
            y = merged_df['target']
            
            # Skip clusters with zero variance
            if X.std() == 0:
                print(f"Skipping cluster {cluster} - zero variance")
                continue
            
            # Method 1: Logistic regression
            X_sm = sm.add_constant(X)
            try:
                model = sm.Logit(y, X_sm).fit(disp=0)
                p_value_logit = model.pvalues[cluster]
                odds_ratio = np.exp(model.params[cluster])
                ci_lower = np.exp(model.conf_int().loc[cluster, 0])
                ci_upper = np.exp(model.conf_int().loc[cluster, 1])
            except Exception as e:
                print(f"Logistic regression failed, trying alternative methods: {e}")
                p_value_logit = 1.0
                odds_ratio = 1.0
                ci_lower = 0.0
                ci_upper = 0.0
            
            # Method 2: Mann-Whitney U test
            group_0 = X[y == 0]
            group_1 = X[y == 1]
            if len(group_0) > 0 and len(group_1) > 0:
                try:
                    # Use U test to compare two groups
                    u_stat, p_value_mw = mannwhitneyu(group_0, group_1, alternative='two-sided')
                except:
                    p_value_mw = 1.0
            else:
                p_value_mw = 1.0
            
            # Method 3: ROC curve and AUC
            try:
                fpr, tpr, _ = roc_curve(y, X)
                auc_value = auc(fpr, tpr)
            except:
                auc_value = 0.5
            
            # Calculate mean difference between groups
            mean_diff = 0
            if len(group_0) > 0 and len(group_1) > 0:
                mean_diff = group_1.mean() - group_0.mean()
            
            # Method 4: t-test
            if len(group_0) > 1 and len(group_1) > 1:
                try:
                    t_stat, p_value_t = ttest_ind(group_1, group_0, equal_var=False)
                except:
                    p_value_t = 1.0
                    t_stat = 0
            else:
                p_value_t = 1.0
                t_stat = 0
            
            # Store results
            results.append({
                'Feature': cluster,
                'Feature_Type': 'cluster',
                'P_Value_Logit': p_value_logit,
                'P_Value_MannWhitney': p_value_mw,
                'P_Value_TTest': p_value_t,
                'P_Value_Min': min(p_value_logit, p_value_mw, p_value_t),  # Use minimum p-value
                'Odds_Ratio': odds_ratio,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'AUC': auc_value,
                'Mean_Diff': mean_diff,
                'Mean_Group_0': group_0.mean() if len(group_0) > 0 else 0,
                'Mean_Group_1': group_1.mean() if len(group_1) > 0 else 0,
                'T_statistic': t_stat,
                'N_Samples': len(merged_df),
                'N_Group_0': len(group_0),
                'N_Group_1': len(group_1)
            })
            
        except Exception as e:
            print(f"Error analyzing cluster {cluster}: {e}")
    
    # Then, analyze meta-features
    meta_feature_cols = [col for col in meta_features.columns if col not in cluster_columns]
    for feature in meta_feature_cols:
        try:
            X = merged_meta_df[feature]
            y = merged_meta_df['target']
            
            # Skip features with zero variance
            if X.std() == 0:
                continue
            
            # Logistic regression
            X_sm = sm.add_constant(X)
            try:
                model = sm.Logit(y, X_sm).fit(disp=0)
                p_value_logit = model.pvalues[feature]
                odds_ratio = np.exp(model.params[feature])
                ci_lower = np.exp(model.conf_int().loc[feature, 0])
                ci_upper = np.exp(model.conf_int().loc[feature, 1])
            except:
                p_value_logit = 1.0
                odds_ratio = 1.0
                ci_lower = 0.0
                ci_upper = 0.0
            
            # Mann-Whitney U test
            group_0 = X[y == 0]
            group_1 = X[y == 1]
            if len(group_0) > 0 and len(group_1) > 0:
                try:
                    u_stat, p_value_mw = mannwhitneyu(group_0, group_1, alternative='two-sided')
                except:
                    p_value_mw = 1.0
            else:
                p_value_mw = 1.0
            
            # ROC curve and AUC
            try:
                fpr, tpr, _ = roc_curve(y, X)
                auc_value = auc(fpr, tpr)
            except:
                auc_value = 0.5
            
            # Calculate mean difference between groups
            mean_diff = 0
            if len(group_0) > 0 and len(group_1) > 0:
                mean_diff = group_1.mean() - group_0.mean()
            
            # t-test
            if len(group_0) > 1 and len(group_1) > 1:
                try:
                    t_stat, p_value_t = ttest_ind(group_1, group_0, equal_var=False)
                except:
                    p_value_t = 1.0
                    t_stat = 0
            else:
                p_value_t = 1.0
                t_stat = 0
            
            # Only add significant meta-features
            min_p = min(p_value_logit, p_value_mw, p_value_t)
            if min_p < 0.1 or auc_value > 0.65:  # Only keep potentially meaningful meta-features
                results.append({
                    'Feature': feature,
                    'Feature_Type': 'meta',
                    'P_Value_Logit': p_value_logit,
                    'P_Value_MannWhitney': p_value_mw,
                    'P_Value_TTest': p_value_t,
                    'P_Value_Min': min_p,
                    'Odds_Ratio': odds_ratio,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'AUC': auc_value,
                    'Mean_Diff': mean_diff,
                    'Mean_Group_0': group_0.mean() if len(group_0) > 0 else 0,
                    'Mean_Group_1': group_1.mean() if len(group_1) > 0 else 0,
                    'T_statistic': t_stat,
                    'N_Samples': len(merged_meta_df),
                    'N_Group_0': len(group_0),
                    'N_Group_1': len(group_1)
                })
        except Exception as e:
            print(f"Error analyzing meta-feature {feature}: {e}")
    
    # Convert to DataFrame and sort by p-value
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('P_Value_Min')
    
    # Machine learning model analysis
    try:
        # Prepare data
        X_ml = merged_df[cluster_columns]
        y_ml = merged_df['target']
        
        # Random Forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_ml, y_ml)
        
        # Get feature importance
        feature_importances = pd.DataFrame({
            'Feature': X_ml.columns,
            'RF_Importance': rf.feature_importances_
        })
        
        # Merge feature importance into results
        if not results_df.empty:
            results_df = results_df.merge(
                feature_importances, 
                on='Feature', 
                how='left'
            )
        
        # Add Random Forest overall performance
        try:
            cv = StratifiedKFold(n_splits=min(5, len(y_ml)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(rf, X_ml, y_ml, cv=cv, scoring='roc_auc')
            print(f"Random Forest cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as e:
            print(f"Random Forest cross-validation error: {e}")
    
    except Exception as e:
        print(f"Machine learning analysis error: {e}")
    
    if not results_df.empty:
        # 对三种原始 P 值分别进行 FDR 校正（Benjamini-Hochberg）
        for col in ["P_Value_Logit", "P_Value_MannWhitney"]:#, "P_Value_TTest"
            reject, pvals_corrected, _, _ = multipletests(results_df[col], method="fdr_bh")
            results_df[f"{col}_adj"] = pvals_corrected
        
        # 对 Min P 做额外校正
        reject, pvals_corrected_min, _, _ = multipletests(results_df["P_Value_Min"], method="fdr_bh")
        results_df["P_Value_Min_adj"] = pvals_corrected_min

        # Optional: 重新排序
        results_df = results_df.sort_values("P_Value_Min_adj") 
    return results_df

def create_dotplot(results_df, tumor_type, clinical_var, output_path, cluster_to_pathotype):
    """
    Dotplot: each point represents a cluster-clinical correlation
    Color = direction (red: group1 > group0, blue: group0 > group1)
    Size = -log10(FDR-adjusted p)
    """
    # Prepare annotation color map
    annotation_colors = {
        "Adipose tissue": "#fdb863",
        "Debris": "#e66101",
        "Immune cells": "#5e3c99",
        "Normal parenchyma": "#b2abd2",
        "Other pathological conditions": "#80cdc1",
        "Stroma": "#018571",
        "Tumor": "#d7191c"
    }

    # Make sure necessary columns exist
    if "P_Value_Min_adj" not in results_df.columns:
        print("❌ 'P_Value_Min_adj' not found in results_df.")
        return

    # Basic cluster rows only
    cluster_results = results_df.copy()
    if "Feature_Type" in cluster_results.columns:
        cluster_results = cluster_results[cluster_results["Feature_Type"] == "cluster"].copy()

    if len(cluster_results) == 0:
        print("No valid cluster results to plot.")
        return

    # Compute dot size and direction
    cluster_results["neg_log10_fdr"] = -np.log10(cluster_results["P_Value_Min_adj"] + 1e-10)
    cluster_results["direction"] = np.where(
        cluster_results["Mean_1"] > cluster_results["Mean_0"], "Positive", "Negative"
    )
    cluster_results["color"] = cluster_results["Feature"].astype(str).map(
        lambda c: annotation_colors.get(cluster_to_pathotype.get(c, "Other pathological conditions"), "gray")
    )

    # Plotting
    plt.figure(figsize=(10, max(6, len(cluster_results) * 0.35)))
    for i, row in cluster_results.iterrows():
        x = 0  # Only one clinical variable
        y = row["Feature"]
        size = row["neg_log10_fdr"] * 60
        color = row["color"]
        edge_color = 'red' if row["direction"] == "Positive" else 'blue'

        plt.scatter(x, y, s=size, c=color, edgecolors=edge_color, linewidths=1.5)

    # Axis and labels
    plt.xticks([0], [clinical_var], fontsize=12)
    plt.yticks(range(len(cluster_results)), [f"Cluster {c}" for c in cluster_results["Feature"]], fontsize=10)
    plt.ylabel("Cluster", fontsize=12)
    plt.title(f"{tumor_type} - Correlation with {clinical_var}", fontsize=14)

    # Legend for direction
    legend_elements = [
        Patch(facecolor="white", edgecolor="red", label="Positive Enrichment"),
        Patch(facecolor="white", edgecolor="blue", label="Negative Enrichment"),
    ]
    legend_elements += [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in annotation_colors.items()
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Direction & Pathotype")
    plt.xlim(-0.5, 0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Dotplot saved to: {output_path}")

def create_combined_report():
    """Create summary report for all analyses"""
    print("\nCreating summary report...")
    
    # Create output directory
    output_dir = os.path.join(correlation_results_path, 'Summary')
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all result files
    all_results = []
    
    # Iterate through all result directories
    for tumor_type in ['OC']:
        # Recurrence correlation
        recurrence_path = os.path.join(correlation_results_path, tumor_type, 'Recurrence', f'{tumor_type}_recurrence_correlation.csv')
        if os.path.exists(recurrence_path):
            try:
                recurrence_df = pd.read_csv(recurrence_path)
                # Only keep basic clusters
                recurrence_df = recurrence_df[recurrence_df['Feature_Type'] == 'cluster']
                recurrence_df['Tumor_Type'] = tumor_type
                recurrence_df['Clinical_Variable'] = 'Recurrence_Status'
                all_results.append(recurrence_df)
            except Exception as e:
                print(f"Error reading {recurrence_path}: {e}")
        
        # OC platinum resistance correlation
        if tumor_type == 'OC':
            platinum_path = os.path.join(correlation_results_path, tumor_type, 'Platinum', 'OC_platinum_correlation.csv')
            if os.path.exists(platinum_path):
                try:
                    platinum_df = pd.read_csv(platinum_path)
                    # Only keep basic clusters
                    platinum_df = platinum_df[platinum_df['Feature_Type'] == 'cluster']
                    platinum_df['Tumor_Type'] = 'OC'
                    platinum_df['Clinical_Variable'] = 'Platinum_Resistance'
                    all_results.append(platinum_df)
                except Exception as e:
                    print(f"Error reading {platinum_path}: {e}")
    
    if not all_results:
        print("No result files found for creating summary report")
        return
    
    # Merge all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Add significance indicator
    combined_df['Significant'] = combined_df['P_Value_Min'] < 0.05
    
    # Save to CSV
    summary_path = os.path.join(output_dir, 'all_results_summary.csv')
    combined_df.to_csv(summary_path, index=False)
    print(f"Summary report saved to: {summary_path}")
    
    # Create summary plot
    create_summary_plot(combined_df, output_dir)

def create_summary_plot(results_df, output_dir):
    """Create summary plot showing significant clusters across all tumor types"""
    # Filter significant results
    sig_results = results_df[results_df['P_Value_Min'] < 0.1].copy()
    
    if len(sig_results) == 0:
        print("No significant results to plot in summary")
        return
    
    # Create category variable for clinical variables
    sig_results['Clinical_Label'] = sig_results['Clinical_Variable'].map({
        'Recurrence_Status': 'Recurrence',
        'Platinum_Resistance': 'Platinum Resistance'
    })
    
    # Create combined label
    sig_results['Combined_Label'] = sig_results['Tumor_Type'] + ' - ' + sig_results['Clinical_Label']
    
    # Sort by tumor type and clinical variable
    sig_results = sig_results.sort_values(['Tumor_Type', 'Clinical_Variable', 'P_Value_Min'])
    
    # Take top 5 most significant clusters for each combination
    top_results = []
    for label in sig_results['Combined_Label'].unique():
        group_df = sig_results[sig_results['Combined_Label'] == label]
        top_results.append(group_df.head(min(5, len(group_df))))
    
    sig_results = pd.concat(top_results, ignore_index=True)
    
    # Create plot
    plt.figure(figsize=(12, max(6, len(sig_results) * 0.3)))
    
    # Get unique combinations of tumor type and clinical variable
    combinations = sig_results['Combined_Label'].unique()
    
    # Create color mapping for these combinations
    cmap = plt.cm.get_cmap('tab10', len(combinations))
    color_dict = {combo: cmap(i) for i, combo in enumerate(combinations)}
    
    # Plot each result
    for i, (_, row) in enumerate(sig_results.iterrows()):
        combo = row['Combined_Label']
        color = color_dict[combo]
        
        # Plot confidence interval
        plt.plot(
            [max(0.01, row['CI_Lower']), min(20, row['CI_Upper'])], 
            [i, i], 
            '-',
            color=color,
            alpha=0.6
        )
        
        # Plot odds ratio
        plt.plot(
            row['Odds_Ratio'], 
            i, 
            'o', 
            color=color, 
            markersize=8
        )
        
        # Add p-value
        plt.text(
            min(20, row['CI_Upper']) + 0.5, 
            i, 
            f"p={row['P_Value_Min']:.3f}", 
            va='center',
            fontsize=8
        )
    
    # Add reference line (OR=1)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # Add y-axis tick labels
    y_labels = [f"{row['Tumor_Type']} - {row['Clinical_Label']} - Cluster {row['Feature']}" 
                for _, row in sig_results.iterrows()]
    plt.yticks(range(len(sig_results)), y_labels)
    
    # Customize plot
    plt.xlabel('Odds Ratio (95% Confidence Interval)')
    plt.title('Significant Cluster Correlations Across All Tumor Types')
    plt.grid(axis='x', alpha=0.3)
    
    # Set x-axis limits
    plt.xlim(0, min(20, sig_results['CI_Upper'].max() * 1.1))
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=color, label=combo)
        for combo, color in color_dict.items()
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save image
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'significant_associations_summary.png'), dpi=300)
    plt.close()
    
    print(f"Summary plot saved to: {os.path.join(output_dir, 'significant_associations_summary.png')}")
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def plot_cluster_annotation_enrichment_heatmap(
    h5ad_path="oc_with_kmeans_and_annotation_tumoronly.h5ad",
    output_path="fig/tumor_cluster_enrichment_heatmap.pdf",
    cluster_col="hc_clusters",
    annotation_col="Pathotype",
    figsize=(15, 3),
    fdr_cutoff=0.1
):
    # === Step 1: Load Data ===
    adata = sc.read_h5ad(h5ad_path)
    adata.obs[cluster_col] = adata.obs[cluster_col].astype(str)
    crosstab = pd.crosstab(adata.obs[annotation_col], adata.obs[cluster_col])
    pathotypes = crosstab.index
    clusters = crosstab.columns

    # === Step 2: Fisher's Exact Test ===
    results = []
    for cluster in clusters:
        for anno in pathotypes:
            a = crosstab.loc[anno, cluster]
            b = crosstab.loc[anno].sum() - a
            c = crosstab.loc[:, cluster].sum() - a
            d = crosstab.values.sum() - (a + b + c)
            try:
                odds, pval = fisher_exact([[a, b], [c, d]])
            except:
                odds, pval = np.nan, 1.0
            results.append({
                "cluster": cluster,
                "anno": anno,
                "odds_ratio": odds,
                "pval": pval,
                "a": a
            })

    result_df = pd.DataFrame(results)
    result_df["fdr"] = multipletests(result_df["pval"], method="fdr_bh")[1]
    result_df["log2_OR"] = np.log2(result_df["odds_ratio"] + 1e-10)
    result_df["logFDR"] = -np.log10(result_df["fdr"] + 1e-10)
    result_df["mask_na"] = (result_df["a"] == 0)
    result_df.loc[result_df["mask_na"], "log2_OR"] = np.nan
    result_df.loc[result_df["mask_na"], "fdr"] = np.nan

    # === Step 3: Pivot Tables ===
    heatmap_data = result_df.pivot(index='anno', columns='cluster', values='log2_OR')
    fdr_matrix = result_df.pivot(index='anno', columns='cluster', values='fdr')
    heatmap_data = heatmap_data.sort_index()
    fdr_matrix = fdr_matrix.loc[heatmap_data.index, heatmap_data.columns]

    # === Step 4: Plot Heatmap ===
    mask = heatmap_data.isna()
    cmap = sns.color_palette("bwr", as_cmap=True)
    cmap.set_bad("lightgray")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        center=0,
        linewidths=0.3,
        cbar_kws={"label": "log2(Odds Ratio)"},
        mask=mask,
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    # Add × for FDR ≥ fdr_cutoff
    for i, row in enumerate(heatmap_data.index):
        for j, col in enumerate(heatmap_data.columns):
            fdr_val = fdr_matrix.loc[row, col]
            if pd.isna(fdr_val):
                continue
            if fdr_val >= fdr_cutoff:
                x0, y0 = j, i
                x1, y1 = j + 1, i + 1
                ax.plot([x0, x1], [y0, y1], color='black', linewidth=0.5)
                ax.plot([x0, x1], [y1, y0], color='black', linewidth=0.5)

    ax.set_title("Cluster vs Annotation Enrichment\nColor: log2(OR), '×' = FDR ≥ {:.2f}".format(fdr_cutoff))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Annotation")
    plt.tight_layout()
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    """Main function: perform all correlation analyses"""
    summary_records = []
    for n_clusters in range(70, 1, -1):
        try:
            print("Starting enhanced correlation analysis...",str(n_clusters))
            correlation_results_path = f'/lustre1/zxzeng/bwqin/SQUALL_main/clustering/OV/hc_enrichment/correlation_results_tumoronly_hc{n_clusters}'
            os.makedirs(correlation_results_path, exist_ok=True)
            adata, clinical_df, cluster_props = load_oc_data(n_clusters = n_clusters)
            #result_df = analyze_correlation_enhanced(cluster_props, clinical_df,'Recurrence_Status')
            #result_df = analyze_correlation_enhanced(cluster_props, clinical_df,'Platinum_Resistance')
            result_df = analyze_correlation_enhanced(cluster_props, clinical_df,'Survival_term')
            result_df =result_df[result_df["Feature_Type"]=="cluster"]
            if not result_df.empty and (result_df['P_Value_Min_adj'] < 0.05).any():
                print(f"✅ Significant result found with {n_clusters} clusters using analyze_correlation_enhanced")
            min_p_enhanced = result_df['P_Value_Min_adj'].min() if not result_df.empty else np.nan
            #result_df = analyze_cluster_vs_recurrence(cluster_props, clinical_df,'Recurrence_Status')
            result_df = analyze_cluster_vs_recurrence(cluster_props, clinical_df,'Survival_term')
            result_df =result_df[result_df["Feature_Type"]=="cluster"]
            #result_df = analyze_cluster_vs_recurrence(cluster_props, clinical_df,'Platinum_Resistance')
            if not result_df.empty and (result_df['P_Value_Min_adj'] < 0.05).any():
                print(f"✅ Significant result found with {n_clusters} clusters using analyze_cluster_vs_recurrence")
            # Analyze OC tumor platinum resistance correlation
            min_p_recurrence = result_df['P_Value_Min_adj'].min() if not result_df.empty else np.nan
            summary_records.append({
                'n_clusters': n_clusters,
                'min_p_enhanced': min_p_enhanced,
                'min_p_recurrence': min_p_recurrence
            })
            #analyze_tumor_recurrence("OC",n_clusters=n_clusters,correlation_results_path = correlation_results_path)
            #analyze_oc_platinum_resistance(n_clusters=n_clusters,correlation_results_path=correlation_results_path)
            # # Perform stratified analysis (if enough samples)
            # for tumor_type in ['CC', 'EC', 'OC']:
            #     stratified_analysis(tumor_type, 'Recurrence_Status')
            
            # # Perform platinum resistance stratified analysis for OC
            # stratified_analysis('OC', 'Platinum_Resistance')
            
            # Create summary report
            #create_combined_report(cluster_to_pathotype)
            
            print("\nCorrelation analysis complete. Results saved to:", correlation_results_path)
        
        except Exception as e:
            summary_records.append({
                'n_clusters': n_clusters,
                'min_p_enhanced': np.nan,
                'min_p_recurrence': np.nan
            })
            print(f"Error in correlation analysis: {e}")
            import traceback
            traceback.print_exc()
    # 保存结果表格
    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values('n_clusters', ascending=False)
    #summary_df.to_csv("correlation_hc_pval_summary_by_n_clusters.csv", index=False)
    summary_df.to_csv("correlation_hc_platium_pval_summary_by_n_clusters_survival_term.csv", index=False)
    print("📄 Summary saved to correlation_pval_summary_by_n_clusters.csv")

if __name__ == "__main__":
    main() 


