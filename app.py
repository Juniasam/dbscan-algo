from flask import Flask, render_template, request, flash, session
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO
import base64
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.xls', '.tsv', '.txt']
app.config['UPLOAD_FOLDER'] = './uploads'
df = None


# Preprocessing function (unchanged)
def advanced_preprocess_data(df, columns_to_remove, scaling_method, enable_label_encoding=False, fill_missing_strategy="drop"):
    """
    Preprocess data for clustering, including cleaning invalid symbols and handling missing values.
    """
    INVALID_SYMBOLS = ['?', 'NA', 'N/A', '--', 'missing', 'null', 'NULL']  # Add any other invalid entries here

    data = df.copy()

    # Drop specified columns
    if columns_to_remove:
        data.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    # Replace invalid symbols with NaN
    data.replace(INVALID_SYMBOLS, np.nan, inplace=True)

    # Encode categorical columns if enabled
    if enable_label_encoding:
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Convert non-numeric values in numeric columns to NaN
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Handle missing values based on selected strategy
    if fill_missing_strategy == "drop":
        data.dropna(inplace=True)  # Drop rows with missing values
    elif fill_missing_strategy == "mean":
        data.fillna(data.mean(), inplace=True)  # Fill missing numeric values with the column mean
    elif fill_missing_strategy == "median":
        data.fillna(data.median(), inplace=True)  # Fill missing numeric values with the column median
    elif fill_missing_strategy == "mode":
        data.fillna(data.mode().iloc[0], inplace=True)  # Fill missing values with the column mode

    # Scale numeric columns
    if scaling_method != "none" and len(numeric_columns) > 0:
        scaler = None
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()

        if scaler:
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data



@app.route("/", methods=["GET", "POST"])
def main():
    global df
    data_preview, columns = None, []
    preprocess_message = ""
    clustering_message = None
    cluster_plot_url = None
    silhouette_avg = None
    plot_urls = {}
    datatype_info = None
    stats_info = None

    try:
        if request.method == "POST":
            # File upload handling
            if "file" in request.files:
                file = request.files["file"]
                session['filename'] = file.filename
                file_extension = file.filename.split('.')[-1].lower()

                if f".{file_extension}" not in app.config['UPLOAD_EXTENSIONS']:
                    flash("Invalid file type. Please upload a CSV, Excel, TSV, or TXT file.")
                    return render_template("index.html", columns=columns, filename="No file chosen")

                try:
                    if file_extension == "csv":
                        df = pd.read_csv(file)
                    elif file_extension in ["xlsx", "xls"]:
                        df = pd.read_excel(file, engine='openpyxl')
                    elif file_extension in ["tsv", "txt"]:
                        df = pd.read_csv(file, sep='\t')

                    if df.empty:
                        flash("The uploaded file is empty or invalid.")
                        return render_template("index.html", columns=columns, filename="No file chosen")

                    columns = df.columns.tolist()
                    session['columns'] = columns
                    data_preview = df.head(10).to_html(classes="table table-striped", index=False)

                    # Generate datatype and statistical info
                    datatype_info = df.dtypes.to_frame(name="Datatype").reset_index()
                    datatype_info.columns = ["Column", "Datatype"]

                    stats_info = df.describe(include="all").transpose().reset_index()
                    stats_info.columns = ["Column"] + stats_info.columns.tolist()[1:]

                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_data.csv"), index=False)
                except Exception as e:
                    flash(f"Error reading file: {str(e)}")
                    return render_template("index.html", columns=columns, filename="No file chosen")


            # Apply preprocessing
            elif request.form.get("action") == "Apply Advanced Preprocessing":
                if df is None:
                    flash("No dataset found. Please upload a file first.")
                    return render_template("index.html", columns=columns, filename="No file chosen")

                columns_to_remove = request.form.getlist("columns_to_remove")
                scaling_method = request.form.get("scaling_method")
                enable_label_encoding = request.form.get("enable_label_encoding", "off") == "on"
                fill_missing_strategy = request.form.get("fill_missing_strategy", "drop")  # Default is "drop"

                df = advanced_preprocess_data(df, columns_to_remove, scaling_method, enable_label_encoding, fill_missing_strategy)
                preprocess_message = (
                    f"Preprocessing applied with Scaling: {scaling_method}, "
                    f"Columns Removed: {columns_to_remove}, "
                    f"Label Encoding: {'Enabled' if enable_label_encoding else 'Disabled'}, "
                    f"Missing Value Strategy: {fill_missing_strategy.capitalize()}"
                )
                data_preview = df.head(10).to_html(classes="table table-striped", index=False)
                columns = df.columns.tolist()
            

            # Train DBSCAN model
            elif request.form.get("action") == "Train Clustering Model":
                if df is None:
                    flash("No dataset found. Please upload and preprocess the data first.")
                    return render_template("index.html", columns=columns, filename="No file chosen",plot_urls=plot_urls)

                # Ensure the dataset is numeric
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    flash("No numeric data found. Ensure the dataset contains numeric columns for clustering.")
                    return render_template("index.html", columns=session.get('columns', []), filename=session.get('filename', 'No file chosen'))

                try:
                    eps = float(request.form.get("eps", 0.5))
                    min_samples = int(request.form.get("min_samples", 5))

                    # Optionally reduce dimensions for high-dimensional data
                    reduce_dimensions = request.form.get("reduce_dimensions", "false").lower() == "true"
                    use_distance_matrix = request.form.get("use_distance_matrix", "false").lower() == "true"

                    if reduce_dimensions:
                        # Reduce dimensions using PCA to 2 for clustering visualization
                        pca = PCA(n_components=2)
                        numeric_df_reduced = pd.DataFrame(pca.fit_transform(numeric_df), columns=["PC1", "PC2"])
                    else:
                        numeric_df_reduced = numeric_df

                    if use_distance_matrix:
                        # Calculate the pairwise distance matrix
                        distance_matrix = euclidean_distances(numeric_df_reduced)
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
                        cluster_labels = dbscan.fit_predict(distance_matrix)
                    else:
                        # Perform DBSCAN directly on the reduced or original data
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        cluster_labels = dbscan.fit_predict(numeric_df_reduced)

                    # Add cluster labels to the dataset
                    numeric_df['cluster'] = cluster_labels
                    df['cluster'] = cluster_labels  # Add clusters back to the original dataframe

                    # Step 3: Analyze clustering results
                    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    num_noise_points = list(cluster_labels).count(-1)
                    clustering_message = f"DBSCAN found {num_clusters} clusters with {num_noise_points} noise points."

                    # Step 4: Calculate silhouette score if clusters are valid
                    if num_clusters > 1:
                        silhouette_avg = silhouette_score(
                            numeric_df_reduced if reduce_dimensions else numeric_df.drop('cluster', axis=1),
                            cluster_labels
                        )
                        clustering_message += f" Silhouette Score: {silhouette_avg:.2f}"

                    # Flash success message
                    flash(clustering_message)

                    # Plot cluster results
                    plt.figure(figsize=(8, 6))
                    if numeric_df_reduced.shape[1] >= 2:  # Ensure there are at least two dimensions to plot
                        sns.scatterplot(x=numeric_df_reduced.iloc[:, 0], y=numeric_df_reduced.iloc[:, 1], hue=numeric_df['cluster'], palette='viridis', s=50)
                        plt.title("DBSCAN Clustering Results")
                        buf = BytesIO()
                        plt.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        cluster_plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
                        buf.close()
                        plt.close()
                    else:
                        flash("Unable to plot clusters: less than 2 numeric features.")

                    return render_template("index.html", columns=session.get('columns', []), filename=session.get('filename', 'No file chosen'), cluster_plot_url=cluster_plot_url,plot_urls=plot_urls)

                except Exception as e:
                    flash(f"Error training DBSCAN: {str(e)}")
                    return render_template("index.html", columns=session.get('columns', []), filename=session.get('filename', 'No file chosen'),plot_urls=plot_urls)
                    

            elif request.form.get("action") == "Generate Plots":
                scatter_x = request.form.get("scatter_x")
                scatter_y = request.form.get("scatter_y")
                bar_x = request.form.get("bar_x")
                bar_y = request.form.get("bar_y")
                histogram_column = request.form.get("histogram_column")
                boxplot_column = request.form.get("boxplot_column")

                # Scatter Plot
                if scatter_x and scatter_y:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=df, x=scatter_x, y=scatter_y, hue='cluster', palette='viridis')
                    plt.title(f"Scatter Plot: {scatter_x} vs {scatter_y}")
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plot_urls['scatter_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()

                # Bar Plot
                if bar_x and bar_y:
                    plt.figure(figsize=(8, 6))
                    sns.barplot(data=df, x=bar_x, y=bar_y, palette='viridis')
                    plt.title(f"Bar Plot: {bar_x} vs {bar_y}")
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plot_urls['bar_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()

                # Histogram
                if histogram_column:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data=df, x=histogram_column, kde=True, color='blue')
                    plt.title(f"Histogram: {histogram_column}")
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plot_urls['histogram_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()

                # Box Plot
                if boxplot_column:
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(data=df, x='cluster', y=boxplot_column, palette='viridis')
                    plt.title(f"Box Plot: {boxplot_column} by Cluster")
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plot_urls['boxplot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()

                # Add message if plots are generated
                clustering_message = "Plots generated successfully!"
    except Exception as e:
        flash(f"An error occurred: {str(e)}")

    return render_template("index.html",
                    columns=session.get('columns', []),
                    filename=session.get('filename', 'No file chosen'),
                    data_preview=data_preview,
                    preprocess_message=preprocess_message,
                    clustering_message=clustering_message,
                    silhouette_avg=silhouette_avg,
                    cluster_plot_url=cluster_plot_url,
                    plot_urls=plot_urls,
                    datatype_info=datatype_info.to_html(classes="table table-striped", index=False) if datatype_info is not None else None,
        stats_info=stats_info.to_html(classes="table table-striped", index=False) if stats_info is not None else None)
if __name__ == "__main__":
    app.run(debug=True)