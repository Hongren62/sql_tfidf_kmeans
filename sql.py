import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

db = mysql.connector.connect(
    host="localhost",
    port=3306,
    database="mydb",
    user="root",
    password="root"
)

mycursor = db.cursor()

# tf-idf
mycursor.execute("SELECT `Abstract (English)` FROM `f01l_patent-2023`")

# 獲取所有資料
data = mycursor.fetchall()

# 將資料轉換為 list
text_data = [row[0] for row in data]

mycursor.close()

# 使用 TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(), token_pattern=None)
tfidf_matrix = vectorizer.fit_transform(text_data)


print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())

# Kmeans
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
kmeans.fit(tfidf_matrix)

# 獲取每條文本所屬的群集標籤
cluster_labels = kmeans.labels_

# 使用 PCA 將 TF-IDF 矩陣降維到二維
pca = PCA(n_components=2)
coordinates = pca.fit_transform(tfidf_matrix.toarray())

# 繪製散點圖，根據 K-means 群集標籤上色
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=cluster_labels, cmap='viridis', s=50)

# 繪製 K-means 中心點，用 "*" 標記
centers = kmeans.cluster_centers_[:, :2]
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, color='red')

# 添加標籤和標題
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('K-means Clustering with TF-IDF')

# 顯示圖形
plt.show()
