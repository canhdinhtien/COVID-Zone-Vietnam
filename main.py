import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('D:/Code/python/Covid19/Thong_tin_covid.csv', encoding='utf-8')
print(df)
# Ensure all values are strings, then remove commas and convert columns to numeric
df['Infected Cases'] = pd.to_numeric(df['Infected Cases'].astype(str).str.replace(',', ''), errors='coerce')
df['Deaths'] = pd.to_numeric(df['Deaths'].astype(str).str.replace(',', ''), errors='coerce')
df['New Cases'] = pd.to_numeric(df['New Cases'].astype(str).str.replace(',', ''), errors='coerce')

# Drop rows with NaN values and reset index
df = df.dropna()
df = df.reset_index(drop=True)


# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with specified axes
scatter = ax.scatter(df['Infected Cases'], df['Deaths'], df['New Cases'], c=df['Infected Cases'], cmap='viridis')
ax.set_xlabel('Infected Cases')
ax.set_ylabel('Deaths')
ax.set_zlabel('New Cases')
plt.colorbar(scatter)
plt.title('3D Scatter Plot of COVID-19 Data')

# # Prepare data for KMeans
X = df[['Infected Cases', 'Deaths', 'New Cases']].values

# # Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Calculate SSE for different values of k
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    sse.append(km.inertia_)

# # Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_rng, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')

# # Choose the optimal k (let's say it's 4 for this example)
optimal_k = 4

# # Apply KMeans with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Add the cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Create a new column 'Zone' based on the 'Cluster'
df['Zone'] = df['Cluster']

# Select only numeric columns for groupby mean calculation
numeric_columns = ['Infected Cases', 'Deaths', 'New Cases', 'Cluster']
df_numeric = df[numeric_columns]

# Create a 3D plot with clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with clusters
scatter = ax.scatter(df['Infected Cases'], df['Deaths'], df['New Cases'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Infected Cases')
ax.set_ylabel('Deaths')
ax.set_zlabel('New Cases')
plt.colorbar(scatter)
plt.title('3D Scatter Plot of COVID-19 Data with KMeans Clusters')
plt.show()

# Calculate and print the mean of numeric columns for each cluster
print(df_numeric.groupby('Cluster').mean())

coords = {
    'An Giang': (10.5216, 105.1259),
    'Bà Rịa - Vũng Tàu': (10.5417, 107.2429),
    'Bắc Giang': (21.2733, 106.1946),
    'Bắc Kạn': (22.1474, 105.8348),
    'Bạc Liêu': (9.2879, 105.7249),
    'Bắc Ninh': (21.1876, 106.0763),
    'Bến Tre': (10.2411, 106.3754),
    'Bình Định': (14.1665, 108.9027),
    'Bình Dương': (11.3254, 106.4770),
    'Bình Phước': (11.7512, 106.7235),
    'Bình Thuận': (11.0904, 108.0721),
    'Cà Mau': (9.1520, 105.1960),
    'Cần Thơ': (10.0452, 105.7469),
    'Cao Bằng': (22.6657, 106.2570),
    'Đà Nẵng': (16.0471, 108.2068),
    'Đắk Lắk': (12.7100, 108.2378),
    'Đắk Nông': (12.2599, 107.7340),
    'Điện Biên': (21.3860, 103.0230),
    'Đồng Nai': (11.0150, 107.491),
    'Đồng Tháp': (10.5920, 105.6789),
    'Gia Lai': (13.8070, 108.1094),
    'Hà Giang': (22.7671, 104.9680),
    'Hà Nam': (20.5832, 106.0218),
    'Hà Nội': (21.0285, 105.8542),
    'Hà Tĩnh': (18.3535, 105.8880),
    'Hải Dương': (20.9408, 106.3330),
    'Hải Phòng': (20.8449, 106.6881),
    'Hậu Giang': (9.7846, 105.4700),
    'Hoà Bình': (20.6861, 105.3131),
    'Hưng Yên': (20.8526, 106.0161),
    'Khánh Hòa': (12.2388, 109.1967),
    'Kiên Giang': (10.0151, 105.0809),
    'Kon Tum': (14.3502, 107.9841),
    'Lai Châu': (22.3964, 103.4559),
    'Lâm Đồng': (11.5753, 108.1429),
    'Lạng Sơn': (21.8470, 106.7571),
    'Lào Cai': (22.3381, 104.1487),
    'Long An': (10.5431, 106.4113),
    'Nam Định': (20.4388, 106.1621),
    'Nghệ An': (18.8126, 105.2830),
    'Ninh Bình': (20.2534, 105.9744),
    'Ninh Thuận': (11.6739, 108.8622),
    'Phú Thọ': (21.2684, 105.2017),
    'Phú Yên': (13.0955, 109.2927),
    'Quảng Bình': (17.6100, 106.3487),
    'Quảng Nam': (15.5731, 108.4740),
    'Quảng Ngãi': (15.0755, 108.7076),
    'Quảng Ninh': (20.9511, 107.0800),
    'Quảng Trị': (16.7422, 107.1855),
    'Sóc Trăng': (9.6039, 105.9804),
    'Sơn La': (21.3280, 103.9106),
    'Tây Ninh': (11.3355, 106.1055),
    'Thái Bình': (20.4463, 106.3364),
    'Thái Nguyên': (21.5671, 105.8252),
    'Thanh Hóa': (19.8067, 105.7852),
    'Thừa Thiên Huế': (16.4637, 107.5909),
    'Tiền Giang': (10.4490, 106.3424),
    'TP. Hồ Chí Minh': (10.8231, 106.6297),
    'Trà Vinh': (9.9354, 106.3458),
    'Tuyên Quang': (22.1797, 105.2131),
    'Vĩnh Long': (10.2392, 105.9575),
    'Vĩnh Phúc': (21.3016, 105.5963),
    'Yên Bái': (21.7167, 104.8985),
}

# Create a base map centered on Vietnam
map_vietnam = folium.Map(location=[15.775, 107.304], zoom_start=6)

# Define color scheme for zones
zone_colors = {0: 'green', 1: 'orange', 2: 'red', 3: 'purple'}
# Iterate through the dataframe and add markers for each city
for index, row in df.iterrows():
    city = row['City']
    zone = row['Zone']
    if city in coords:
        lat, lon = coords[city]
        color = zone_colors.get(zone, 'gray')  # Default to gray if zone is not 0, 1, 2, or 3
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=f"{city}<br>Zone: {zone}<br>Infected Cases: {row['Infected Cases']}<br>Deaths: {row['Deaths']}<br>New Cases: {row['New Cases']}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(map_vietnam)
    
# Add a legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 120px; height: 110px; 
    border:2px solid grey; z-index:9999; font-size:14px; background-color:white;
    ">&nbsp; Zone Legend <br>
    &nbsp; <i class="fa fa-circle fa-1x" style="color:green"></i> Zone 0 <br>
    &nbsp; <i class="fa fa-circle fa-1x" style="color:orange"></i> Zone 1 <br>
    &nbsp; <i class="fa fa-circle fa-1x" style="color:red"></i> Zone 2 <br>
    &nbsp; <i class="fa fa-circle fa-1x" style="color:purple"></i> Zone 3
</div>
'''
map_vietnam.get_root().html.add_child(folium.Element(legend_html))

# Save the map
map_vietnam.save('vietnam_covid_map.html')

