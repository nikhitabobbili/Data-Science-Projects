import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle

with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

def user_input_parameters():
  MntWines=st.sidebar.number_input('Amount spent on wines',min_value=0)
  MntFruits=st.sidebar.number_input('Amount spent on fruits',min_value=0)
  MntMeatProducts=st.sidebar.number_input('Amount spent on meat products',min_value=0)
  MntFishProducts=st.sidebar.number_input('Amount spent on fish products',min_value=0)
  MntSweetProducts=st.sidebar.number_input('Amount spent on sweet products',min_value=0)
  MntGoldProds=st.sidebar.number_input('Amount spent on gold products',min_value=0)
  NumDealsPurchases=st.sidebar.number_input('Number of deal purchases',min_value=0)
  NumWebPurchases=st.sidebar.number_input('Number of Web purchases',min_value=0)
  NumStorePurchases=st.sidebar.number_input('Number of Store purchases',min_value=0)
  NumCatalogPurchases=st.sidebar.number_input('Number of Catalog purchases',min_value=0)
  NumWebVisitsMonth=st.sidebar.number_input('Number of visits to company’s website in the last month',min_value=0)


  data={'MntWines':MntWines,'MntFruits':MntFruits,
        'MntMeatProducts':MntMeatProducts,'MntFishProducts':MntFishProducts,'MntSweetProducts':MntSweetProducts,
        'MntGoldProds':MntGoldProds,'NumDealsPurchases':NumDealsPurchases,'NumWebPurchases':NumWebPurchases,
        'NumCatalogPurchases':NumCatalogPurchases,'NumStorePurchases':NumStorePurchases,
        'NumWebVisitsMonth':NumWebVisitsMonth }
  features=pd.DataFrame(data,index=[0])
  return features

df1=user_input_parameters()
scaled_all = scaler.transform(df1)
pca_all = pca.transform(scaled_all)
cluster_labels = loaded_model.predict(pca_all)

df_clustered = df1.copy()
df_clustered['Cluster'] = cluster_labels
cluster_averages = df_clustered.groupby('Cluster').mean()

st.title('Customer Segmentation based on purchasing behavoiur')
if st.button('Predict'):
    # Predict cluster for user's input
    predicted_cluster = int(cluster_labels[0])

    # Display the result
    st.write(f"The predicted cluster for this customer is: {predicted_cluster}")
    if predicted_cluster==0:
      st.success('Low spender')
    else:
      st.success('High Spender')

    # Generate dummy data for visualization
    dummy_data = np.random.randn(100, df1.shape[1]) * 10 + df1.mean().values
    dummy_scaled = scaler.transform(dummy_data)
    dummy_pca = pca.transform(dummy_scaled)
    dummy_clusters = loaded_model.predict(dummy_pca)
    centers = loaded_model.cluster_centers_

    #Scatter plot of customer clusters
    st.subheader("Customer Cluster Visualization")
    fig1 = plt.figure(figsize=(7, 5))
    plt.scatter(dummy_pca[:, 0], dummy_pca[:, 1], c=dummy_clusters, cmap='viridis', alpha=0.5)
    plt.scatter(pca_all[0, 0], pca_all[0, 1], color='black', marker='*', s=200, label='Your Customer')
    plt.scatter(centers[:, 0], centers[:, 1], color='red', marker='X', s=150, label='Cluster Centers')
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Customer Cluster Visualization")
    plt.legend()
    st.pyplot(fig1)

# Visualization
spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
spending_df = df1[spending_cols].T  # Transpose for vertical bar chart
spending_df.columns = ['Amount']
spending_df = spending_df.reset_index().rename(columns={'index': 'Product'})

st.bar_chart(spending_df.set_index('Product'))
