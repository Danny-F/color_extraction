

from matplotlib import pyplot as plt
from skimage import io
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab
from skimage.color import lab2rgb
import streamlit as st
import itertools


############### functions ###############

#streamlit sidebar variables
def collect_user_entered_variables():
    st.title('Extract Colors From An Image')
    image_link = st.sidebar.text_input('Image Link', 'paste image link here')
    inertia_change_threshold = st.sidebar.slider('Inertia % Change Threshold', -100, -1, -25)
    ignore_background = st.sidebar.checkbox('Ignore background color?', value=True)
    user_variables = {'image_link': image_link,
                            'inertia_change_threshold': inertia_change_threshold,
                            'ignore_background': ignore_background}
    return user_variables


#display original image
def display_original_image(image_link):
    st.image(image_link, caption = 'normal image')


#reading in image and generating l*a*b version as well
def import_image(image_link):
    image_rgb = io.imread(image_link)
    image_cie = rgb2lab(image_rgb)
    image_shape = image_rgb.shape
    return image_rgb, image_cie, image_shape

#create df that has the cie and rgb values for each pixel
def format_image_data_into_dataframe(image_rgb, image_cie, ignore_background):
    rgb_list = []
    cie_list = []
    for x,_ in enumerate(image_rgb):
        for y,_ in enumerate(_):
            rgb_list.append(image_rgb[x][y])
            cie_list.append(image_cie[x][y])
    rgb_df = pd.DataFrame(rgb_list, columns=['r', 'g', 'b'])
    cie_df = pd.DataFrame(cie_list, columns=['c', 'i', 'e'])
    rgb_df['rgb'] = rgb_list
    cie_df['cie'] = cie_list
    df = rgb_df.join(cie_df)
    df['rgb_comb'] = df['r']*1000000 + df['g']*1000 + df['b']
    background = df['rgb_comb'].value_counts().index[0]
    if ignore_background == False:
        background = 100000000000 # this won't match to any RGB combo
    no_background_index = df['rgb_comb']!=background
    no_background_df = df[no_background_index].copy()
    background_df = df[~no_background_index].copy()
    print(background_df.shape)
    return no_background_df, background_df

#determing the ideal cluster number for clustering pixels
def determine_ideal_cluster_number(image_data_df, inertia_change_threshold):
    chart = st.line_chart()
    i=0
    large_change = True
    prev_inertia = 1e32
    inertias = []
    while large_change == True:
        i += 1
        km = MiniBatchKMeans(n_clusters = i)
        km = km.fit(image_data_df[['c', 'i', 'e']])
        new_inertia = km.inertia_
        change = (new_inertia-prev_inertia)/prev_inertia
        large_change = change < inertia_change_threshold/100
        prev_inertia = new_inertia
        chart.add_rows([prev_inertia])
    ideal_cluster_num = i - 1
    return ideal_cluster_num

#assigning pixels to clusters
def assign_clusters(image_data_df, ideal_cluster_num):
    st.write('Ideal cluster number: {}'.format(ideal_cluster_num))
    km = KMeans(n_clusters = ideal_cluster_num)
    km = km.fit(image_data_df[['c', 'i', 'e']])
    return km.labels_

#average data by cluster cluster
def avg_data_by_cluster(image_data_df):
    cluster_grouped_df = image_data_df.groupby('cluster')[['r', 'g', 'b']].mean().round(0).astype(int)
    cluster_grouped_df['avg_rgb'] = cluster_grouped_df[['r', 'g', 'b']].apply(lambda row: np.asarray([row[0], row[1], row[2]], dtype='uint8'), axis=1)
    cluster_grouped_df['rgb_plt_color'] = cluster_grouped_df[['r', 'g', 'b']].apply(lambda row: [row[0]/255, row[1]/255, row[2]/255], axis=1)
    cluster_grouped_df['count'] = image_data_df['cluster'].value_counts()
    cluster_grouped_df['percent'] = round(cluster_grouped_df['count']/ cluster_grouped_df['count'].sum(),2)
    cluster_grouped_df = cluster_grouped_df.sort_values(['count'], ascending = False).reset_index()
    return cluster_grouped_df

#displaying the pie chart of the clustered colors
def display_color_pie_graph(cluster_grouped_df):
    plt.pie(cluster_grouped_df['count'],
            colors=cluster_grouped_df['rgb_plt_color'].tolist()
            )
    st.pyplot()

#joining the avgeraged rgb values back into the full list of pixels
def join_cluster_data_to_full_image_data(image_data_df, background_df, cluster_grouped_df):
    full_image_data_df = pd.concat([image_data_df,background_df], sort=True).sort_index()
    full_image_data_df = full_image_data_df.reset_index()
    full_image_data_df = pd.merge(full_image_data_df, cluster_grouped_df[['cluster','avg_rgb']], how='left', on='cluster')
    full_image_data_df.set_index(full_image_data_df['index'], inplace=True)
    #for the background color pixels, assign their rgb value to the avg_rgb since they are equal
    full_image_data_df['avg_rgb'] = np.where(full_image_data_df['avg_rgb'].isnull(), full_image_data_df['rgb'], full_image_data_df['avg_rgb'])
    return full_image_data_df

#displaying the background color that was ignored, if "Ignore background color" was checked
def display_background_color(background_df):
    bg_rgb = background_df.iloc[0]['rgb']
    bg_image = np.asarray(list(itertools.repeat(bg_rgb, 2000))).reshape(20,100,3)
    st.write('Background Color')
    st.image(bg_image)

#create and display the new image with the averaged colors
def display_image_with_clustered_colors(full_image_data_df, image_shape):
    print(image_shape)
    image_avg_rgb = np.asarray(full_image_data_df['avg_rgb'].tolist()).reshape(image_shape)
    st.image(image_avg_rgb, caption = 'clustered colors')
