#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:22:32 2020

@author: dannyfarrington
"""


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

#streamlit sidebar variables
st.title('Extract Colors From An Image')
image_link = st.sidebar.text_input('Image Link', 'paste image link here')
inertia_change_threshold = st.sidebar.slider('Inertia % Change Threshold', -100, -1, -25)
ignore_background = st.sidebar.checkbox('Ignore background color?', value=True)

if image_link != 'paste image link here':

    st.image(image_link, caption = 'normal image')

    #reading in image and generating l*a*b version as well
    image_rgb = io.imread(image_link)
    image_cie = rgb2lab(image_rgb)
    image_shape = image_rgb.shape


    #creating df that has the cie and rgb values for each pixel
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
        background = 100000000000 # this won't make to any RGB combo
    nobg_index = df['rgb_comb']!=background
    nobg_df = df[nobg_index].copy()


    #finding the ideal cluster number
    chart = st.line_chart()
    i=0
    large_change = True
    prev_inertia = 1e32
    inertias = []
    while large_change == True:
        i += 1
        km = MiniBatchKMeans(n_clusters = i)
        km = km.fit(nobg_df[['c', 'i', 'e']])
        new_inertia = km.inertia_
        change = (new_inertia-prev_inertia)/prev_inertia
        large_change = change < inertia_change_threshold/100
        prev_inertia = new_inertia
        chart.add_rows([prev_inertia])

    #using the ideal cluster number to find the individual color info
    ideal_cluster_num = i-1
    st.write('Ideal cluster number: {}'.format(ideal_cluster_num))
    km = KMeans(n_clusters = ideal_cluster_num)
    km = km.fit(nobg_df[['c', 'i', 'e']])
    nobg_df['label'] = km.labels_

    label_grouped_df = nobg_df.groupby('label')[['r', 'g', 'b']].mean().round(0).astype(int)
    label_grouped_df['avg_rgb'] = label_grouped_df[['r', 'g', 'b']].apply(lambda row: np.asarray([row[0], row[1], row[2]], dtype='uint8'), axis=1)
    label_grouped_df['rgb_plt_color'] = label_grouped_df[['r', 'g', 'b']].apply(lambda row: [row[0]/255, row[1]/255, row[2]/255], axis=1)
    label_grouped_df['count'] = nobg_df['label'].value_counts()
    label_grouped_df['percent'] = round(label_grouped_df['count']/ label_grouped_df['count'].sum(),2)
    label_grouped_df = label_grouped_df.sort_values(['count'], ascending = False).reset_index()


    # displaying the pie chart of the extracted colors
    plt.pie(label_grouped_df['count'],
            colors=label_grouped_df['rgb_plt_color'].tolist()
            )
    st.pyplot()

    #joining the avgeraged rgb values back into the full list of pixels
    nobg_df = nobg_df.reset_index()
    nobg_df = pd.merge(nobg_df, label_grouped_df[['label','avg_rgb']], how='left', on='label')
    nobg_df.set_index(nobg_df['index'], inplace=True)
    df['avg_rgb'] = nobg_df['avg_rgb']
    df.loc[~nobg_index, 'avg_rgb'] = df.loc[~nobg_index, 'rgb']

    #creating the new image with the averaged colors
    image_avg_rgb = np.asarray(df['avg_rgb'].tolist()).reshape(image_shape)

    # displaying the background color that was ignored, if "Ignore background color" was checked
    if ignore_background == True:
        bg_rgb = df[nobg_index==False].iloc[0]['rgb']
        image_bg = np.asarray(list(itertools.repeat(bg_rgb, 2000))).reshape(20,100,3)
        st.write('Background Color')
        st.image(image_bg)

    #dsiplaying the image with the averaged colors
    st.image(image_avg_rgb, caption = 'clustered colors')

else:
    st.write('Past an image link (no quotes needed) into the image link field in the sidebar.')
