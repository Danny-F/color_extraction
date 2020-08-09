
from color_extraction_functions import *

############### execution ###############
user_variables = collect_user_entered_variables()
if user_variables['image_link'] != 'paste image link here':
    display_original_image(user_variables['image_link'])
    image_rgb, image_cie, image_shape = import_image(user_variables['image_link'])
    image_data_df, background_df = format_image_data_into_dataframe(image_rgb, image_cie,
                                                                    user_variables['ignore_background'])
    ideal_cluster_num = determine_ideal_cluster_number(image_data_df,
                                                       user_variables['inertia_change_threshold'])
    image_data_df['cluster'] = assign_clusters(image_data_df, ideal_cluster_num)
    cluster_grouped_df = avg_data_by_cluster(image_data_df)
    display_color_pie_graph(cluster_grouped_df)
    full_image_data_df = join_cluster_data_to_full_image_data(image_data_df, background_df,
                                                              cluster_grouped_df)
    if user_variables['ignore_background'] == True:
        display_background_color(background_df)
    display_image_with_clustered_colors(full_image_data_df, image_shape)
else:
    st.write('Past an image link (no quotes needed) into the image link field in the sidebar.')
